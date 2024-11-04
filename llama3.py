from argparse import ArgumentParser
from pathlib import Path

import jax
import jax.numpy as jnp
import tiktoken
from safetensors.flax import load_file
from tiktoken.load import load_tiktoken_bpe

# References
# Meta: https://github.com/meta-llama/llama-models/blob/main/models/llama3/reference_impl/model.py
# Unsloth: https://github.com/unslothai/unsloth/blob/main/unsloth/models/llama.py
# Huggingface: https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py
# Sebastian Raschka: https://github.com/rasbt/LLMs-from-scratch/blob/main/ch05/07_gpt_to_llama/standalone-llama32.ipynb


def rms_norm(x, gamma, eps):
    # https://arxiv.org/pdf/1910.07467
    return (
        gamma
        * x
        * jax.lax.rsqrt(jnp.mean(jnp.square(x), axis=-1, keepdims=True) + eps)
    )


def swish(x, beta: float):
    # https://arxiv.org/pdf/1710.05941v1
    return x * jax.nn.sigmoid(beta * x)


def ffn(x, W1, V, W2):
    # https://arxiv.org/pdf/2002.05202v1
    x = swish(jnp.dot(x, W1.T), beta=1.0) * jnp.dot(x, V.T)
    x = jnp.dot(x, W2.T)
    return x


def rescale_theta(theta_old, ctx_len_old, ctx_len_new):
    scaling_factor = ctx_len_new / ctx_len_old
    theta_new = theta_old * scaling_factor
    return theta_new


def rope_frequencies(dim, ctx_len, theta, dtype=jnp.float32):
    # https://arxiv.org/pdf/2104.09864
    m = jnp.arange(ctx_len, dtype=jnp.float32)
    t = 1.0 / (
        theta ** (jnp.arange(0, dim, 2, dtype=jnp.float32)[: (dim // 2)] / dim)
    )
    f = jnp.einsum("i, j -> ij", m, t)
    f = jnp.concatenate([f, f], axis=-1)
    return jnp.cos(f).astype(dtype), jnp.sin(f).astype(dtype)


def rope(x, f_cos, f_sin):
    # https://arxiv.org/pdf/2104.09864
    x1, x2 = jnp.split(x, 2, axis=-1)
    rotated = jnp.concatenate([-x2, x1], axis=-1)
    x_rotated = x * f_cos + rotated * f_sin
    return x_rotated


def gqa(x, h, h_kv, W_Q, W_K, W_V, W_O, f_cos, f_sin):
    s, _ = x.shape

    # https://arxiv.org/pdf/2305.13245v3
    # Project
    Q = jnp.einsum("ik, jk -> ij", x, W_Q)
    K = jnp.einsum("ik, jk -> ij", x, W_K)
    V = jnp.einsum("ik, jk -> ij", x, W_V)

    # Split along head
    Q = Q.reshape(s, h, -1).transpose(1, 0, 2)
    K = K.reshape(s, h_kv, -1).transpose(1, 0, 2)
    V = V.reshape(s, h_kv, -1).transpose(1, 0, 2)

    # Apply rotary embeddings
    Q = rope(Q, f_cos, f_sin)
    K = rope(K, f_cos, f_sin)

    # Grouped Query / Repeated Key-Value
    K = jnp.repeat(K, h // h_kv, axis=0)
    V = jnp.repeat(V, h // h_kv, axis=0)

    # Query-Key Lookup
    scores = jnp.einsum("hik, hjk -> hij", Q, K)
    scores /= jnp.sqrt(K.shape[-1])

    # Causal mask
    mask = jnp.triu(jnp.full_like(scores, -1e10), k=1)
    scores += mask

    # Softmax
    scores = jax.nn.softmax(scores, axis=-1)

    # Project to value space
    attention = jnp.einsum("hik, hkj -> hij", scores, V)
    attention = attention.transpose(1, 0, 2).reshape(s, -1)

    # Output projection
    out = jnp.einsum("ik, jk -> ij", attention, W_O)
    return out


def xfmr(tokens, w, params, f_cos, f_sin, pos):
    s = len(tokens)

    f_cos, f_sin = f_cos[pos : pos + s], f_sin[pos : pos + s]
    x = w["model.embed_tokens.weight"][jnp.array(tokens)]

    for i in range(params["n"]):
        r = jnp.copy(x)
        x = rms_norm(
            x,
            w[f"model.layers.{i}.input_layernorm.weight"],
            eps=params["norm_eps"],
        )
        x = gqa(
            x,
            h=params["h"],
            h_kv=params["h_kv"],
            W_Q=w[f"model.layers.{i}.self_attn.q_proj.weight"],
            W_K=w[f"model.layers.{i}.self_attn.k_proj.weight"],
            W_V=w[f"model.layers.{i}.self_attn.v_proj.weight"],
            W_O=w[f"model.layers.{i}.self_attn.o_proj.weight"],
            f_cos=f_cos,
            f_sin=f_sin,
        )
        x += r

        r = jnp.copy(x)
        x = rms_norm(
            x,
            w[f"model.layers.{i}.post_attention_layernorm.weight"],
            eps=params["norm_eps"],
        )
        x = ffn(
            x,
            W1=w[f"model.layers.{i}.mlp.up_proj.weight"],
            W2=w[f"model.layers.{i}.mlp.down_proj.weight"],
            V=w[f"model.layers.{i}.mlp.gate_proj.weight"],
        )
        x += r

    x = rms_norm(x, w["model.norm.weight"], eps=params["norm_eps"])

    logits = jnp.einsum("ik, kj -> ij", x, w["model.embed_tokens.weight"].T)
    return logits[-1, :]


def sample(logits, key, temperature=0.6):
    _, key = jax.random.split(key)
    if temperature != 0:
        logits /= temperature
    probs = jax.nn.softmax(logits, axis=-1)
    topk_probs, topk_indices = jax.lax.top_k(probs, 50)
    topk_probs /= jnp.sum(topk_probs)
    idx = jax.random.choice(key, topk_indices, p=topk_probs)
    return idx


class Tokenizer:
    def __init__(self, model_path):
        model_path = Path(model_path)
        assert model_path.is_file(), f"Model file {model_path} not found"
        mergeable_ranks = load_tiktoken_bpe(str(model_path))

        self.special_tokens = {
            "<|begin_of_text|>": 128000,
            "<|end_of_text|>": 128001,
            "<|start_header_id|>": 128006,
            "<|end_header_id|>": 128007,
            "<|eot_id|>": 128009,
        }
        self.special_tokens.update(
            {
                f"<|reserved_{i}|>": 128002 + i
                for i in range(256)
                if (128002 + i) not in self.special_tokens.values()
            }
        )

        self.model = tiktoken.Encoding(
            name=Path(model_path).name,
            pat_str=r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+",
            mergeable_ranks=mergeable_ranks,
            special_tokens=self.special_tokens,
        )

    def encode(
        self,
        text,
        bos=False,
        eos=False,
        allowed_special=set(),
        disallowed_special=(),
    ):
        if bos:
            tokens = [self.special_tokens["<|begin_of_text|>"]]
        else:
            tokens = []

        tokens += self.model.encode(
            text,
            allowed_special=allowed_special,
            disallowed_special=disallowed_special,
        )

        if eos:
            tokens.append(self.special_tokens["<|end_of_text|>"])
        return tokens

    def decode(self, tokens):
        return self.model.decode(tokens)


class ChatFormat:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def encode_header(self, message):
        tokens = []
        tokens.append(self.tokenizer.special_tokens["<|start_header_id|>"])
        tokens.extend(
            self.tokenizer.encode(message["role"], bos=False, eos=False)
        )
        tokens.append(self.tokenizer.special_tokens["<|end_header_id|>"])
        tokens.extend(self.tokenizer.encode("\n\n", bos=False, eos=False))
        return tokens

    def encode(self, text):
        message = {"role": "user", "content": text}

        tokens = self.encode_header(message)
        tokens.extend(
            self.tokenizer.encode(
                message["content"].strip(), bos=False, eos=False
            )
        )
        tokens.append(self.tokenizer.special_tokens["<|eot_id|>"])
        return tokens

    def decode(self, token_ids):
        return self.tokenizer.decode(token_ids)


def main(args):
    # Load weights & tokenizer from huggingface
    tokenizer = Tokenizer(args.tokenizer)
    if args.instruct:
        tokenizer = ChatFormat(tokenizer)
    weights = load_file(args.weights)

    # Tokenize
    tokens: list[int] = tokenizer.encode(args.prompt)

    # https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct/blob/main/config.json
    # https://github.com/meta-llama/llama-models/blob/main/models/llama3/api/args.py
    params = dict(
        v=128256,
        n=16,
        d=2048,
        h=32,
        h_kv=8,
        ctx_len=131_072,
        norm_eps=1e-5,
        rope_theta=500_000.0,
        rope_scale_factor=32,
        rope_high_freq_factor=4.0,
        rope_low_freq_factor=1.0,
        dtype=jnp.bfloat16,
    )

    # Reduce context len & rescale theta
    params["rope_theta"] = rescale_theta(
        params["rope_theta"],
        params["ctx_len"],
        8192,
    )
    params["ctx_len"] = 8192

    # Precompute frequences
    f_cos, f_sin = rope_frequencies(
        dim=params["d"] // params["h"],
        ctx_len=params["ctx_len"],
        theta=params["rope_theta"],
        dtype=params["dtype"],
    )

    # Downcasst weights
    weights = {k: v.astype(params["dtype"]) for k, v in weights.items()}  #  type: ignore

    # Generate
    key = jax.random.PRNGKey(args.key)
    for pos in range(args.len):
        _, key = jax.random.split(key)
        logits = xfmr(tokens, weights, params, f_cos, f_sin, pos=pos)
        sampled = sample(logits, key)
        gen = tokenizer.decode([sampled])
        print(gen, end="", flush=True)
    print()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--weights", type=str, required=True)
    parser.add_argument("--tokenizer", type=str, required=True)
    parser.add_argument("--instruct", action="store_true")
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--key", type=int, required=True)
    parser.add_argument("--len", type=int, default=50)
    args = parser.parse_args()
    main(args)
