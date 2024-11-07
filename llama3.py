from argparse import ArgumentParser
from functools import partial
from pathlib import Path
from typing import Dict, NamedTuple, Optional, Tuple

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


class RopeScalingParams(NamedTuple):
    scale_fctr: float
    hi_freq_fctr: float
    lo_freq_fctr: float
    og_ctx_len: int


class Params(NamedTuple):
    v: int
    n: int
    d: int
    h: int
    h_kv: int
    ctx_len: int
    norm_eps: float
    rope_theta: float
    rope_scaling: RopeScalingParams
    dtype: jnp.dtype


class DecoderWeights(NamedTuple):
    GAMMA_ATTN: jax.Array
    W_Q: jax.Array
    W_K: jax.Array
    W_V: jax.Array
    W_O: jax.Array
    GAMMA_FFN: jax.Array
    W1: jax.Array
    W2: jax.Array
    W3: jax.Array


class XfmrWeights(NamedTuple):
    W_E: jax.Array
    BLOCKS: DecoderWeights
    GAMMA_OUT: jax.Array


def load_weights(w: Dict[str, jax.Array], params: Params) -> XfmrWeights:
    w = {k: v.astype(params.dtype) for k, v in w.items()}
    return XfmrWeights(
        W_E=w["model.embed_tokens.weight"],
        BLOCKS=DecoderWeights(
            GAMMA_ATTN=jnp.stack([w[f"model.layers.{i}.input_layernorm.weight"] for i in range(params.n)]),
            W_Q=jnp.stack([w[f"model.layers.{i}.self_attn.q_proj.weight"] for i in range(params.n)]),
            W_K=jnp.stack([w[f"model.layers.{i}.self_attn.k_proj.weight"] for i in range(params.n)]),
            W_V=jnp.stack([w[f"model.layers.{i}.self_attn.v_proj.weight"] for i in range(params.n)]),
            W_O=jnp.stack([w[f"model.layers.{i}.self_attn.o_proj.weight"] for i in range(params.n)]),
            GAMMA_FFN=jnp.stack([w[f"model.layers.{i}.post_attention_layernorm.weight"] for i in range(params.n)]),
            W1=jnp.stack([w[f"model.layers.{i}.mlp.gate_proj.weight"] for i in range(params.n)]),
            W2=jnp.stack([w[f"model.layers.{i}.mlp.up_proj.weight"] for i in range(params.n)]),
            W3=jnp.stack([w[f"model.layers.{i}.mlp.down_proj.weight"] for i in range(params.n)]),
        ),
        GAMMA_OUT=w["model.norm.weight"],
    )


@jax.jit
def rms_norm(x: jax.Array, gamma: jax.Array, eps: float) -> jax.Array:
    # https://arxiv.org/pdf/1910.07467
    return gamma * x * jax.lax.rsqrt(jnp.mean(jnp.square(x), axis=-1, keepdims=True) + eps)


@jax.jit
def swish(x: jax.Array, b: float) -> jax.Array:
    return x * jax.lax.logistic(x * b)


@jax.jit
def ffn(x: jax.Array, W1: jax.Array, W2: jax.Array, W3: jax.Array) -> jax.Array:
    # https://arxiv.org/pdf/2002.05202v1
    x = swish(jnp.dot(x, W1.T), b=1.0) * jnp.dot(x, W2.T)
    x = jnp.dot(x, W3.T)
    return x


def rescale_theta(theta_old: float, ctx_len_old: int, ctx_len_new: int) -> float:
    scaling_factor = ctx_len_new / ctx_len_old
    theta_new = theta_old * scaling_factor
    return theta_new


# Weird scaling added in from grid search
def rope_scaling(f: jax.Array, scaling_params: RopeScalingParams) -> jax.Array:
    scale_fctr, hi_freq_fctr, lo_freq_fctr, og_ctx_len = scaling_params

    # Get wavelengths from frequencies
    wl = 2 * jnp.pi / f

    # Conditions on wavelength
    is_lo_wl = wl < (og_ctx_len / lo_freq_fctr)  # Low
    is_hi_wl = wl > (og_ctx_len / hi_freq_fctr)  # High
    is_bo_wl = ~(is_lo_wl | is_hi_wl)  # Bound

    # Scale frequencies based on conditions
    scaled_f = jnp.where(is_hi_wl, f / scale_fctr, f)
    if is_bo_wl.any():
        smooth = (og_ctx_len / wl[is_bo_wl] - lo_freq_fctr) / (hi_freq_fctr - lo_freq_fctr)
        scaled_f = jnp.where(is_bo_wl, (1 - smooth) * f / scale_fctr + smooth * f, scaled_f)

    return scaled_f


def rope_frequencies(
    dim: int,
    ctx_len: int,
    theta: float,
    scaling_params: Optional[RopeScalingParams] = None,
    dtype: jnp.dtype = jnp.float32,
) -> Tuple[jax.Array, jax.Array]:
    # https://arxiv.org/pdf/2104.09864
    m = jnp.arange(ctx_len, dtype=jnp.float32)
    t = 1.0 / (theta ** (jnp.arange(0, dim, 2, dtype=jnp.float32)[: (dim // 2)] / dim))
    if scaling_params is not None:
        t = rope_scaling(t, scaling_params)
    f = jnp.einsum("i, j -> ij", m, t)
    f = jnp.concatenate([f, f], axis=-1)
    return jnp.cos(f).astype(dtype), jnp.sin(f).astype(dtype)


@jax.jit
def rope(x: jax.Array, f_cos: jax.Array, f_sin: jax.Array) -> jax.Array:
    # https://arxiv.org/pdf/2104.09864
    x1, x2 = jnp.split(x, 2, axis=-1)
    rotated = jnp.concatenate([-x2, x1], axis=-1)
    x_rotated = x * f_cos + rotated * f_sin
    return x_rotated


@partial(jax.jit, static_argnames=["h", "h_kv"])
def gqa(
    x: jax.Array,
    h: int,
    h_kv: int,
    W_Q: jax.Array,
    W_K: jax.Array,
    W_V: jax.Array,
    W_O: jax.Array,
    f_cos: jax.Array,
    f_sin: jax.Array,
) -> jax.Array:
    s, _ = x.shape

    # TODO KVCache

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


@partial(jax.jit, static_argnames=["params"])
def xfmr(
    tokens: list[int],
    w: XfmrWeights,
    params: Params,
    f_cos: jax.Array,
    f_sin: jax.Array,
) -> jax.Array:
    s = len(tokens)

    f_cos, f_sin = f_cos[:s, :], f_sin[:s, :]
    x = w.W_E[jnp.array(tokens)]

    def layer(x, w_i):
        x += gqa(
            rms_norm(
                x,
                w_i.GAMMA_ATTN,
                eps=params.norm_eps,
            ),
            h=params.h,
            h_kv=params.h_kv,
            W_Q=w_i.W_Q,
            W_K=w_i.W_K,
            W_V=w_i.W_V,
            W_O=w_i.W_O,
            f_cos=f_cos,
            f_sin=f_sin,
        )
        x += ffn(
            rms_norm(
                x,
                w_i.GAMMA_FFN,
                eps=params.norm_eps,
            ),
            W1=w_i.W1,
            W2=w_i.W2,
            W3=w_i.W3,
        )
        return x, None

    x, _ = jax.lax.scan(layer, x, w.BLOCKS)
    x = rms_norm(x, w.GAMMA_OUT, eps=params.norm_eps)

    logits = jnp.einsum("ik, jk -> ij", x, w.W_E)
    return logits[-1, :]


# @partial(jax.jit, static_argnames=["key", "temperature"])
def sample(logits: jax.Array, key: jax.Array, temperature: float = 0.6) -> int:
    _, key = jax.random.split(key)
    temperature = min(temperature, 1e-3)
    logits /= temperature
    probs = jax.nn.softmax(logits, axis=-1)
    topk_probs, topk_indices = jax.lax.top_k(probs, 50)
    topk_probs /= jnp.sum(topk_probs)
    idx = jax.random.choice(key, topk_indices, p=topk_probs)
    return int(idx)


class Tokenizer:
    def __init__(self, model_path: str | Path):
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
            {f"<|reserved_{i}|>": 128002 + i for i in range(256) if (128002 + i) not in self.special_tokens.values()}
        )

        self.model = tiktoken.Encoding(
            name=Path(model_path).name,
            pat_str=r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+",
            mergeable_ranks=mergeable_ranks,
            special_tokens=self.special_tokens,
        )

    def encode(self, text: str, bos: bool = False, eos: bool = False, allowed_special=set(), disallowed_special=()):
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
    def __init__(self, tokenizer: Tokenizer):
        self.tokenizer = tokenizer

    def encode_header(self, message):
        tokens = []
        tokens.append(self.tokenizer.special_tokens["<|start_header_id|>"])
        tokens.extend(self.tokenizer.encode(message["role"], bos=False, eos=False))
        tokens.append(self.tokenizer.special_tokens["<|end_header_id|>"])
        tokens.extend(self.tokenizer.encode("\n\n", bos=False, eos=False))
        return tokens

    def encode(self, text: str):
        message = {"role": "user", "content": text}

        tokens = self.encode_header(message)
        tokens.extend(self.tokenizer.encode(message["content"].strip(), bos=False, eos=False))
        tokens.append(self.tokenizer.special_tokens["<|eot_id|>"])
        return tokens

    def decode(self, token_ids: list[int]):
        return self.tokenizer.decode(token_ids)


def main(args):
    # https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct/blob/main/config.json
    # https://github.com/meta-llama/llama-models/blob/main/models/llama3/api/args.py
    # Original config
    og = {
        "ctx_len": 131_072,
        "rope_theta": 500_000.0,
    }
    # Reduce context len & rescale theta
    new_ctx_len = 8192
    new_rope_theta = rescale_theta(og["rope_theta"], og["ctx_len"], new_ctx_len)
    # New config
    params = Params(
        v=128256,
        n=16,
        d=2048,
        h=32,
        h_kv=8,
        norm_eps=1e-5,
        dtype=jnp.bfloat16,
        ctx_len=new_ctx_len,
        rope_theta=new_rope_theta,
        rope_scaling=RopeScalingParams(
            scale_fctr=32,
            hi_freq_fctr=4.0,
            lo_freq_fctr=1.0,
            og_ctx_len=8192,
        ),
    )

    # Load weights & tokenizer from huggingface
    tokenizer = Tokenizer(args.tokenizer)
    if args.instruct:
        tokenizer = ChatFormat(tokenizer)
    weights = load_weights(load_file(args.weights), params)

    # Precompute frequences
    f_cos, f_sin = rope_frequencies(
        dim=params.d // params.h,
        ctx_len=params.ctx_len,
        theta=params.rope_theta,
        scaling_params=params.rope_scaling,
        dtype=params.dtype,
    )

    # Tokenize
    tokens: list[int] = tokenizer.encode(args.prompt)

    # Generate
    key = jax.random.PRNGKey(args.key)
    for _ in range(args.len):
        logits = xfmr(tokens, weights, params, f_cos, f_sin)

        _, key = jax.random.split(key)
        sampled = sample(logits, key)
        gen = tokenizer.decode([sampled])

        print(gen, end="", flush=True)
        tokens.append(sampled)
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
