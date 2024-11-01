import os
from argparse import ArgumentParser
from dataclasses import dataclass

import jax
import jax.numpy as jnp
from einops import rearrange
from safetensors.flax import load_file
from transformers import AutoTokenizer

# References
# Meta: https://github.com/meta-llama/llama-models/blob/main/models/llama3/reference_impl/model.py
# Unsloth: https://github.com/unslothai/unsloth/blob/main/unsloth/models/llama.py
# Huggingface: https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py
# Sebastian Raschka: https://github.com/rasbt/LLMs-from-scratch/blob/main/ch05/07_gpt_to_llama/standalone-llama32.ipynb


@dataclass
class Params:
    v: int
    n: int
    d: int
    h: int
    h_kv: int
    max_seq_len: int

    norm_eps: float

    rope_theta: float
    rope_scaling: bool
    rope_scale_factor: float
    rope_high_freq_factor: float
    rope_low_freq_factor: float


def rms_norm(x, gamma, eps):
    # https://arxiv.org/pdf/1910.07467
    return gamma * x * jax.lax.rsqrt(jnp.mean(jnp.square(x), axis=-1, keepdims=True) + eps)


def swish(x, beta: float):
    return x * jax.nn.sigmoid(beta * x)


def ffn(x, W1, V, W2):
    # https://arxiv.org/pdf/2002.05202v1
    # Pytorch's nn.Linear is equivalent to jnp.dot(x, W.T)
    x = swish(jnp.dot(x, W1.T), beta=1.0) * jnp.dot(x, V.T)
    x = jnp.dot(x, W2.T)
    return x


def rope_frequencies(dim, ctx_len, theta, use_scaling=False):
    # https://arxiv.org/pdf/2104.09864
    def rope_scaling(f):
        # Weird scaling added in from grid search
        scale_factor = 32.0
        high_freq_factor = 4.0
        low_freq_factor = 1.0
        og_ctx_len = 8192

        # Get wavelengths from frequencies
        wl = 2 * jnp.pi / f

        # Conditions on wavelength
        is_low_wl = wl < (og_ctx_len / low_freq_factor)
        is_high_wl = wl > (og_ctx_len / high_freq_factor)
        is_bound_wl = ~(is_low_wl | is_high_wl)

        # Scale frequencies based on conditions
        scaled_f = jnp.copy(f)
        scaled_f[is_high_wl] = f[is_high_wl] / scale_factor
        if is_bound_wl.any():
            smooth = (og_ctx_len / wl[is_bound_wl] - low_freq_factor) / (high_freq_factor - low_freq_factor)
            scaled_f[is_bound_wl] = (1 - smooth) * f[is_bound_wl] / scale_factor + smooth * f[is_bound_wl]
        return scaled_f

    m = jnp.arange(ctx_len, dtype=jnp.float32)
    t = 1.0 / (theta ** (jnp.arange(0, dim, 2, dtype=jnp.float32)[: (dim // 2)] / dim))
    if use_scaling:
        t = rope_scaling(t)
    f = jnp.einsum("i, j -> ij", m, t)

    f = jnp.concatenate([f, f], axis=-1)
    return jnp.cos(f), jnp.sin(f)


def rope(x, f_cos, f_sin):
    x1, x2 = jnp.split(x, 2, axis=-1)
    rotated = jnp.concatenate([-x2, x1], axis=-1)
    x_rotated = x * f_cos + rotated * f_sin
    return x_rotated


def gqa(x, h, h_kv, W_Q, W_K, W_V, W_O, f_cos, f_sin):
    # https://arxiv.org/pdf/2305.13245v3
    # Project
    Q = jnp.einsum("ik, jk -> ij", x, W_Q)
    K = jnp.einsum("ik, jk -> ij", x, W_K)
    V = jnp.einsum("ik, jk -> ij", x, W_V)

    # Split along head
    Q = rearrange(Q, "s (head head_dim) -> head s head_dim", head=h)
    K = rearrange(K, "s (head head_dim) -> head s head_dim", head=h_kv)
    V = rearrange(V, "s (head head_dim) -> head s head_dim", head=h_kv)

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
    mask = jnp.triu(jnp.ones_like(scores), k=1) * (-1e8)
    scores += mask

    # Softmax
    scores = jax.nn.softmax(scores, axis=-1)

    # Project to value space
    attention = jnp.einsum("hik, hkj -> hij", scores, V)
    attention = rearrange(attention, "head s head_dim -> s (head head_dim)")

    # Output projection
    out = jnp.einsum("ik, jk -> ij", attention, W_O)
    return out


def xfmr(tokens, w, params, f_cos, f_sin, pos):
    s = len(tokens)

    x = w["model.embed_tokens.weight"][jnp.array(tokens)]
    f_cos, f_sin = f_cos[pos : pos + s], f_sin[pos : pos + s]

    for i in range(params.n):
        r = x
        x = rms_norm(x, w[f"model.layers.{i}.input_layernorm.weight"], eps=params.norm_eps)
        x = gqa(
            x,
            h=params.h,
            h_kv=params.h_kv,
            W_Q=w[f"model.layers.{i}.self_attn.q_proj.weight"],
            W_K=w[f"model.layers.{i}.self_attn.k_proj.weight"],
            W_V=w[f"model.layers.{i}.self_attn.v_proj.weight"],
            W_O=w[f"model.layers.{i}.self_attn.o_proj.weight"],
            f_cos=f_cos,
            f_sin=f_sin,
        )
        x += r

        r = x
        x = rms_norm(x, w[f"model.layers.{i}.post_attention_layernorm.weight"], eps=params.norm_eps)
        x = ffn(
            x,
            W1=w[f"model.layers.{i}.mlp.up_proj.weight"],
            W2=w[f"model.layers.{i}.mlp.down_proj.weight"],
            V=w[f"model.layers.{i}.mlp.gate_proj.weight"],
        )
        x += r

    # Llama models have a final norm
    x = rms_norm(x, w["model.norm.weight"], eps=params.norm_eps)

    # Use transpose of word embeddings matrix as tie_word_embeddings is true
    # https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct/blob/9213176726f574b556790deb65791e0c5aa438b6/config.json#L34
    logits = jnp.einsum("ij, jk -> ik", x, w["model.embed_tokens.weight"].T)

    return logits[-1, :]


def sample(logits, key, temperature=0.6):
    if temperature != 0:
        logits /= temperature
    probs = jax.nn.softmax(logits, axis=-1)
    topk_probs, topk_indices = jax.lax.top_k(probs, 50)
    topk_probs /= jnp.sum(topk_probs)
    idx = jax.random.choice(key, topk_indices, p=topk_probs)
    return idx


def main(args):
    # Load weights & tokenizer from huggingface
    if not args.instruct:
        model_id = "meta-llama/Llama-3.2-1B"
        weights = load_file(os.path.join(args.weights, "Llama-3.2-1B.safetensors"))
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        tokens: list[int] = tokenizer(args.prompt)["input_ids"]
    else:
        model_id = "meta-llama/Llama-3.2-1B-Instruct"
        weights = load_file(os.path.join(args.weights, "Llama-3.2-1B-Instruct.safetensors"))
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        messages = [{"role": "user", "content": args.prompt}]
        tokens: list[int] = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
        )["input_ids"]

    # https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct/blob/main/config.json
    # https://github.com/meta-llama/llama-models/blob/main/models/llama3/api/args.py
    params = Params(
        v=128256,
        n=16,
        d=2048,
        h=32,
        h_kv=8,
        max_seq_len=2048,
        norm_eps=1e-5,
        rope_theta=500_000.0,
        rope_scaling=False,
        rope_scale_factor=32,
        rope_high_freq_factor=4.0,
        rope_low_freq_factor=1.0,
    )

    # Precompute frequences for 2048*2 = 4096
    f_cos, f_sin = rope_frequencies(
        dim=params.d // params.h,
        ctx_len=params.max_seq_len * 2,
        theta=params.rope_theta,
        use_scaling=params.rope_scaling,
    )

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
    parser.add_argument("--instruct", action="store_true")
    parser.add_argument("--weights", type=str, help="Weights directory", required=True)
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--key", type=int, required=True)
    parser.add_argument("--len", type=int, default=50)
    args = parser.parse_args()
    main(args)
