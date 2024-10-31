import os
from argparse import ArgumentParser
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import jax.random as jrn
from einops import rearrange, repeat
from safetensors.flax import load_file
from transformers import AutoTokenizer


@dataclass
class Params:
    v: int
    n: int
    d: int
    d_ff: int
    h: int
    h_kv: int

    rope_theta: float
    rope_factor: float
    rope_high_freq_factor: float
    rope_low_freq_factor: float


def rms_norm(x, gamma, eps=1e-8, axis=0):
    # https://arxiv.org/pdf/1910.07467
    # https://github.com/meta-llama/llama-models/blob/2fe1a1690162910660332e3294a552cf0ec7e754/models/llama3/reference_impl/model.py#L31-L42
    return x * gamma / jnp.sqrt(jnp.mean(jnp.pow(x, 2), axis=axis, keepdims=True) + eps)


def swish(x, beta: float):
    return x * jax.nn.sigmoid(beta * x, axis=-1)


def ffn(x, W1, V, W2):
    # https://arxiv.org/pdf/2002.05202v1
    # https://github.com/meta-llama/llama-models/blob/2fe1a1690162910660332e3294a552cf0ec7e754/models/llama3/reference_impl/model.py#L218-L244
    x = swish(jnp.dot(x, W1), beta=1.0) * jnp.dot(x, V)
    x = jnp.dot(x, W2)
    return x


def compute_rope_frequencies(k: int, s: int, theta: float):
    # https://github.com/meta-llama/llama-models/blob/2fe1a1690162910660332e3294a552cf0ec7e754/models/llama3/reference_impl/model.py#L45-L100
    # https://arxiv.org/pdf/2104.09864
    m = jnp.arange(0, s)
    t = jnp.pow(theta, -jnp.arange(0, k, 2) / k)
    f = jnp.einsum("i, j -> ij", m, t)
    return jnp.cos(f) + 1j * jnp.sin(f)


def rope(x, f):
    x = x[..., ::2] + 1j * x[..., 1::2]
    f = jnp.expand_dims(f, 1)
    x_rotated = x * f
    x[..., ::2], x[..., 1::2] = x_rotated.real(), x_rotated.imag()
    return x


def gqa(x, h, h_kv, W_Q, W_K, W_V, W_O, f):
    # https://arxiv.org/pdf/2305.13245v3
    # https://github.com/meta-llama/llama-models/blob/2fe1a1690162910660332e3294a552cf0ec7e754/models/llama3/reference_impl/model.py#L103-L215
    s, d = x.shape

    # Project
    Q = jnp.einsum("ik, kj -> ij", x, W_Q)
    K = jnp.einsum("ik, kj -> ij", x, W_K)
    V = jnp.einsum("ik, kj -> ij", x, W_V)

    # Split along head
    Q = rearrange(Q, "s (head head_dim) -> head s head_dim", h=h)
    K = rearrange(K, "s (head_kv head_dim) -> head_kv s head_dim", h=h_kv)
    V = rearrange(V, "s (head_kv head_dim) -> head_kv s head_dim", h=h_kv)

    # Apply rotary embeddings
    Q = rope(Q, f)
    K = rope(K, f)

    # KV Cache
    # TODO

    # Grouped Query / Repeated Key-Value
    K = repeat(K, "s head_kv head_dim -> s (head_kv n_rep) head_dim", n_rep=h / h_kv)
    V = repeat(V, "s head_kv head_dim -> s (head_kv n_rep) head_dim", n_rep=h / h_kv)

    # Query-Key Lookup
    scores = jnp.einsum("hik, hkj -> hij", Q, K)
    scores /= jnp.sqrt(d // h)

    # Causal mask
    mask = jnp.triu(jnp.ones_like(scores), k=1) * (-1e8)
    scores += mask

    # Softmax
    scores = jax.nn.softmax(scores, axis=-1)

    # Project to value space
    attention = jnp.einsum("hik, hkj -> hij", scores, V)
    attention = attention.transpose(1, 0, 2).reshape(s, d)

    # Output projection
    out = jnp.einsum("ik, kj -> ij", attention, W_O)
    return out


def xfmr(tokens, w, params):
    # https://github.com/meta-llama/llama-models/blob/2fe1a1690162910660332e3294a552cf0ec7e754/models/llama3/reference_impl/model.py#L247-L334
    print(tokens)

    x = w["model.embed_tokens.weight"][jnp.array(tokens)]
    f = compute_rope_frequencies(k=params.d // params.h, s=len(tokens), theta=params.rope_theta)
    for i in range(params.n):
        x += gqa(
            rms_norm(x, w[f"model.layers.{i}.input_layernorm.weight"]),
            W_Q=w[f"model.layers.{i}.self_attn.q_proj.weight"],
            W_K=w[f"model.layers.{i}.self_attn.k_proj.weight"],
            W_V=w[f"model.layers.{i}.self_attn.v_proj.weight"],
            W_O=w[f"model.layers.{i}.self_attn.o_proj.weight"],
            h=params.h,
            h_kv=params.h_kv,
            f=f,
        )
        x += ffn(
            rms_norm(x, w[f"model.layers.{i}.post_attention_layernorm.weight"]),
            W1=w[f"model.layers.{i}.mlp.up_proj"].T,
            W2=w[f"model.layers.{i}.mlp.down_proj"].T,
            V=w[f"model.layers.{i}.mlp.gate_proj"].T,
        )
    x = rms_norm(x, w["model.norm.weight"])
    logits = jnp.einsum("ij, jk -> ik", x, w["model.embed_tokens.weight"].T)
    return logits[-1, :]


def sample(logits, key):
    probs = jax.nn.softmax(logits, axis=-1)
    topk_values, topk_indices = jax.lax.top_k(probs, 50)
    _idx = jrn.choice(key, topk_indices, p=topk_values)
    sampled_idx = topk_indices[_idx]
    return sampled_idx


def main(args):
    key = jrn.key(args.key)
    # https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct/blob/main/config.json
    params = Params(
        v=128256,
        n=16,
        d=2048,
        d_ff=8192,
        h=32,
        h_kv=8,
        rope_theta=500000.0,
        rope_factor=32,
        rope_high_freq_factor=4.0,
        rope_low_freq_factor=1.0,
    )

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

    xfmr(tokens, weights, params)

    for _ in range(args.len):
        _, key = jrn.split(key)
        logits = xfmr(tokens, weights, params)
        sampled = sample(logits, key)
        gen = tokenizer.decode([sampled])
        print(gen, end="", flush=True)
    print()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--weights", type=str, help="Weights directory")
    parser.add_argument("--prompt", type=str)
    parser.add_argument("--instruct", action="store_true")
    parser.add_argument("--key", type=int)
    parser.add_argument("--len", type=int, default=50)
    args = parser.parse_args()
    main(args)
