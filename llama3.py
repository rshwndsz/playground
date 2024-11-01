import os
from argparse import ArgumentParser
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import jax.random as jrn
from einops import rearrange, repeat
from safetensors.flax import load_file
from transformers import AutoTokenizer

# References
# Meta: https://github.com/meta-llama/llama-models/blob/main/models/llama3/reference_impl/model.py
# Unsloth: https://github.com/unslothai/unsloth/blob/main/unsloth/models/llama.py
# Huggingface: https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py


@dataclass
class Params:
    v: int
    n: int
    d: int
    d_ff: int
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
    return gamma * x * jax.lax.rsqrt(jnp.mean(jnp.pow(x, 2), axis=-1, keepdims=True) + eps)


def swish(x, beta: float):
    return x * jax.lax.logistic(beta * x)


def ffn(x, W1, V, W2):
    # https://arxiv.org/pdf/2002.05202v1
    # Pytorch's nn.Linear is equivalent to jnp.dot(x, W.T)
    x = swish(jnp.dot(x, W1.T), beta=1.0) * jnp.dot(x, V.T)
    x = jnp.dot(x, W2.T)
    return x


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
    scaled_f = f.clone()
    scaled_f[is_high_wl] = f[is_high_wl] / scale_factor
    if is_bound_wl.any():
        smooth = (og_ctx_len / wl[is_bound_wl] - low_freq_factor) / (high_freq_factor - low_freq_factor)
        scaled_f[is_bound_wl] = (1 - smooth) * f[is_bound_wl] / scale_factor + smooth * f[is_bound_wl]
    return scaled_f


def rope_frequencies(dim: int, end: int, theta: float, scaling: bool = False):
    # https://arxiv.org/pdf/2104.09864
    m = jnp.arange(end)
    t = 1.0 / (theta ** (jnp.arange(0, dim, 2)[: dim // 2] / dim))
    if scaling:
        t = rope_scaling(t)
    f = jnp.einsum("i, j -> ij", m, t)
    return jnp.cos(f) + 1j * jnp.sin(f)


def rope(x, f_complex):
    x = x.reshape(*x.shape[:-1], -1, 2).astype(jnp.float32)
    x_complex = jax.lax.complex(x[..., 0], x[..., 1])
    f_complex = jnp.expand_dims(f_complex, 1)

    x_rotated = x_complex * f_complex
    x_out = jnp.stack([x_rotated.real, x_rotated.imag], axis=-1)
    x_out = rearrange(x_out, "s head k complex -> s head (k complex)")
    return x_out


def gqa(x, h, h_kv, W_Q, W_K, W_V, W_O, f_complex):
    # https://arxiv.org/pdf/2305.13245v3
    _, d = x.shape

    # Project
    Q = jnp.einsum("ik, jk -> ij", x, W_Q)
    K = jnp.einsum("ik, jk -> ij", x, W_K)
    V = jnp.einsum("ik, jk -> ij", x, W_V)

    # Split along head
    Q = rearrange(Q, "s (head head_dim) -> s head head_dim", head=h)
    K = rearrange(K, "s (head head_dim) -> s head head_dim", head=h_kv)
    V = rearrange(V, "s (head head_dim) -> s head head_dim", head=h_kv)

    # Apply rotary embeddings
    Q = rope(Q, f_complex)
    K = rope(K, f_complex)

    # KV Cache
    # TODO

    # Rearrange to parallelise attention computation across head
    Q = rearrange(Q, "s head head_dim -> head s head_dim")
    K = rearrange(K, "s head head_dim -> head s head_dim")
    V = rearrange(V, "s head head_dim -> head s head_dim")

    # Grouped Query / Repeated Key-Value
    K = repeat(K, "head_kv s head_dim -> (head_kv n_rep) s head_dim", n_rep=h // h_kv)
    V = repeat(V, "head_kv s head_dim -> (head_kv n_rep) s head_dim", n_rep=h // h_kv)

    # Query-Key Lookup
    scores = jnp.einsum("hik, hjk -> hij", Q, K)
    scores /= jnp.sqrt(d // h)

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


def xfmr(tokens, w, params, f_complex, pos):
    s = len(tokens)

    x = w["model.embed_tokens.weight"][jnp.array(tokens)]
    f_complex = f_complex[pos : pos + s]

    for i in range(params.n):
        x += gqa(
            rms_norm(x, w[f"model.layers.{i}.input_layernorm.weight"], eps=params.norm_eps),
            W_Q=w[f"model.layers.{i}.self_attn.q_proj.weight"],
            W_K=w[f"model.layers.{i}.self_attn.k_proj.weight"],
            W_V=w[f"model.layers.{i}.self_attn.v_proj.weight"],
            W_O=w[f"model.layers.{i}.self_attn.o_proj.weight"],
            h=params.h,
            h_kv=params.h_kv,
            f_complex=f_complex,
        )
        x += ffn(
            rms_norm(x, w[f"model.layers.{i}.post_attention_layernorm.weight"], eps=params.norm_eps),
            W1=w[f"model.layers.{i}.mlp.up_proj.weight"],
            W2=w[f"model.layers.{i}.mlp.down_proj.weight"],
            V=w[f"model.layers.{i}.mlp.gate_proj.weight"],
        )

    # Llama models have a final norm
    x = rms_norm(x, w["model.norm.weight"], eps=params.norm_eps)

    # Use transpose of word embeddings matrix as tie_word_embeddings is true
    # https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct/blob/9213176726f574b556790deb65791e0c5aa438b6/config.json#L34
    logits = jnp.einsum("ij, jk -> ik", x, w["model.embed_tokens.weight"].T)

    return logits[-1, :]


def sample(logits, key):
    probs = jax.nn.softmax(logits, axis=-1)
    topk_probs, topk_indices = jax.lax.top_k(probs, 50)
    topk_probs /= jnp.sum(topk_probs)
    idx = jrn.choice(key, topk_indices, p=topk_probs)
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
        d_ff=8192,
        h=32,
        h_kv=8,
        max_seq_len=2048,
        norm_eps=1e-5,
        rope_theta=500000.0,
        rope_scaling=False,
        rope_scale_factor=32,
        rope_high_freq_factor=4.0,
        rope_low_freq_factor=1.0,
    )

    # Precompute frequences for 2048*2 = 4096
    frequencies = rope_frequencies(
        dim=params.d // params.h,
        end=params.max_seq_len * 2,
        theta=params.rope_theta,
        scaling=params.rope_scaling,
    )

    # Generate
    key = jrn.key(args.key)
    for pos in range(args.len):
        _, key = jrn.split(key)
        logits = xfmr(tokens, weights, params, f_complex=frequencies, pos=pos)
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
