import os
from argparse import ArgumentParser
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import jax.random as jrn
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
    return x * gamma / jnp.sqrt(jnp.mean(jnp.pow(x, 2), axis=axis, keepdims=True) + eps)


def ffn(x, W1, V, W2):
    # https://arxiv.org/pdf/2002.05202v1
    def swish(x, beta: float):
        return x * jax.nn.sigmoid(beta * x, axis=-1)

    x = swish(jnp.dot(x, W1), beta=1.0) * jnp.dot(x, V)
    x = jnp.dot(x, W2)
    return x


def rope():
    # https://arxiv.org/pdf/2104.09864

    pass


def gqa(x, h, W_Q, W_K, W_V, W_O):
    # https://arxiv.org/pdf/2305.13245v3
    s, d = x.shape

    Q = jnp.einsum("ik, kj", x, W_Q).reshape(s, h, d // h).transpose(1, 0, 2)
    K = jnp.einsum("ik, kj", x, W_K).reshape(s, h, d // h).transpose(1, 0, 2)
    V = jnp.einsum("ik, kj", x, W_V).reshape(s, h, d // h).transpose(1, 0, 2)

    scores = jnp.einsum("hik, hkj -> hij", Q, K)
    scores /= jnp.sqrt(d // h)

    mask = jnp.triu(jnp.ones_like(scores), k=1) * (-1e8)
    scores += mask

    scores = jax.nn.softmax(scores, axis=-1)

    attention = jnp.einsum("hik, hkj -> hij", scores, V)
    attention = attention.transpose(1, 0, 2).reshape(s, d)

    out = jnp.einsum("ik, kj -> ij", attention, W_O)
    return out


def xfmr(tokens, weights, params):
    print(tokens)

    # x = W_E[jnp.array(tokens)] + rope(tokens)
    # for i in range(params.n):
    #     x += mhsa(rms_norm(x, gamma), W_QKV, W_O)
    #     x += ffn(rms_norm(x, gamma), W1, V, W2)
    # x = rms_norm(x, gamma)
    # logits = jnp.einsum("ij, jk -> ik", x, lm_head)
    # return logits[-1, :]


def sample(logits, key):
    probs = jax.nn.softmax(logits, axis=-1)
    topk_values, topk_indices = jax.lax.top_k(probs, 50)
    _idx = jrn.choice(key, topk_indices, p=topk_values)
    sampled_idx = topk_indices[_idx]
    return sampled_idx


def main(args):
    key = jrn.key(args.key)
    # https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct/blob/main/config.json
    params = Params(v=128256, n=16, d=2048, d_ff=8192, h=32, h_kv=8)

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
