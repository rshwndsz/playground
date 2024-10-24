from argparse import ArgumentParser
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import jax.random as jrn
from safetensors.flax import load
from transformers import AutoTokenizer


@dataclass
class Params:
    n: int
    h: int
    d: int
    d_ff: int
    v: int


def gelu(x):
    return (
        0.5
        * x
        * (
            1.0
            + jnp.tanh(jnp.sqrt(2.0 / jnp.pi) * (x + 0.044715 * jnp.pow(x, 3)))
        )
    )


def ffn(x, W1, b1, W2, b2):
    act = gelu
    x = act(jnp.einsum("ik, kj -> ij", x, W1) + b1)
    x = jnp.einsum("ik, kj -> ij", x, W2) + b2
    return x


def layer_norm(x, gamma, beta, eps=1e-5):
    mean = jnp.mean(x, axis=0).reshape(1, -1)
    std = jnp.std(x, axis=0).reshape(1, -1)
    return gamma * ((x - mean) / (std + eps)) + beta.reshape(1, -1)


def mhsa(x, mask, h, W_QKV, b_QKV, W_O, b_O):
    S, d = x.shape

    W_Q, W_K, W_V = jnp.split(W_QKV, 3, axis=-1)
    b_Q, b_K, b_V = jnp.split(b_QKV, 3, axis=-1)

    Q = jnp.dot(x, W_Q) + b_Q
    K = jnp.dot(x, W_K) + b_K
    V = jnp.dot(x, W_V) + b_V

    Q = Q.reshape(S, d // h, h).transpose(2, 0, 1)
    K = K.reshape(S, d // h, h).transpose(2, 0, 1)
    V = V.reshape(S, d // h, h).transpose(2, 0, 1)

    scores = jnp.einsum("hik, hjk -> hij", Q, K)
    scores /= jnp.sqrt(d)

    # [
    #     [ 0, -inf, -inf, -inf],  # 'I' can only attend to itself
    #     [ 0,    0, -inf, -inf],  # 'love' can attend to 'I' and itself
    #     [ 0,    0,    0, -inf],  # 'to' can attend to 'I', 'love', 'to'
    #     [ 0,    0,    0,    0],  # 'code' can attend to 'I', 'love', 'to', 'code'
    # ]

    #  Use -1e8 instead of -jnp.inf to avoid nan
    mask = jnp.triu(jnp.ones((S, S)), k=1) * (-1e8)
    scores += mask

    scores = jax.nn.softmax(scores, axis=-1)
    attention = jnp.einsum("hsi, hsj -> sjh", scores, V)
    attention = attention.reshape(S, d)

    out = jnp.dot(attention, W_O) + b_O
    return out


def xfmr(tokens, mask, weights, params):
    w = weights

    word_embed = w["wte.weight"]
    posn_embed = w["wpe.weight"]

    x = word_embed[jnp.array(tokens)] + posn_embed
    for i in range(params.n):
        x += mhsa(
            layer_norm(
                x,
                gamma=w[f"h.{i}.ln_1.weight"],
                beta=w[f"h.{i}.ln_1.bias"],
            ),
            mask,
            h=params.h,
            W_QKV=w[f"h.{i}.attn.c_attn.weight"],
            b_QKV=w[f"h.{i}.attn.c_attn.bias"],
            W_O=w[f"h.{i}.attn.c_proj.weight"],
            b_O=w[f"h.{i}.attn.c_proj.bias"],
        )

        x += ffn(
            layer_norm(
                x,
                gamma=w[f"h.{i}.ln_2.weight"],
                beta=w[f"h.{i}.ln_2.bias"],
            ),
            W1=w[f"h.{i}.mlp.c_fc.weight"],
            b1=w[f"h.{i}.mlp.c_fc.bias"],
            W2=w[f"h.{i}.mlp.c_proj.weight"],
            b2=w[f"h.{i}.mlp.c_proj.bias"],
        )

    # GPT-2 has a layer norm at the output
    x = layer_norm(x, gamma=w["ln_f.weight"], beta=w["ln_f.bias"])

    # lm_head is tied to wte.weight
    x = jnp.einsum("ik, jk -> ij", x, word_embed)

    # for a batch of sequences with a specific length,
    # the model will output logits for each token position in the input
    # indicating the likelihood of each token in the vocab being the next token.
    # we select the last token
    logits = x[-1, :]

    return logits


def sample(logits, key=None, temperature=0.0, k=8):
    if temperature == 0.0:
        probs = jax.nn.softmax(logits)
        token = jnp.argmax(probs)
    else:
        assert key is not None
        temperature = max(temperature, 1e-8)
        logits /= temperature
        probs = jax.nn.softmax(logits)
        token = jrn.choice(key, a=len(probs), p=probs)
    top_k = jnp.argsort(probs)[::-1][:k]
    return token, top_k


def main(args):
    key = jrn.key(args.key)

    tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    inputs = tokenizer(
        args.prompt, padding="max_length", max_length=1024, truncation=True
    )
    tokens, mask = inputs["input_ids"], inputs["attention_mask"]
    tokens, mask = jnp.array(tokens), jnp.array(mask)

    weights = load(open(args.weights, "rb").read())
    params = Params(n=12, h=12, d=768, d_ff=3072, v=50257)

    for _ in range(args.len):
        _, key = jrn.split(key)
        outputs = xfmr(tokens, mask, weights, params)
        gen, top_k = sample(outputs, key, args.temperature, args.k)

        print(tokenizer.decode([int(gen)]), end="", flush=True)

        pos = len(jnp.nonzero(mask)[0])
        tokens = tokens.at[pos].set(gen)
        mask = mask.at[pos].set(1)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--weights", type=str, required=True)
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--len", type=int, default=50)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--key", type=int, default=42)
    parser.add_argument("--device", type=str)
    args = parser.parse_args()

    main(args)
