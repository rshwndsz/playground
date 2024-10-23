from argparse import ArgumentParser
from dataclasses import dataclass

import jax
import jax.numpy as jnp
from safetensors.flax import load
from transformers import AutoTokenizer


@dataclass
class Params:
    n: int
    h: int
    d: int
    d_ff: int
    v: int


def ffn(x, W1, b1, W2, b2):
    act = jax.nn.relu
    x = act(jnp.einsum("ik, kj -> ij", x, W1)) + b1
    x = act(jnp.einsum("ik, kj -> ij", x, W2)) + b2
    return x


def layer_norm(x, gamma, beta, eps=1e-8):
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

    # TODO: There should be a more efficient way than tiling & branching
    mask = jnp.triu(jnp.tile(jnp.array(mask), (len(mask), 1)))
    mask = jnp.where(mask == 1.0, -jnp.inf, 0.0)
    scores += mask

    scores = jax.nn.softmax(scores)
    attention = jnp.einsum("hsi, hsj -> sjh", scores, V)
    attention = attention.reshape(S, d)

    out = jnp.dot(attention, W_O) + b_O
    return out


def xfmr(tokens, mask, weights, params):
    w = weights

    x = w["wte.weight"][jnp.array(tokens)] + w["wpe.weight"]
    for i in range(params.n):
        x += layer_norm(
            mhsa(
                x,
                mask,
                params.h,
                w[f"h.{i}.attn.c_attn.weight"],
                w[f"h.{i}.attn.c_attn.bias"],
                w[f"h.{i}.attn.c_proj.weight"],
                w[f"h.{i}.attn.c_proj.bias"],
            ),
            w[f"h.{i}.ln_1.weight"],
            w[f"h.{i}.ln_1.bias"],
        )
        x += layer_norm(
            ffn(
                x,
                w[f"h.{i}.mlp.c_fc.weight"],
                w[f"h.{i}.mlp.c_fc.bias"],
                w[f"h.{i}.mlp.c_proj.weight"],
                w[f"h.{i}.mlp.c_proj.bias"],
            ),
            w[f"h.{i}.ln_2.weight"],
            w[f"h.{i}.ln_2.bias"],
        )
    x = layer_norm(x, w["ln_f.weight"], w["ln_f.bias"])

    # lm_head is tied to wte.weight
    x = jnp.einsum("ik, jk -> ij", x, w["wte.weight"])

    # for a batch of sequences with a specific length,
    # the model will output logits for each token position in the input
    # indicating the likelihood of each token in the vocab being the next token.
    # we select the last token
    logits = x[-1, :]
    return logits


def sample(logits):
    token = jnp.argmax(jax.nn.softmax(logits))
    return token


def main(args):
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
        outputs = xfmr(tokens, mask, weights, params)
        gen = sample(outputs)
        print(tokenizer.decode([gen]), end="", flush=True)

        pos = len(jnp.nonzero(mask)[0])
        tokens = tokens.at[pos].set(gen)
        mask = mask.at[pos].set(1)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--weights", type=str, required=True)
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--len", type=int, default=50)
    parser.add_argument("--device", type=str)
    args = parser.parse_args()

    main(args)
