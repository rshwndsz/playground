from argparse import ArgumentParser
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import jax.random as jrn
import tiktoken
from rich.console import Console
from transformers import GPT2LMHeadModel


@dataclass
class Params:
    n: int
    h: int
    d: int
    d_ff: int
    v: int


def gelu(x):
    return 0.5 * x * (1.0 + jnp.tanh(jnp.sqrt(2.0 / jnp.pi) * (x + 0.044715 * jnp.pow(x, 3))))


def ffn(x, W1, b1, W2, b2):
    x = gelu(jnp.einsum("ik, kj -> ij", x, W1) + b1)
    x = jnp.einsum("ik, kj -> ij", x, W2) + b2
    return x


def layer_norm(x, gamma, beta, eps=1e-5):
    mean = jnp.mean(x, axis=-1, keepdims=True)
    var = jnp.var(x, axis=-1, keepdims=True)
    return gamma * (x - mean) / jnp.sqrt(var + eps) + beta


def mhsa(x, h, W_QKV, b_QKV, W_O, b_O):
    S, d = x.shape

    QKV = jnp.einsum("ik, kj -> ij", x, W_QKV) + b_QKV
    Q, K, V = jnp.split(QKV, 3, axis=-1)

    Q = Q.reshape(S, h, d // h).transpose(1, 0, 2)
    K = K.reshape(S, h, d // h).transpose(1, 0, 2)
    V = V.reshape(S, h, d // h).transpose(1, 0, 2)

    scores = jnp.einsum("hik, hjk -> hij", Q, K)
    # Divide by the head's d_k dimension
    scores /= jnp.sqrt(d // h)

    #  Use -1e10 instead of -jnp.inf to avoid nan
    mask = jnp.triu(jnp.ones((S, S)), k=1) * (-1e10)
    scores += mask

    scores = jax.nn.softmax(scores, axis=-1)
    attention = jnp.einsum("hik, hkj -> hij", scores, V)
    attention = attention.transpose(1, 0, 2).reshape(S, d)

    out = jnp.einsum("sk, kj -> sj", attention, W_O) + b_O
    return out


def xfmr(tokens, weights, params):
    w = weights

    word_embed = w["transformer.wte.weight"]
    posn_embed = w["transformer.wpe.weight"]

    x = word_embed[jnp.array(tokens)] + posn_embed[jnp.array(range(len(tokens)))]
    for i in range(params.n):
        x += mhsa(
            layer_norm(
                x,
                gamma=w[f"transformer.h.{i}.ln_1.weight"],
                beta=w[f"transformer.h.{i}.ln_1.bias"],
            ),
            h=params.h,
            W_QKV=w[f"transformer.h.{i}.attn.c_attn.weight"],
            b_QKV=w[f"transformer.h.{i}.attn.c_attn.bias"],
            W_O=w[f"transformer.h.{i}.attn.c_proj.weight"],
            b_O=w[f"transformer.h.{i}.attn.c_proj.bias"],
        )

        x += ffn(
            layer_norm(
                x,
                gamma=w[f"transformer.h.{i}.ln_2.weight"],
                beta=w[f"transformer.h.{i}.ln_2.bias"],
            ),
            W1=w[f"transformer.h.{i}.mlp.c_fc.weight"],
            b1=w[f"transformer.h.{i}.mlp.c_fc.bias"],
            W2=w[f"transformer.h.{i}.mlp.c_proj.weight"],
            b2=w[f"transformer.h.{i}.mlp.c_proj.bias"],
        )

    # GPT-2 has a layer norm at the output
    x = layer_norm(
        x,
        gamma=w["transformer.ln_f.weight"],
        beta=w["transformer.ln_f.bias"],
    )

    # lm_head is tied to wte.weight
    logits = jnp.einsum("ik, jk -> ij", x, w["lm_head.weight"])

    # for a batch of sequences with a specific length,
    # the model will output logits for each token position in the input
    # indicating the likelihood of each token in the vocab being the next token.
    # we select the last token
    return logits[-1, :]


def sample(logits, key, k=50):
    _, key = jrn.split(key)
    probs = jax.nn.softmax(logits)

    # Select top-k tokens
    topk_probs, topk_indices = jax.lax.top_k(probs, k)

    # Renormalize distribution
    topk_probs /= jnp.sum(topk_probs)

    # Sample a token from the top-k distribution
    sampled = jrn.choice(key, a=len(topk_probs), p=topk_probs)

    token = int(topk_indices[sampled])
    token_prob = float(topk_probs[sampled])
    topk = [int(ix) for ix in topk_indices]
    return token, token_prob, topk, topk_probs


def colorise(words, probs, primary_color="green", end=" ", add_newline=True):
    console = Console()
    words = [word.replace(" ", "_") for word in words]
    for i, (word, prob) in enumerate(zip(words, probs)):
        if primary_color == "blue":
            b, r, g = max(int((prob * 1e5) % 255), 40), 0, 0
        elif primary_color == "red":
            r, g, b = max(int((prob * 1e5) % 255), 40), 0, 0
        elif primary_color == "green":
            g, r, b = max(int((prob * 1e5) % 255), 40), 0, 0
        else:
            raise NotImplementedError()
        color = f"#{r:02x}{g:02x}{b:02x}"
        console.print(f"[{color}]{word}[/]", end=end)
    if add_newline:
        console.print()


def inference(args):
    key = jrn.key(args.key)

    enc = tiktoken.get_encoding("gpt2")
    tokens = enc.encode(args.prompt)

    # Use the GPT2 model to get the weights
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    weights = model.state_dict()
    # Convert the weights to JAX arrays
    weights = {k: jnp.array(v.data.cpu().numpy()) for k, v in weights.items()}

    params = Params(n=12, h=12, d=768, d_ff=3072, v=50257)

    for _ in range(args.len):
        _, key = jrn.split(key)
        outputs = xfmr(tokens, weights, params)
        gen, gen_prob, topk, topk_probs = sample(outputs, key)

        if args.show_topk:
            colorise([enc.decode([tok]) for tok in topk], topk_probs)
            print(enc.decode([gen]), end="\n", flush=True)
        else:
            colorise([enc.decode([gen])], [gen_prob], end="", add_newline=False)
        tokens.append(gen)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--len", type=int, default=50)
    parser.add_argument("--key", type=int, default=42)
    parser.add_argument("--show-topk", action="store_true")
    parser.add_argument("--device", type=str)
    args = parser.parse_args()

    inference(args)
