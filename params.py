# def parameters(v, mpe, d, d_qkv, d_ff, key, n=12):
#     _, *keys = jrn.split(key, num=2 + 4 * n + 1)
#     p = {}
#
#     p["wte.weight"] = jrn.normal(keys[0], v, d)
#     p["wpe.weight"] = jrn.normal(keys[1], mpe, d)
#
#     for i in range(n):
#         p[f"h.{i}.attn.bias"] = jnp.tril(jnp.ones(d, d))  # Mask
#
#         p[f"h.{i}.attn.c_attn.weight"] = jrn.normal(keys[i + 2], d, d_qkv)
#         p[f"h.{i}.attn.c_attn.bias"] = jnp.zeros(d_qkv)
#         p[f"h.{i}.attn.c_proj.weight"] = jrn.normal(keys[i + 2], d, d)
#         p[f"h.{i}.attn.c_proj.bias"] = jnp.zeros(d)
#
#         p[f"h.{i}.ln_1.weight"] = jnp.ones(d)
#         p[f"h.{i}.ln_1.bias"] = jnp.zeros(d)
#
#         p[f"h.{i}.mlp.c_fc.weight"] = jrn.normal(keys[i + 4], d, d_ff)
#         p[f"h.{i}.mlp.c_fc.bias"] = jnp.zeros(d_ff)
#         p[f"h.{i}.mlp.c_proj.weight"] = jrn.normal(keys[i + 5], d_ff, d)
#         p[f"h.{i}.mlp.c_proj.weight"] = jnp.zeros(d)
#
#         p[f"h.{i}.ln_2.weight"] = jnp.ones(d)
#         p[f"h.{i}.ln_2.bias"] = jnp.zeros(d)
#
#     p["ln_f.weight"] = jnp.ones(d)
#     p["ln_f.bias"] = jnp.ones(d)
#
#     return p
