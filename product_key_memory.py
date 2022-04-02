"""
JAX implementation of Product-Key Memory.
Paper: https://arxiv.org/pdf/1907.05242.pdf
Helpful blog: https://www.pragmatic.ml/large-memory-layers-with-product-keys/
"""

from typing import Dict, Tuple, List, Any
import jax
import flax.linen as nn
import jax.numpy as jnp
import math


class PKM(nn.Module):
    dim: int
    heads: int = 4
    num_keys: int = 128
    topk: int = 32
    dim_head: int = 256
    input_dropout_p: float = 0.0
    query_dropout_p: float = 0.0
    value_dropout_p: float = 0.0

    def setup(self):
        assert self.dim % self.heads == 0, "dimension must be divisible by number of heads"
        self.dim_query = self.dim_head * self.heads
        self.to_queries = nn.Dense(features=self.dim_query, use_bias=False)
        self.norm = nn.LayerNorm()

        self.input_dropout = nn.Dropout(self.input_dropout_p)

        # initialize weights for keys and value parameters
        rnd_key = jax.random.PRNGKey(0)
        variance_init = jax.nn.initializers.variance_scaling(scale=1, mode="fan_in", distribution="normal")
        self.keys = variance_init(rnd_key, shape=(self.heads, self.num_keys, 2, self.dim_head // 2))
        self.values = nn.Embed(self.num_keys ** 2, self.dim, embedding_init=variance_init)

    @nn.compact
    def __call__(self, x, input_mask=None, **kwargs):
        t, b, e = x.shape
        h = self.heads
        x = self.input_dropout(x)

        queries = self.to_queries(x)
        queries = self.norm(queries)
        # queries = nn.Dropout(self.query_dropout)(queries)

        # ((t, b, e // 2), (t, b, e // 2))
        queries = jnp.array_split(queries, 2, axis=-1)

        # (p, t, b, h, d)
        queries = jnp.reshape(a=jnp.stack(queries), newshape=(2, t, b, h, -1))

        # (t, b, h, p, n)
        dots = jnp.einsum("ptbhd,hnpd->tbhpn", queries, self.keys)
        scores, indices = jax.lax.top_k(dots, self.topk)  # defaults to axis=-1

        # ((t, b, h, p // 2, -1), (t, b, h, p // 2, -1))
        scores, indices = map(lambda x: jnp.array_split(x, 2, axis=3), (scores, indices))

        all_topk = self.topk ** 2
        shape = (t, b, h, all_topk)

        all_scores = (scores[0][..., :, None] + scores[1][..., None, :])
        all_scores = all_scores.reshape(*shape)

        all_indices = (
            indices[0][..., :, None] * self.num_keys + indices[1][..., None, :]
        ).reshape(*shape)

        final_topk, final_indices = jax.lax.top_k(all_scores, self.topk)
        value_indices = jnp.take_along_axis(all_indices, final_indices, axis=-1)

        attn = nn.softmax(final_topk, axis=-1)

        value_indices, attn = map(
            lambda x: jnp.reshape(a=x, newshape=(-1, self.topk * h)), (value_indices, attn)
        )

        #out = self.values(value_indices, per_sample_weights=attn)
        out = jnp.sum(self.values(value_indices * attn), axis=1)

        out = nn.Dropout(self.value_dropout)(out)
        return out.reshape(t, b, e)


if __name__ == "__main__":
    from flax.core import unfreeze
    d = 128
    pkm = PKM(d)

    N = 8
    D = 128

    rnd_key = jax.random.PRNGKey(42)
    x1 = jax.random.normal(rnd_key, shape=(N,D,D))

    params = pkm.init(rnd_key, x1)
    y = pkm.apply(params, x1)
    print('initialized parameter shapes:\n', jax.tree_map(jnp.shape, unfreeze(params)))
    print('output shape:\n', y.shape)
