"""
This is a barebones reimplementation of the code in model.py that works on matrices and not just vectors.
"""

import functools
from typing import Any, Callable, Optional, Tuple

from flax.linen.initializers import zeros
from flax.linen.linear import default_kernel_init
from flax.linen.linear import DenseGeneral
from flax.linen.linear import PrecisionLike
from flax.linen.module import compact
from flax.linen.module import merge_param
from flax.linen.module import Module

import jax
from jax import lax
from jax import random
import jax.numpy as jnp

PRNGKey = Any
Shape = Tuple[int]
Dtype = Any
Array = Any


def dot_product_attention(
    Q: Array,
    K: Array,
    V: Array
):
    """Computes dot-product attention given query key, and value matrices.
    Currently, does not support dropout of attention probabilities or mask of padding tokens.

    Combines dot_product_attention_weights and dot_product_attention from model.py
    
    Args:
        Q: query matrix with shape [batch, num_tokens, num_heads, d_qkv]
        K: key matrix with shape [batch, num_tokens, num_heads, d_qkv]
        V: values matrix with shape [batch, num_tokens, num_heads, d_qkv]
    Returns:
        output of shape [batch, num_tokens, num_heads, d_qkv]
    """
    assert Q.shape == K.shape == V.shape

    # calculate attention matrix 
    d_qkv = Q.shape[-1]

    '''eisnum abbreviations:
        - h: num heads
        - q: num query tokens
        - k: num key tokens
        - d: d_qkv
    '''
    attention_logits = jnp.einsum("...qhd,...khd->...hqk", Q, K) # [batch, num_heads, num_tokens, num_tokens]
    attention_logits /= jnp.sqrt(Q.shape[-1])
    attention_probs = jax.nn.softmax(attention_logits, axis=-1)
    
    out = jnp.einsum("...hqk,...khd->...qhd", attention_probs, V)
    return out


class MultiHeadSelfAttention(Module):
    """Multi-head self attention.
    Does not support dropout, masking, or special autoregressive cache during decoding.
    
    Attributes:
        n_heads: number of attention heads
        d_qkv: dimension of the key, query, and value (this is per head, unlike model.py)
        d_output: dimension of the output
    """

    n_heads: Optional[int] = 4
    d_qkv: Optional[int] = 32
    d_output: Optional[int] = 256

    @compact
    def __call__(
        self,
        inputs: Array
    ):
        """
        Applies multi-head dot product self-attention on the input data.

        Args:
            inputs: input of shape [batch, n_tokens, d_input]
        Returns:
            output of shape [batch, n_tokens, d_output]
        """
        dense_QKV = functools.partial(
            DenseGeneral,
            axis=-1,
            features=(self.n_heads, self.d_qkv)
        )

        # project inputs to multi-headed Q, K, V
        # dimensions for all three matrices are [batch, n_tokens, n_heads, d_qkv]
        Q, K, V = (
            dense_QKV(name='query')(inputs),
            dense_QKV(name='key')(inputs),
            dense_QKV(name='value')(inputs)
        )

        out = dot_product_attention(Q, K, V)

        return DenseGeneral(
            features=self.d_output,
            axis=(-2, -1),
            name='out'
        )(out)


def run_self_attention():
    B, L, D = 10, 128, 200
    H = 6

    rkey = random.PRNGKey(0)
    k1, k2 = random.split(rkey, 2)
    x = random.uniform(k1, (B, L, D))

    model = MultiHeadSelfAttention(n_heads=H)
    params = model.init(k2, x)
    y = model.apply(params, x)

    print("initialized parameter shapes: \n", jax.tree_map(jnp.shape, params))
    print("output shape: \n", y.shape)


if __name__ == '__main__':
    run_self_attention()

