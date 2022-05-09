"""
This is a barebones reimplementation of the code in model.py that works on matrices and not just vectors.
"""

from doctest import OutputChecker
import functools
from typing import Any, Callable, Optional, Tuple

from flax import linen as nn, struct
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
import math
import numpy as np
import pdb

PRNGKey = Any
Shape = Tuple[int]
Dtype = Any
Array = Any


def dot_product_attention_weights(
    query: Array,
    key: Array,
    bias: Optional[Array] = None,
    mask: Optional[Array] = None,
    broadcast_dropout: bool = True,
    dropout_rng: Optional[PRNGKey] = None,
    dropout_rate: float = 0.0,
    deterministic: bool = False,
    dtype: Dtype = jnp.float32,
    precision: PrecisionLike = None,
):
    """Computes dot-product attention weights given query and key.

    Used by :func:`dot_product_attention`, which is what you'll most likely use.
    But if you want access to the attention weights for introspection, then
    you can directly call this function and call einsum yourself.

    Args:
      query: queries for calculating attention with shape of
        `[batch..., q_length, num_heads, qk_depth_per_head]`.
      key: keys for calculating attention with shape of
        `[batch..., kv_length, num_heads, qk_depth_per_head]`.
      bias: bias for the attention weights. This should be broadcastable to the
        shape `[batch..., num_heads, q_length, kv_length]`.
        This can be used for incorporating causal masks, padding masks,
        proximity bias, etc.
      mask: mask for the attention weights. This should be broadcastable to the
        shape `[batch..., num_heads, q_length, kv_length]`.
        This can be used for incorporating causal masks.
        Attention weights are masked out if their corresponding mask value
        is `False`.
      broadcast_dropout: bool: use a broadcasted dropout along batch dims.
      dropout_rng: JAX PRNGKey: to be used for dropout
      dropout_rate: dropout rate
      deterministic: bool, deterministic or not (to apply dropout)
      dtype: the dtype of the computation (default: float32)
      precision: numerical precision of the computation see `jax.lax.Precision`
        for details.

    Returns:
      Output of shape `[batch..., num_heads, q_length, kv_length]`.
    """
    assert query.ndim == key.ndim, "q, k must have same rank."
    assert query.shape[:-3] == key.shape[:-3], "q, k batch dims must match."
    assert query.shape[-2] == key.shape[-2], "q, k num_heads must match."
    assert query.shape[-1] == key.shape[-1], "q, k depths must match."


    # calculate attention matrix
    depth = query.shape[-1]
    query = query / jnp.sqrt(depth).astype(dtype)

    # attn weight shape is (batch..., num_heads, q_length, kv_length)
    attn_weights = jnp.einsum("...qhd,...khd->...hqk", query, key, precision=precision)

    # apply attention bias: masking, dropout, proximity bias, etc.
    if bias is not None:
        attn_weights = attn_weights + bias
    # apply attention mask
    if mask is not None:
        big_neg = jnp.finfo(dtype).min
        attn_weights = jnp.where(mask, attn_weights, big_neg)

    # normalize the attention weights
    attn_weights = jax.nn.softmax(attn_weights).astype(dtype)

    # apply attention dropout
    if not deterministic and dropout_rate > 0.0:
        keep_prob = 1.0 - dropout_rate
        if broadcast_dropout:
            # dropout is broadcast across the batch + head dimensions
            dropout_shape = tuple([1] * (key.ndim - 2)) + attn_weights.shape[-2:]
            keep = random.bernoulli(dropout_rng, keep_prob, dropout_shape)
        else:
            keep = random.bernoulli(dropout_rng, keep_prob, attn_weights.shape)
        multiplier = keep.astype(attn_weights.dtype) / jnp.asarray(
            keep_prob, dtype=dtype
        )
        attn_weights = attn_weights * multiplier

    return attn_weights


def dot_product_attention(
    query: Array,
    key: Array,
    value: Array,
    bias: Optional[Array] = None,
    mask: Optional[Array] = None,
    broadcast_dropout: bool = True,
    dropout_rng: Optional[PRNGKey] = None,
    dropout_rate: float = 0.0,
    deterministic: bool = False,
    dtype: Dtype = jnp.float32,
    precision: PrecisionLike = None,
):
    """Computes dot-product attention given query, key, and value.

    This is the core function for applying attention based on
    https://arxiv.org/abs/1706.03762. It calculates the attention weights given
    query and key and combines the values using the attention weights.

    Note: query, key, value needn't have any batch dimensions.

    Args:
      query: queries for calculating attention with shape of
        `[batch..., q_length, num_heads, qk_depth_per_head]`.
      key: keys for calculating attention with shape of
        `[batch..., kv_length, num_heads, qk_depth_per_head]`.
      value: values to be used in attention with shape of
        `[batch..., kv_length, num_heads, v_depth_per_head]`.
      bias: bias for the attention weights. This should be broadcastable to the
        shape `[batch..., num_heads, q_length, kv_length]`.
        This can be used for incorporating causal masks, padding masks,
        proximity bias, etc.
      mask: mask for the attention weights. This should be broadcastable to the
        shape `[batch..., num_heads, q_length, kv_length]`.
        This can be used for incorporating causal masks.
        Attention weights are masked out if their corresponding mask value
        is `False`.
      broadcast_dropout: bool: use a broadcasted dropout along batch dims.
      dropout_rng: JAX PRNGKey: to be used for dropout
      dropout_rate: dropout rate
      deterministic: bool, deterministic or not (to apply dropout)
      dtype: the dtype of the computation (default: float32)
      precision: numerical precision of the computation see `jax.lax.Precision`
        for details.

    Returns:
      Output of shape `[batch..., q_length, num_heads, v_depth_per_head]`.
    """
    assert key.ndim == query.ndim == value.ndim, "q, k, v must have same rank."
    assert (
        query.shape[:-3] == key.shape[:-3] == value.shape[:-3]
    ), "q, k, v batch dims must match."
    assert (
        query.shape[-2] == key.shape[-2] == value.shape[-2]
    ), "q, k, v num_heads must match."
    assert key.shape[-3] == value.shape[-3], "k, v lengths must match."

    # compute attention weights
    attn_weights = dot_product_attention_weights(
        query,
        key,
        bias,
        mask,
        broadcast_dropout,
        dropout_rng,
        dropout_rate,
        deterministic,
        dtype,
        precision,
    )

    # return weighted sum over values for each query position
    # note that value and key are initialized to be the same `input_kv`
    # `q` and `k` and query and key lengths, respectively
    # `d` is the input D_emb divided by H_heads
    return jnp.einsum("...hqk,...khd->...qhd", attn_weights, value, precision=precision)


class MultiHeadDotProductAttention(Module):
    """Multi-head dot-product attention.

    Attributes:
      num_heads: number of attention heads. Features (i.e. inputs_q.shape[-1])
        should be divisible by the number of heads.
      dtype: the dtype of the computation (default: float32)
      param_dtype: the dtype passed to parameter initializers (default: float32).
      qkv_features: dimension of the key, query, and value.
      out_features: dimension of the last projection
      broadcast_dropout: bool: use a broadcasted dropout along batch dims.
      dropout_rate: dropout rate
      deterministic: if false, the attention weight is masked randomly
        using dropout, whereas if true, the attention weights
        are deterministic.
      precision: numerical precision of the computation see `jax.lax.Precision`
        for details.
      kernel_init: initializer for the kernel of the Dense layers.
      bias_init: initializer for the bias of the Dense layers.
      use_bias: bool: whether pointwise QKVO dense transforms use bias.
      attention_fn: dot_product_attention or compatible function. Accepts
        query, key, value, and returns output of shape
        `[bs, dim1, dim2, ..., dimN,, num_heads, value_channels]``
      decode: whether to prepare and use an autoregressive cache.
    """

    num_heads: int
    dtype: Dtype = jnp.float32
    param_dtype: Dtype = jnp.float32
    qkv_features: Optional[int] = None
    out_features: Optional[int] = None
    broadcast_dropout: bool = True
    dropout_rate: float = 0.0
    deterministic: Optional[bool] = None
    precision: PrecisionLike = None
    kernel_init: Callable[[PRNGKey, Shape, Dtype], Array] = default_kernel_init
    bias_init: Callable[[PRNGKey, Shape, Dtype], Array] = zeros
    use_bias: bool = True
    attention_fn: Callable[[Array, Array, Array], Array] = dot_product_attention
    decode: bool = False

    @compact
    def __call__(
        self,
        inputs_q: Array,
        inputs_kv: Array,
        mask: Optional[Array] = None,
        deterministic: Optional[bool] = None,
    ):
        """Applies multi-head dot product attention on the input data.

        Projects the inputs into multi-headed query, key, and value vectors,
        applies dot-product attention and project the results to an output vector.

        Args:
          inputs_q: input queries of shape
            `[batch_sizes..., length, features]`.
          inputs_kv: key/values of shape
            `[batch_sizes..., length, features]`.
          mask: attention mask of shape
            `[batch_sizes..., num_heads, query_length, key/value_length]`.
            Attention weights are masked out if their corresponding mask value
            is `False`.
          deterministic: if false, the attention weight is masked randomly
            using dropout, whereas if true, the attention weights
            are deterministic.

        Returns:
          output of shape `[batch_sizes..., length, features]`.
        """
        features = self.out_features or inputs_q.shape[-1]
        qkv_features = self.qkv_features or inputs_q.shape[-1]
        assert (
            qkv_features % self.num_heads == 0
        ), "Memory dimension must be divisible by number of heads."
        head_dim = qkv_features // self.num_heads

        dense = functools.partial(
            DenseGeneral,
            axis=-1,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            features=(self.num_heads, head_dim),
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
            use_bias=self.use_bias,
            precision=self.precision,
        )
        # project inputs_q to multi-headed q/k/v
        # dimensions are then [batch..., length, n_heads, n_features_per_head]
        # i.e. query, key, value shapes: (N, M, L, D_kv, D_kv, D_emb / H_heads)
        query, key, value = (
            dense(name="query")(inputs_q),
            dense(name="key")(inputs_kv),
            dense(name="value")(inputs_kv),
        )

        # During fast autoregressive decoding, we feed one position at a time,
        # and cache the keys and values step by step.
        if self.decode:
            # detect if we're initializing by absence of existing cache data.
            is_initialized = self.has_variable("cache", "cached_key")
            cached_key = self.variable(
                "cache", "cached_key", jnp.zeros, key.shape, key.dtype
            )
            cached_value = self.variable(
                "cache", "cached_value", jnp.zeros, value.shape, value.dtype
            )
            cache_index = self.variable(
                "cache", "cache_index", lambda: jnp.array(0, dtype=jnp.int32)
            )
            if is_initialized:
                (
                    *batch_dims,
                    max_length,
                    num_heads,
                    depth_per_head,
                ) = cached_key.value.shape
                # shape check of cached keys against query input
                expected_shape = tuple(batch_dims) + (1, num_heads, depth_per_head)
                if expected_shape != query.shape:
                    raise ValueError(
                        "Autoregressive cache shape error, "
                        "expected query shape %s instead got %s."
                        % (expected_shape, query.shape)
                    )
                # update key, value caches with our new 1d spatial slices
                cur_index = cache_index.value
                indices = (0,) * len(batch_dims) + (cur_index, 0, 0)
                key = lax.dynamic_update_slice(cached_key.value, key, indices)
                value = lax.dynamic_update_slice(cached_value.value, value, indices)
                cached_key.value = key
                cached_value.value = value
                cache_index.value = cache_index.value + 1
                # causal mask for cached decoder self-attention:
                # our single query position should only attend to those key
                # positions that have already been generated and cached,
                # not the remaining zero elements.
                mask = combine_masks(
                    mask,
                    jnp.broadcast_to(
                        jnp.arange(max_length) <= cur_index,
                        tuple(batch_dims) + (1, 1, max_length),
                    ),
                )

        dropout_rng = None
        if self.dropout_rate > 0.0:  # Require `deterministic` only if using dropout.
            m_deterministic = merge_param(
                "deterministic", self.deterministic, deterministic
            )
            if not m_deterministic:
                dropout_rng = self.make_rng("dropout")
        else:
            m_deterministic = True

        # apply attention
        x = self.attention_fn(
            query,
            key,
            value,
            mask=mask,
            dropout_rng=dropout_rng,
            dropout_rate=self.dropout_rate,
            broadcast_dropout=self.broadcast_dropout,
            deterministic=m_deterministic,
            dtype=self.dtype,
            precision=self.precision,
        )  # pytype: disable=wrong-keyword-args
        # back to the original inputs dimensions
        out = DenseGeneral(
            features=features,
            axis=(-2, -1),
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
            use_bias=self.use_bias,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            precision=self.precision,
            name="out",
        )(x)
        return out


class SelfAttention(MultiHeadDotProductAttention):
    """Self-attention special case of multi-head dot-product attention."""

    @compact
    def __call__(
        self,
        inputs_q: Array,
        mask: Optional[Array] = None,
        deterministic: Optional[bool] = None,
    ):
        """Applies multi-head dot product self-attention on the input data.

        Projects the inputs into multi-headed query, key, and value vectors,
        applies dot-product attention and project the results to an output vector.

        Args:
          inputs_q: input queries of shape
            `[batch_sizes..., length, features]`.
          mask: attention mask of shape
            `[batch_sizes..., num_heads, query_length, key/value_length]`.
            Attention weights are masked out if their corresponding mask value
            is `False`.
          deterministic: if false, the attention weight is masked randomly
            using dropout, whereas if true, the attention weights
            are deterministic.

        Returns:
          output of shape `[batch_sizes..., length, features]`.
        """
        return super().__call__(inputs_q, inputs_q, mask, deterministic=deterministic)


@struct.dataclass
class TransformerConfig:
    input_vocab_size: int
    output_size: int
    emb_dim: int = 512
    n_heads: int = 8
    n_layers: int = 6
    d_qkv: int = 64
    d_mlp: int = 2048
    max_len: int = 10_000 # code will not work on matrices with more this many elements
    dropout_rate: float = 0.3
    kernel_init: Callable = nn.initializers.xavier_uniform()
    bias_init: Callable = nn.initializers.normal(stddev=1e-6)


def sinusoidal_init(max_len=2048):
    """1D Sinusoidal Position Embedding Initializer. Copied from implementatoin in model.py
    Args:
        max_len: maximum possible length for the input
    Returns:
        output: init function returning `(1, max_len, d_feature)`
    """

    def init(key, shape, dtype=np.float32):
        """Sinusoidal init."""
        del key, dtype
        d_feature = shape[-1]
        pe = np.zeros((max_len, d_feature), dtype=np.float32)
        position = np.arange(0, max_len)[:, np.newaxis]
        div_term = np.exp(np.arange(0, d_feature, 2) * -(np.log(10000.0) / d_feature))
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)
        pe = pe[np.newaxis, :, :]  # [1, max_len, d_feature]
        return jnp.array(pe)

    return init


class AddPositionalEmbedding(Module):
    """Adds sinusoidal positional embeddings to the inputs
    """

    config: TransformerConfig

    @compact
    def __call__(self, inputs):
        """
        Args:
            inputs: shape of [batch, n_tokens, d_input]
        Returns:
            output of shape [batch, n_tokens, d_input]
        """
        assert inputs.ndim == 3
        pos_embeddings_shape = (1, self.config.max_len, inputs.shape[-1])
        pos_embedding = sinusoidal_init(max_len=self.config.max_len)(None, pos_embeddings_shape, None)
        pe = pos_embedding[:, :inputs.shape[1], :]
        return inputs + pe


class PositionwiseFeedForward(Module):
    config: TransformerConfig

    @compact
    def __call__(self, inputs, deterministic):
        """
        Args:
            inputs: shape of [batch, n_tokens, d_input]
        Returns:
            output of shape [batch, n_tokens, d_input]
        """
        out_dim = inputs.shape[-1]
        x = DenseGeneral(
            features=self.config.d_mlp,
            kernel_init=self.config.kernel_init,
            bias_init=self.config.bias_init
        )(inputs)
        x = nn.elu(x)
        x = nn.Dropout(rate=self.config.dropout_rate)(x, deterministic=deterministic)
        x = DenseGeneral(
            features=out_dim,
            kernel_init=self.config.kernel_init,
            bias_init=self.config.bias_init
        )(x)
        return nn.Dropout(rate=self.config.dropout_rate)(x, deterministic=deterministic)


class TransformerEncoderLayer(nn.Module):
    config: TransformerConfig

    @compact
    def __call__(self, inputs, deterministic=None):
        """Applies one layer of the Transformer encoder.
        Args:
            inputs: shape of [batch, n_tokens, d_input]
        Returns: output of shape [batch, n_tokens, d_input]
        """
        assert inputs.ndim == 3
        x = nn.LayerNorm()(inputs)

        x = SelfAttention(
            num_heads=self.config.n_heads,
            qkv_features=self.config.d_qkv*self.config.n_heads,
            kernel_init=self.config.kernel_init,
            bias_init=self.config.bias_init,
            dropout_rate=self.config.dropout_rate
        )(x, deterministic=deterministic)
        x = nn.LayerNorm()(x + inputs)
        y = PositionwiseFeedForward(self.config)(x, deterministic=deterministic)
        return x + y


class Transformer(nn.Module):
    config: TransformerConfig

    @compact
    def __call__(self, inputs, deterministic=None):
        """Applies a Transformer model on inputs of arbitrary size. Performs full self-attention
        between all tokens. Because the MultiHeadSelfAttention module only works only on
        1-dimensional arrays, the input is flattened first.
        Args:
            inputs: shape of [batch, input_shape...]
        Returns:
            output of shape [batch, input_shape..., output_size]
        """
        assert inputs.ndim >= 2

#         x = inputs.astype("int32")

        # Flatten the input
        n_tokens = math.prod([inputs.shape[i] for i in range(1, inputs.ndim)])
        x = jnp.reshape(inputs, (inputs.shape[0], n_tokens))
        
        x = nn.Embed(
            num_embeddings=self.config.input_vocab_size,
            features = self.config.emb_dim,
            name="embed"
        )(x)
        x = nn.Dropout(rate=self.config.dropout_rate)(x, deterministic=deterministic)
        x = AddPositionalEmbedding(self.config)(x)
        
        cfg = self.config
        
        for i in range(self.config.n_layers):
            x = TransformerEncoderLayer(cfg)(x, deterministic=deterministic)

        x = nn.LayerNorm()(x)
        outputs = nn.Dense(
            self.config.output_size,
            kernel_init=self.config.kernel_init,
            bias_init=self.config.bias_init
        )(x)

        # Unflatten the outputs
        outputs = jnp.reshape(outputs, (*inputs.shape, outputs.shape[-1]))
        return outputs


def test_self_attention():
    print("Checking self attention on 1D sequence")
    batch_size, length, dim = 10, 128, 200
    n_heads = 6

    rkey = random.PRNGKey(0)
    k1, k2 = random.split(rkey, 2)
    x = random.uniform(k1, (batch_size, length, dim))

    model = MultiHeadSelfAttention(n_heads=n_heads)
    params = model.init(k2, x)
    y = model.apply(params, x)

    desired_shape = (batch_size, length, dim)
    assert y.shape == desired_shape, f"output shape is {y.shape} instead of {desired_shape}"


def test_transformer_on_tensor(input_shape: tuple):
    """Tests forward pass of a transformer on an arbitrary tensor.
    Args:
        input_shape: first dimension must be the batch size
    """
    print(f"Running transformer on input with shape: {input_shape}")
    vocab_size = 200
    config = TransformerConfig(
        input_vocab_size=vocab_size,
        output_vocab_size=vocab_size
    )

    rng = random.PRNGKey(42)
    input_rng, init_rng = random.split(rng)
    dropout_rng = random.PRNGKey(10)

    input_logits = random.uniform(input_rng, (*input_shape, vocab_size))
    inputs = random.categorical(input_rng, input_logits, axis=-1)

    transformer = Transformer(config)
    params = transformer.init({"params": init_rng, "dropout": dropout_rng}, inputs)
    y = transformer.apply(params, inputs, rngs={"dropout": dropout_rng})

    desired_shape = (*input_shape, vocab_size)
    assert y.shape == desired_shape, f"output shape is {y.shape} instead of {desired_shape}"


if __name__ == '__main__':
    test_self_attention()
    test_transformer_on_tensor((10, 500))
    test_transformer_on_tensor((10, 30, 30))
    test_transformer_on_tensor((10, 10, 10, 10))
    test_transformer_on_tensor((10, 5, 5, 5, 5))