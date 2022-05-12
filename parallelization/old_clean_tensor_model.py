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

PRNGKey = Any
Shape = Tuple[int]
Dtype = Any
Array = Any


def dot_product_attention(
    Q: Array,
    K: Array,
    V: Array,
    dropout: nn.Module,
    deterministic: Optional[bool] = None
):
    """Computes dot-product attention given query key, and value matrices.
    Currently, does not support masking padding tokens.

    Combines dot_product_attention_weights and dot_product_attention from model.py
    
    Args:
        Q: query matrix with shape [batch, num_tokens, num_heads, d_qkv]
        K: key matrix with shape [batch, num_tokens, num_heads, d_qkv]
        V: values matrix with shape [batch, num_tokens, num_heads, d_qkv]
        dropout: dropout module
    Returns:
        output of shape [batch, num_tokens, num_heads, d_qkv]
    """
    assert Q.shape == K.shape == V.shape

    '''eisnum abbreviations:
        - h: num heads
        - q: num query tokens
        - k: num key tokens
        - d: d_qkv
    '''
    attention_logits = jnp.einsum("...qhd,...khd->...hqk", Q, K) # [batch, num_heads, num_tokens, num_tokens]
    attention_logits /= jnp.sqrt(Q.shape[-1])
    attention_probs = jax.nn.softmax(attention_logits, axis=-1)
    attention_probs = dropout(attention_probs, deterministic=deterministic)
    
    out = jnp.einsum("...hqk,...khd->...qhd", attention_probs, V)
    return dropout(out, deterministic=deterministic)


class MultiHeadSelfAttention(Module):
    """Multi-head self attention.
    Does not support masking or special autoregressive cache during decoding.
    """

    n_heads: Optional[int] = 8
    d_qkv: Optional[int] = 64
    kernel_init: Callable[[PRNGKey, Shape, Dtype], Array] = default_kernel_init
    bias_init: Callable[[PRNGKey, Shape, Dtype], Array] = zeros
    dropout_rate: float = 0.0

    @compact
    def __call__(
        self,
        inputs: Array,
        deterministic: Optional[bool] = None
    ):
        """
        Applies multi-head dot product self-attention on the input data.

        Args:
            inputs: input of shape [batch, n_tokens, d_input]
        Returns:
            output of shape [batch, n_tokens, d_input]
        """
        dense_QKV = functools.partial(
            DenseGeneral,
            axis=-1,
            features=(self.n_heads, self.d_qkv),
            kernel_init=self.kernel_init,
            bias_init=self.bias_init
        )

        # project inputs to multi-headed Q, K, V
        # dimensions for all three matrices are [batch, n_tokens, n_heads, d_qkv]
        Q, K, V = (
            dense_QKV(name='query')(inputs),
            dense_QKV(name='key')(inputs),
            dense_QKV(name='value')(inputs)
        )

        out = dot_product_attention(Q, K, V, nn.Dropout(rate=self.dropout_rate), deterministic=deterministic)
        return DenseGeneral(
            features=inputs.shape[-1],
            axis=(-2, -1),
            name='out',
            kernel_init=self.kernel_init,
            bias_init=self.bias_init
        )(out)


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
    def __call__(self, inputs, deterministic=None):
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

        x = MultiHeadSelfAttention(
            n_heads=self.config.n_heads,
            d_qkv=self.config.d_qkv,
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

        x = inputs.astype("int32")

        # Flatten the input
        n_tokens = math.prod([inputs.shape[i] for i in range(1, inputs.ndim)])
        x = jnp.reshape(x, (inputs.shape[0], n_tokens))
        
        x = nn.Embed(
            num_embeddings=self.config.input_vocab_size,
            features = self.config.emb_dim,
            name="embed"
        )(x)
        x = nn.Dropout(rate=self.config.dropout_rate)(x, deterministic=deterministic)
        x = AddPositionalEmbedding(self.config)(x)

        for _ in range(self.config.n_layers):
            x = TransformerEncoderLayer(self.config)(x, deterministic=deterministic)

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