from typing import Any, Tuple, Callable
from functools import partial

import flax.linen as nn
import jax
import jax.numpy as jnp
import math

from configs import MSATransformerConfig


Shape = Tuple[int]
Array = Any


def masked_fill(mask, a, fill):
    return jnp.where(mask, a, fill_value=fill)


class RowSelfAttention(nn.Module):
    config: MSATransformerConfig

    @nn.compact
    def __call__(
        self, x: Array, deterministic: bool = False, self_attn_padding_mask: Array = None
    ):
        """

        :param x:
        :param self_attn_padding_mask:
        :param deterministic:
        :return: Array of size (num_attention_heads, num
        """
        cfg = self.config
        head_dim = cfg.embed_dim // cfg.attention_heads

        num_rows, num_cols, batch_size, embed_dim = x.shape
        shape = (num_rows, num_cols, batch_size, cfg.attention_heads, head_dim)
        q = nn.Dense(embed_dim)(x).reshape(shape)
        k = nn.Dense(embed_dim)(x).reshape(shape)
        v = nn.Dense(embed_dim)(x).reshape(shape)

        # dividing by sqrt(num_rows) is only done for row self attention in original code
        scaling = (head_dim**-0.5) / math.sqrt(num_rows)
        q *= scaling

        # For row attention, zero out any padded aligned positions for query;
        # this is important since we take a sum across the alignment axis.
        if self_attn_padding_mask is not None:
            # mask shape: (batch_size, n_rows, n_cols)
            # (B, R, C) -> (R, C, B)
            q_mask = self_attn_padding_mask.copy()
            q_mask = jnp.transpose(q_mask, axes=(1, 2, 0))

            # (R, C, B) -> (R, C, B, 1, 1)
            q_mask = jnp.expand_dims(q_mask, axis=(3, 4))

            # (R, C, B, 1, 1) -> (R, C, B, H, D)
            q_mask = jnp.broadcast_to(q_mask, q.shape)
            q *= 1 - q_mask

        # For the final version, tied attention is used, so the row dimension "disappears"
        # such that all rows share the same attention weights.
        # https://github.com/rmrao/msa-transformer/blob/main/modules.py#L774
        attn_weights = jnp.einsum(f"rinhd,rjnhd->hnij", q, k)

        if self_attn_padding_mask is not None:
            # (B, R, C) -> (H, B, C, C)
            attn_mask = jnp.expand_dims(self_attn_padding_mask[:, 0, :], axis=(0,3))
            attn_mask = jnp.broadcast_to(attn_mask, attn_weights.shape)
            import pdb;pdb.set_trace()
            attn_weights = jnp.where(condition=attn_mask, x=attn_weights, y=jnp.full(attn_weights.shape, -10000))


        attn_probs = nn.softmax(attn_weights, axis=-1)
        attn_probs = nn.Dropout(rate=cfg.attention_dropout)(
            attn_probs, deterministic=deterministic
        )

        context = jnp.einsum("hnij,rjnhd->rinhd", attn_probs, v)
        context = jnp.reshape(context, (num_rows, num_cols, batch_size, embed_dim))
        output = nn.Dense(embed_dim)(context)

        # return attn weights for inspection if needed
        return output


class ColumnSelfAttention(nn.Module):
    config: MSATransformerConfig

    @nn.compact
    def __call__(
        self, x: Array, deterministic: bool = False, self_attn_padding_mask: Array = None
    ):
        """

        :param x:
        :param self_attn_padding_mask:
        :param deterministic:
        :return: Array of size (cfg.attention_heads, num_cols, batch_size, num_rows, num_rows)
        """
        cfg = self.config
        head_dim = cfg.embed_dim // cfg.attention_heads

        num_rows, num_cols, batch_size, embed_dim = x.shape
        qkv_shape = (num_rows, num_cols, batch_size, cfg.attention_heads, head_dim)

        if num_rows == 1:
            # if there is only 1 position, this is equivalent and doesn't break with
            # padding
            attn_probs = jnp.ones(
                (cfg.attention_heads, num_cols, batch_size, num_rows, num_rows),
                dtype=x.dtype,
            )
            output = nn.Dense(embed_dim)(x)

        else:
            q = nn.Dense(embed_dim)(x).reshape(qkv_shape)
            k = nn.Dense(embed_dim)(x).reshape(qkv_shape)
            v = nn.Dense(embed_dim)(x).reshape(qkv_shape)

            # dividing by sqrt(num_rows) is only done for row self attention in original code
            scaling = head_dim ** -0.5
            q *= scaling

            # https://github.com/rmrao/msa-transformer/blob/main/modules.py#L907
            attn_weights = jnp.einsum("icnhd,jcnhd->hcnij", q, k)
            if self_attn_padding_mask is not None:
                self_attn_padding_mask = jnp.transpose(
                    self_attn_padding_mask, (2, 0, 1)
                )
                self_attn_padding_mask = jnp.expand_dims(self_attn_padding_mask, (0, 3))
                attn_weights = masked_fill(self_attn_padding_mask, attn_weights, -10000)

            attn_probs = nn.softmax(attn_weights, axis=-1)
            attn_probs = nn.Dropout(rate=cfg.attention_dropout)(
                attn_probs, deterministic=deterministic
            )

            # https://github.com/rmrao/msa-transformer/blob/main/modules.py#L919
            context = jnp.einsum("hcnij,jcnhd->icnhd", attn_probs, v)
            context = jnp.reshape(context, (num_rows, num_cols, batch_size, embed_dim))
            output = nn.Dense(embed_dim)(context)

        # return attn weights for inspection if needed
        return output


class FeedForwardNetwork(nn.Module):
    config: MSATransformerConfig

    @nn.compact
    def __call__(self, x, deterministic: bool = False):
        cfg = self.config
        x = nn.Dense(cfg.ffn_embed_dim)(x)
        x = nn.gelu(x)
        x = nn.Dense(cfg.embed_dim)(x)
        x = nn.Dropout(rate=cfg.activation_dropout)(x, deterministic=deterministic)
        return x


class AxialMSAEncoderBlock(nn.Module):
    """
    See Fig 1. (right) of Rao et al. Each encoder block consists of
    a row self-attention block, a column self-attention block, and
    a feedforward block. Each block is a residual block with a layer
    norm before attention block.

    Attributes:
      config: TransformerConfig dataclass containing hyperparameters.
    """

    config: MSATransformerConfig

    @nn.compact
    def __call__(self, inputs: Array, deterministic: bool = False, self_attn_padding_mask: Array = None):
        """Applies the MSA encoder module to an input batch of MSAs

        Args:
          inputs: Input batch of MSAs. Shape: (N_msas, num_rows, num_cols, H_heads, D_emb).
          deterministic: if true dropout is applied otherwise not.
        Returns:
          Output (i.e. embedding) after transformer encoder for MLM loss, etc.
        """
        cfg = self.config

        row_attn = RowSelfAttention(cfg)
        column_attn = ColumnSelfAttention(cfg)
        ffn = FeedForwardNetwork(cfg)
        x = inputs.copy()

        # Row attention residual block w/ normalization
        x = row_attn(nn.LayerNorm()(x), deterministic, self_attn_padding_mask)
        x = nn.Dropout(rate=cfg.dropout)(x, deterministic)

        # Column attention residual block w/ normalization
        x = column_attn(nn.LayerNorm()(x), deterministic, self_attn_padding_mask)
        x = nn.Dropout(rate=cfg.dropout)(x, deterministic)

        # Feed-forward residual block w/ normalization
        x = ffn(nn.LayerNorm()(x), deterministic)
        x = nn.Dropout(rate=cfg.dropout)(x, deterministic)

        return x


if __name__ == "__main__":
    from jax import random
    from configs import MSATransformerConfig

    rng = random.PRNGKey(0)
    init_rng, dropout_rng, input_rng = random.split(rng, 3)
    cfg = MSATransformerConfig()

    # Initialize dummy arrays
    # https://github.com/facebookresearch/esm/blob/main/esm/model.py#L367
    # (R, C, B, D)
    N_msas = 8
    num_cols = 100
    num_rows = 12
    inputs = random.randint(input_rng, (num_rows, num_cols, N_msas, cfg.embed_dim), 0, 20)
    encoder = AxialMSAEncoderBlock(cfg)
    encoder_params = encoder.init({"params": init_rng, "dropout": dropout_rng}, inputs)
    out = encoder.apply(encoder_params, inputs, deterministic=True)
