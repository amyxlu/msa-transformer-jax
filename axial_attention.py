from typing import Any, Tuple, Callable

import flax.linen as nn
import jax
import jax.numpy as jnp
import math

from configs import MSATransformerConfig


Shape = Tuple[int]
Array = Any


def masked_fill(mask, a, fill):
    return jax.lax.select(mask, a, jax.lax.broadcast(fill, a.shape))


@nn.compact
def qkv_project(x: Array, embed_dim: int, shape: Shape):
    x = nn.Dense(embed_dim, embed_dim)(x)
    return jnp.reshape(x, shape)


class RowSelfAttention(nn.Module):
    config: MSATransformerConfig

    @nn.compact
    def __call__(
        self, x: Array, deterministic: bool, self_attn_padding_mask: Array = None
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
        shape = (num_rows, num_cols, batch_size, embed_dim, head_dim)
        q = qkv_project(x, shape)
        k = qkv_project(x, shape)
        v = qkv_project(x, shape)

        # dividing by sqrt(num_rows) is only done for row self attention in original code
        scaling = (head_dim**-0.5) / math.sqrt(num_rows)
        q *= scaling

        # For row attention, zero out any padded aligned positions for query;
        # this is important since we take a sum across the alignment axis.
        if self_attn_padding_mask is not None:
            self_attn_padding_mask = jnp.transpose(
                self_attn_padding_mask, axes=(1, 2, 0)
            )
            self_attn_padding_mask = jnp.expand_dims(
                self_attn_padding_mask, axis=(3, 4)
            )
            q *= 1 - self_attn_padding_mask

        # https://github.com/rmrao/msa-transformer/blob/main/modules.py#L774
        attn_weights = jnp.einsum(f"rinhd,rjnhd->hnij", q, k)
        if self_attn_padding_mask is not None:
            attn_weights = masked_fill(self_attn_padding_mask, attn_weights, -10000)

        attn_probs = nn.softmax(attn_weights, axis=-1)
        attn_probs = nn.Dropout(rate=cfg.attention_dropout)(
            attn_probs, deterministic=deterministic
        )

        context = jnp.einsum("hnij,rjnhd->rinhd", attn_probs, v)
        context = jnp.reshape(context, (num_rows, num_cols, batch_size, embed_dim))
        output = self.out_proj(context)

        # return attn weights for inspection if needed
        return output


class ColumnSelfAttention(nn.Module):
    config: MSATransformerConfig

    @nn.compact
    def __call__(
        self, x: Array, deterministic: bool, self_attn_padding_mask: Array = None
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
        qkv_shape = (num_rows, num_cols, batch_size, embed_dim, head_dim)

        if num_rows == 1:
            # if there is only 1 position, this is equivalent and doesn't break with
            # padding
            attn_probs = jnp.ones(
                (cfg.attention_heads, num_cols, batch_size, num_rows, num_rows),
                dtype=x.dtype,
            )
            output = qkv_project(qkv_project(x))

        else:
            q = qkv_project(x, qkv_shape)
            k = qkv_project(x, qkv_shape)
            v = qkv_project(x, qkv_shape)

            # dividing by sqrt(num_rows) is only done for row self attention in original code
            scaling = head_dim**-0.5
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
            output = self.out_proj(context)

        # return attn weights for inspection if needed
        return output


class FeedForwardNetwork(nn.Module):
    config = MSATransformerConfig

    @nn.compact
    def __call__(self, x, deterministic):
        cfg = self.config
        x = nn.Dense(cfg.ffn_embed_dim)(x)
        x = nn.gelu(x)
        x = nn.Dense(cfg.embed_dim)(x)
        x = nn.Dropout(rate=cfg.activation_dropout)(x, deterministic=deterministic)
        return x


class NormalizedResidualBlock(nn.Module):
    config: MSATransformerConfig
    layer: Callable

    @nn.compact
    def __call__(self, inputs, deterministic):
        cfg = self.config
        x = nn.LayerNorm()(inputs)
        x = self.layer(x, deterministic)
        x = nn.Dropout(rate=cfg.dropout)(x, deterministic=deterministic)
        return x + inputs


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
    def __call__(self, inputs, deterministic, self_attn_padding_mask=None):
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

        row_attn_block = NormalizedResidualBlock(cfg, row_attn)
        column_attn_block = NormalizedResidualBlock(cfg, column_attn)
        ffn_block = NormalizedResidualBlock(cfg, ffn)

        x = row_attn_block(inputs, deterministic, self_attn_padding_mask)
        x = column_attn_block(x, deterministic, self_attn_padding_mask)
        x = ffn_block(x, deterministic)

        return x


if __name__ == "__main__":
    from jax import random
    rkey = random.PRNGKey(0)
    k1, k2, k3, k4 = random.split(rkey, 4)

    # Initialize dummy arrays
    # https://github.com/facebookresearch/esm/blob/main/esm/model.py#L367
    # (R, C, B, D)
    N_msas = 8
    num_cols = 100
    num_rows = 12
    H_heads = 4
    D_emb = 256
    inputs = random.randint(k1, (num_rows, num_cols, N_msas, D_emb), 0, 20)

    cfg = MSATransformerConfig()
    encoder = AxialMSAEncoderBlock(cfg)

    enc_params = encoder.init(k2, inputs, deterministic=True)
    out = encoder.apply(enc_params, inputs, deterministic=True)
