from typing import Any

import flax.linen as nn
import jax.numpy as jnp

from configs import MSATransformerConfig
from axial_attention import AxialMSAEncoderBlock
from position_embeddings import AddPositionEmbs

Array = Any

class MLMHead(nn.Module):
    """Head for masked language modeling.
    https://github.com/facebookresearch/esm/blob/main/esm/modules.py#L296
    """
    config: MSATransformerConfig

    @nn.compact
    def __call__(self, x: Array):
        cfg = self.config
        vocab_size = len(self.alphabet.tok_to_idx)

        x = nn.gelu(x)
        x = nn.LayerNorm(dtype=x.dtype)(x)
        # project back to size of vocabulary with bias
        # TODO(axl): Technically should be done by passing in the exact emb weights
        # see https://github.com/facebookresearch/esm/blob/main/esm/model.py#L325
        # x = F.linear(x, self.weight) + self.bias
        x = nn.Dense(cfg.vocab_size)(x)
        return x


class MSATransformer(nn.Module):
    config: MSATransformerConfig

    @nn.compact
    def __call__(self, tokens: Array[int], train: bool = False) -> Array:
        """
        Note: unlike ESM implementation, does not return contacts,
        attention head weights, or representations from a specific layer.

        Args:
            tokens: tokenized input
            train: boolean for whether we're in training mode (for dropout)

        Returns:
            output: output to feed into MLM head.
        """
        cfg = self.config
        vocab_size = len(self.alphabet.tok_to_idx)
        assert tokens.ndim == 3
        batch_size, num_alignments, seqlen = tokens.shape
        padding_mask = jnp.equal(tokens, self.alphabet.padding_idx)  # B, R, C

        """
        Embed and add positional token
        """
        x = nn.Embed(num_embeddings=vocab_size, features=cfg.embed_dim)(tokens)

        # The positional embeddings used in MSA transformer is not a
        # sinuosoidal embedding, but a learned embedding. Ignore this for now.
        x = AddPositionEmbs(cfg)(x)  #TODO(axl): Check if shapes match

        x = nn.LayerNorm(dtype=x.dtype)(x)
        x = nn.Dropout(cfg.dropout)(x, deterministic=not train)

        x = x * (1 - padding_mask[:, None])  # todo(axl): check if shapes match

        # For some reason, this is the orientation used
        # for the attention modules. I *think* if this wasn't the case
        # we don't even need to reimplement row & column attention
        # and can just use flax.nn.SelfAttention with the right transposes
        # and einsum will automatically take care of it when calculating
        # the attention weights, since an arbitrary of batch size dimensions
        # are allowed. But just in case, this is reimplemented
        # until I can wrap my head around this :shrug:
        # The shape change is: B x R x C x D -> R x C x B x D
        x = jnp.transpose(x, (1, 2, 0, 3))

        """
        Main attention encoder blocks + MLM head.
        """
        for _ in enumerate(cfg.num_layers):
            x = AxialMSAEncoderBlock(cfg)(x, deterministic=(not train), self_attn_padding_mask=padding_mask)

        x = nn.LayerNorm(dtype=x.dtype)(x)
        x = jnp.transpose(x, (2, 0, 1, 3))  # R x C x B x D -> B x R x C x D

        representations = x.copy()
        logits = MLMHead(cfg)(x)

        return representations, logits
