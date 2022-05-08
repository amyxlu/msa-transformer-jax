import functools
from typing import Any, Callable, Optional, Tuple

from flax.linen.initializers import zeros
import flax.linen as nn
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

import esm
import os
from pathlib import Path
import matplotlib.pyplot as plt
from Bio import SeqIO
import esm
import torch
import os
import itertools
from typing import List, Tuple
import string
import dataclasses

PRNGKey = Any
Shape = Tuple[int]
Dtype = Any
Array = Any


def load_msas():
    #msa_dir = Path("/home/amylu/variant_effect_prediction/data/MSAs")
    # msa_dir = Path("/nfs/kun2/users/amylu/project_data_storage/variant_effect_prediction_data/MSAs")
    msa_dir = Path("/Users/amylu/Research/vareffect/MSAs")
    all_files = os.listdir(msa_dir)
    all_files.sort()
    files = all_files[1:5]

    # This is an efficient way to delete lowercase characters and insertion characters from a string
    deletekeys = dict.fromkeys(string.ascii_lowercase)
    deletekeys["."] = None
    deletekeys["*"] = None
    translation = str.maketrans(deletekeys)

    def read_sequence(filename: str) -> Tuple[str, str]:
        """ Reads the first (reference) sequences from a fasta or MSA file."""
        record = next(SeqIO.parse(filename, "fasta"))
        return record.description, str(record.seq)

    def remove_insertions(sequence: str) -> str:
        """ Removes any insertions into the sequence. Needed to load aligned sequences in an MSA. """
        return sequence.translate(translation)

    def read_msa(filename: str, nseq: int) -> List[Tuple[str, str]]:
        """ Reads the first nseq sequences from an MSA file, automatically removes insertions."""
        return [(record.description, remove_insertions(str(record.seq)))
                for record in itertools.islice(SeqIO.parse(filename, "fasta"), nseq)]


    msa_transformer, msa_alphabet = esm.pretrained.esm_msa1b_t12_100M_UR50S()
    # msa_transformer = msa_transformer.eval().cuda()
    msa_batch_converter = msa_alphabet.get_batch_converter()

    msa_data = [read_msa(msa_dir / fname, 64) for fname in files]
    msa_batch_labels, msa_batch_strs, msa_batch_tokens = msa_batch_converter(msa_data)
    # (N_msas, max_msa_depth, max_seq_len)

    return msa_batch_tokens


def dot_product_attention_weights(
        query: Array,
        key: Array,
        mask: Optional[Array] = None,
        broadcast_dropout: bool = True,
        dropout_rng: Optional[PRNGKey] = None,
        dropout_rate: float = 0.0,
        deterministic: bool = False,
        dtype: Dtype = jnp.float32,
):
    """https://flax.readthedocs.io/en/latest/_modules/flax/linen/attention.html#SelfAttention

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

class FeedForwardNetwork(nn.Module):
    # TODO(axl): architecutre should match
    #  https://github.com/rmrao/msa-transformer/blob/ebd9551e4fe1a8b3dd85a45ad25c6142fdf0b487/modules.py#L404
    pass


class NormalizedResidualBlock(nn.Module):
    config: MSATransformerConfig
    layer: Callable
    # TODO(axl): Deterministic option?

    @nn.compact
    def __call__(self, inputs, deterministic):
        cfg = self.config

        x = nn.LayerNorm()(inputs)
        x = self.layer(x)
        x = nn.Dropout(rate=cfg.dropout_rate)(x, deterministic=deterministic)
        return x + inputs

class MSAEncoderBlock(nn.Module):
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
    def __call__(self, inputs, deterministic):
        """Applies the MSA encoder module to an input batch of MSAs

        Args:
          inputs: Input batch of MSAs. Shape: (N_msas, num_rows, num_cols, H_heads, D_emb).
          deterministic: if true dropout is applied otherwise not.
        Returns:
          Output (i.e. embedding) after transformer encoder for MLM loss, etc.
        """
        cfg = self.config
        assert inputs.ndim == 5

        self_attn = nn.SelfAttention(
            num_heads=cfg.attention_heads,
            use_bias=cfg.use_attn_weight_bias,
            broadcast_dropout=False,  #TODO(axl) do we need dropout across M?
            dropout_rate=cfg.attention_dropout,
            deterministic=deterministic,
        )
        row_attn_block = NormalizedResidualBlock(cfg, self_attn)
        column_attn_block = NormalizedResidualBlock(cfg, self_attn)
        mlp_block = NormalizedResidualBlock(cfg, nn.Dense()) #TODO: n dimensions

        x = row_attn_block(inputs)
        x = column_attn_block(x)
        x = mlp_block(x)

        return x

    # Initialize dummy arrays
    N_msas = 8
    num_cols = 100
    num_rows = 12
    H_heads = 4
    D_emb = 256

    rkey = random.PRNGKey(0)
    k1, k2, k3, k4 = random.split(rkey, 4)

    # Multi head attention
    multi_head_attn = nn.MultiHeadDotProductAttention(H_heads)

    row_input_q = random.randint(k1, (N_msas, num_rows, num_cols, H_heads, D_emb), 0, 20)
    col_input_q = random.randint(k2, (N_msas, num_cols, num_rows, H_heads, D_emb), 0, 20)

    mh_attn_params = multi_head_attn.init(k1, row_input_q, row_input_q)
    out = multi_head_attn.apply(mh_attn_params, row_input_q, row_input_q)