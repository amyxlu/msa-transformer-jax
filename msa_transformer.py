from typing import Any

import flax.linen as nn
import jax.numpy as jnp

from configs import MSATransformerConfig
from alphabet import MSATransformerAlphabet
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
    alphabet: MSATransformerAlphabet

    @nn.compact
    def __call__(self, tokens: Array, train: bool = False) -> Array:
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
        assert len(self.alphabet.tok_to_idx) == cfg.vocab_size
        assert tokens.ndim == 3
        batch_size, num_alignments, seqlen = tokens.shape
        padding_mask = jnp.equal(tokens, self.alphabet.padding_idx)  # (B, R, C)

        """
        Embed and add positional token
        """
        # (batch_size, n_rows, n_cols) -> (B, R, C, ffn_emb_dim)
        x = nn.Embed(num_embeddings=cfg.vocab_size, features=cfg.embed_dim)(tokens)

        # The positional embeddings used in MSA transformer is not a
        # sinuosoidal embedding, but a learned embedding. Ignore this for now.
        # x = AddPositionEmbs(cfg)(x)  #TODO(axl): Check if shapes match

        x = nn.LayerNorm(dtype=x.dtype)(x)
        x = nn.Dropout(cfg.dropout)(x, deterministic=not train)
        x = x * (1 - jnp.expand_dims(padding_mask, axis=3))

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
        for _ in range(cfg.num_layers):
            x = AxialMSAEncoderBlock(cfg)(x, deterministic=(not train), self_attn_padding_mask=padding_mask)

        x = nn.LayerNorm(dtype=x.dtype)(x)
        x = jnp.transpose(x, (2, 0, 1, 3))  # R x C x B x D -> B x R x C x D

        representations = x.copy()
        logits = MLMHead(cfg)(x)

        return representations, logits

if __name__ == "__main__":
    from pathlib import Path
    import os
    import string
    from Bio import SeqIO
    from typing import Tuple, List
    import itertools
    import esm
    from jax import random
    import jax

    def dummy_tokens():

        # msa_dir = Path("/home/amylu/variant_effect_prediction/data/MSAs")
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

        def read_msa(filename: Path, nseq: int) -> List[Tuple[str, str]]:
            """ Reads the first nseq sequences from an MSA file, automatically removes insertions."""
            return [(record.description, remove_insertions(str(record.seq)))
                    for record in itertools.islice(SeqIO.parse(filename, "fasta"), nseq)]

        msa_transformer, msa_alphabet = esm.pretrained.esm_msa1b_t12_100M_UR50S()
        # msa_transformer = msa_transformer.eval().cuda()
        msa_batch_converter = msa_alphabet.get_batch_converter()

        msa_data = [read_msa(msa_dir / fname, 64) for fname in files]
        msa_batch_labels, msa_batch_strs, msa_batch_tokens = msa_batch_converter(msa_data)

        return msa_batch_tokens.numpy()

    """
    Forward pass for debugging.
    """

    from jax import random
    from configs import MSATransformerConfig

    rng = random.PRNGKey(0)
    init_rng, dropout_rng, input_rng = random.split(rng, 3)
    cfg = MSATransformerConfig()
    alphabet = MSATransformerAlphabet()

    # Initialize dummy arrays
    # tokens = random.randint(input_rng, (4, 64, 448), 0, 33)
    tokens = dummy_tokens()   # (N_msas, max_msa_depth, max_seq_len)
    msa_transformer = MSATransformer(cfg, alphabet)
    msa_transformer_params = msa_transformer.init({"params": init_rng, "dropout": dropout_rng}, tokens)
    out = msa_transformer.apply(msa_transformer_params, tokens, deterministic=True)

    print(jax.tree_map(lambda x: print(x.shape), msa_transformer_params))