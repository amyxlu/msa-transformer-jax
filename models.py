import functools
from typing import Any, Callable, Optional, Tuple
import jax
import numpy as np
from jax import lax
from jax import random
import jax.numpy as jnp

import flax.linen as nn

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

PRNGKey = Any
Shape = Tuple[int]
Dtype = Any
Array = Any


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
msa_transformer = msa_transformer.eval().cuda()
msa_batch_converter = msa_alphabet.get_batch_converter()

msa_data = [read_msa(msa_dir / fname, 64) for fname in files]
msa_batch_labels, msa_batch_strs, msa_batch_tokens = msa_batch_converter(msa_data)
msa_batch_tokens.shape
# (N_msas, max_msa_depth, max_seq_len)