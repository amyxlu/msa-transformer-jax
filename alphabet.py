"""
For consistent tokenization.
https://github.com/facebookresearch/esm/blob/main/esm/data.py#L91

"""
from typing import Sequence, Dict

esm_tok_to_idx = {'<cls>': 0,
 '<pad>': 1,
 '<eos>': 2,
 '<unk>': 3,
 'L': 4,
 'A': 5,
 'G': 6,
 'V': 7,
 'S': 8,
 'E': 9,
 'R': 10,
 'T': 11,
 'I': 12,
 'D': 13,
 'P': 14,
 'K': 15,
 'Q': 16,
 'N': 17,
 'F': 18,
 'Y': 19,
 'M': 20,
 'H': 21,
 'W': 22,
 'C': 23,
 'X': 24,
 'B': 25,
 'U': 26,
 'Z': 27,
 'O': 28,
 '.': 29,
 '-': 30,
 '<null_1>': 31,
 '<mask>': 32}


class MSATransformerAlphabet:
    tok_to_idx: Dict = esm_tok_to_idx
    prepend_toks: Sequence[str] = ['<cls>', '<pad>', '<eos>', '<unk>']
    append_toks: Sequence[str] = ['<mask>']
    append_eos: bool = False
    prepend_bos: bool = True
    padding_idx: int = 1
    mask_idx: int = 32
    cls_idx: int = 0
    eos_idx: int = 2
    unk_idx: int = 3
