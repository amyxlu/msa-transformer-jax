"""
See https://github.com/facebookresearch/esm/blob/main/esm/model.py#L206
Default config for MSA Transformer (full size)
"""
import dataclasses

@dataclasses.dataclass
class MSATransformerConfig:
    num_layers: int = 12
    embed_dim: int = 768
    ffn_embed_dim: int = 3072
    attention_heads: int = 12
    dropout: float = 0.1
    attention_dropout: float = 0.1
    activation_dropout: float = 0.1
    max_tokens_per_msa: int = 2 ** 14
    vocab_size: int = 33
    use_attn_weight_bias: bool = False


"""
Smaller config for testing
"""
@dataclasses.dataclass
class SmallConfig(MSATransformerConfig):
    num_layers: int = 4
    embed_dim: int = 128
    ffn_embed_dim: int = 1024
    attention_heads: int = 6
    dropout: float = 0.1
    attention_dropout: float = 0.1
    activation_dropout: float = 0.1