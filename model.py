import jax
import flax.linen as nn
import dataclasses

import torch


@dataclasses.dataclass
class ModelConfig:
    attention_heads: int = 3
    embed_dim: int = 64
    ffn_embed_dim: int = 64
    add_bias_kv: bool = True
    use_esm1b_layer_norm: bool = False

cfg = ModelConfig()

self_attn = nn.MultiHeadDotProductAttention(
    num_heads=cfg.attention_heads,
)




class TransformerLayer(nn.Module):

    def setup(self):

    """Transformer layer block."""

    def __init__(
        self,
        embed_dim,
        ffn_embed_dim,
        attention_heads,
        add_bias_kv=True,
        use_esm1b_layer_norm=False,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.ffn_embed_dim = ffn_embed_dim
        self.attention_heads = attention_heads
        self._init_submodules(add_bias_kv, use_esm1b_layer_norm)

    def _init_submodules(self, add_bias_kv, use_esm1b_layer_norm):
        BertLayerNorm = ESM1bLayerNorm if use_esm1b_layer_norm else ESM1LayerNorm

        self.self_attn = MultiheadAttention(
            self.embed_dim,
            self.attention_heads,
            add_bias_kv=add_bias_kv,
            add_zero_attn=False,
        )
        self.self_attn_layer_norm = BertLayerNorm(self.embed_dim)

        self.fc1 = nn.Linear(self.embed_dim, self.ffn_embed_dim)
        self.fc2 = nn.Linear(self.ffn_embed_dim, self.embed_dim)

        self.final_layer_norm = BertLayerNorm(self.embed_dim)

    def forward(
        self, x, self_attn_mask=None, self_attn_padding_mask=None, need_head_weights=False
    ):
        residual = x
        x = self.self_attn_layer_norm(x)
        x, attn = self.self_attn(
            query=x,
            key=x,
            value=x,
            key_padding_mask=self_attn_padding_mask,
            need_weights=True,
            need_head_weights=need_head_weights,
            attn_mask=self_attn_mask,
        )
        x = residual + x

        residual = x
        x = self.final_layer_norm(x)
        x = gelu(self.fc1(x))
        x = self.fc2(x)
        x = residual + x

        return x, attn




def recombination(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """


    :param a: torch.Tensor
        Shape: (L,), tokenied
    :param b:
    :param a: torch.Tensor
        Shape: (L,), tokenied
    :return:
    """

    a, b = torch.as_tensor(a).squeeze(), torch.as_tensor(b).squeeze()
    assert len(a.shape) == len(b.shape) == 1, "Inputs must be of shape (L,)"

    cut_off = np.random.randint(0, min(len(a), len(b)))
    out = a[:cut_off] + b[cut_off]
    return out
