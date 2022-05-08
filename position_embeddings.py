"""
https://github.com/google/flax/blob/main/examples/nlp_seq/models.py#L44
"""

import flax.linen as nn
import numpy as np
import jax.numpy as jnp

from configs import MSATransformerConfig


def sinusoidal_init(max_len=2048):
  """1D Sinusoidal Position Embedding Initializer.
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
    div_term = np.exp(
        np.arange(0, d_feature, 2) * -(np.log(10000.0) / d_feature))
    pe[:, 0::2] = np.sin(position * div_term)
    pe[:, 1::2] = np.cos(position * div_term)
    pe = pe[np.newaxis, :, :]  # [1, max_len, d_feature]
    return jnp.array(pe)

  return init


class AddPositionEmbs(nn.Module):
  """Adds (optionally learned) positional embeddings to the inputs.
  Attributes:
    config: TransformerConfig dataclass containing hyperparameters.
  """
  config: MSATransformerConfig

  @nn.compact
  def __call__(self, inputs):
    """Applies AddPositionEmbs module.
    By default this layer uses a fixed sinusoidal embedding table. If a
    learned position embedding is desired, pass an initializer to
    posemb_init in the configuration.
    Args:
      inputs: input data.
    Returns:
      output: `(bs, timesteps, in_dim)`
    """
    cfg = self.config
    # inputs.shape is (batch_size, seq_len, emb_dim)
    assert inputs.ndim == 3, ('Number of dimensions should be 3,'
                              ' but it is: %d' % inputs.ndim)
    length = inputs.shape[1]
    pos_emb_shape = (1, cfg.max_len, inputs.shape[-1])
    if cfg.posemb_init is None:
      # Use a fixed (non-learned) sinusoidal position embedding.
      pos_embedding = sinusoidal_init(max_len=cfg.max_len)(
          None, pos_emb_shape, None)
    else:
      pos_embedding = self.param('pos_embedding',
                                 cfg.posemb_init,
                                 pos_emb_shape)
    pe = pos_embedding[:, :length, :]
    return inputs + pe

