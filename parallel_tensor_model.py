"""
Parallelized version of the tensor_model transformer
"""

from doctest import OutputChecker
import functools
from typing import Any, Callable, Optional, Tuple

from flax import linen as nn, struct
from flax.linen.initializers import zeros
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
from jax.experimental import maps
from jax.experimental import PartitionSpec
from jax.experimental.pjit import pjit
import optax

import tensorflow_datasets as tfds

import numpy as np
import os
import pdb
from tqdm import tqdm

from tensor_model import *

os.environ["NCCL_LAUNCH_MODE"] = "PARALLEL"

PRNGKey = Any
Shape = Tuple[int]
Dtype = Any
Array = Any

class ParallelizedTransformer():
    def __init__(self, config, input_shape, n_gpus=2, seed=97, sample_input_batch_size=2):
        """Constructs the parallelized version of the Transformer model defined in tensor_model.py.
        Args:
            config: TransformerConfig, used to define the transformer
            input_shape: tuple/list, shape of the input (not including the batch dimension)
            n_gpus: int, the number of gpus over which the transformer will be parallelized
            seed: int, random seed used to initialize the model
            sample_input_batch_size: int, batch size used to construct the dummy sample input 
                                          which is then used to initialize the model
        """
        self.config = config
        self.n_gpus = n_gpus
        
        # create random seeds
        rng = random.PRNGKey(seed)
        init_rng, dropout_rng, input_rng = random.split(rng, num=3)
        
        # create dummy sample one-hot inputs used to initialize the model
        sample_input_shape = (sample_input_batch_size, *input_shape)
        sample_input_logits = random.uniform(input_rng, (*sample_input_shape, config.input_vocab_size))
        sample_inputs = random.categorical(input_rng, sample_input_logits, axis=-1)
        
        # create and initialize model
        self.model = Transformer(config)
        self.model_params = self.model.init({"params": init_rng, "dropout": dropout_rng}, sample_inputs)
        
        # small check to see if output dims are as expected
        y = self.model.apply(self.model_params, sample_inputs, rngs={"dropout": dropout_rng})
        desired_shape = (sample_input_batch_size, *input_shape, config.output_size)
        assert y.shape == desired_shape, f"output shape is {y.shape} instead of {desired_shape}"
        
        # create pjit mesh to shard the model along the second input dimension (usually, the length of the sequence)
        self.mesh_shape = (1, n_gpus)
        self.devices = np.asarray(jax.devices()[:n_gpus]).reshape(*self.mesh_shape)
        self.mesh = maps.Mesh(self.devices, ('batch', 'length'))
        
        self.model_pjit = pjit(lambda model_params, inputs: self.model.apply(model_params, inputs, 
                                                                             rngs={"dropout": dropout_rng}),
                               in_axis_resources=[None, PartitionSpec('batch', 'length')],
                               out_axis_resources=PartitionSpec('batch', 'length'))
    
    def forward(self, params, inputs):
        """Forward pass of the model
        Args:
            params: the parameters of the model
            inputs: the inputs to the model for the forward pass
        Returns:
            logits: the outputs of the forward pass
        """            
        with maps.Mesh(self.mesh.devices, self.mesh.axis_names):
            logits = self.model_pjit(params, inputs)
        return logits
    
    def create_train_state(self, optimizer_module, **optimize_params):
        """Creates initial `TrainState`."""
        self.optimizer = optimizer_module(**optimize_params)
        self.state = train_state.TrainState.create(apply_fn=self.forward, params=self.model_params, tx=self.optimizer)
    
    def cross_entropy_loss(*, logits, labels):
        labels_onehot = jax.nn.one_hot(labels, num_classes=10)
        return optax.softmax_cross_entropy(logits=logits, labels=labels_onehot).mean()
    
    def compute_metrics(*, logits, labels):
        loss = cross_entropy_loss(logits=logits, labels=labels)
        accuracy = jnp.mean(jnp.argmax(logits, -1) == labels)
        metrics = {
          'loss': loss,
          'accuracy': accuracy,
        }
        return metrics
    
    @jax.jit
    def train_step(self, state, inputs, labels):        
        """Train for a single step."""
        def loss_fn(params):
            logits = self.forward(params, inputs)
            loss = cross_entropy_loss(logits=logits, labels=labels)            
            return loss, logits
        
        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
        (_, logits), grads = grad_fn(state.params)
        state = state.apply_gradients(grads=grads)
        metrics = compute_metrics(logits=logits, labels=labels)
        return state, metrics
    
    @jax.jit
    def eval_step(self, inputs, labels):
        logits = self.forward(params, inputs)
        return compute_metrics(logits=logits, labels=labels)
    
    def train_epoch(self, train_ds, batch_size, epoch, rng, inputs_name="image", labels_name="label"):
        """Train for a single epoch."""
        train_ds_size = len(train_ds[inputs_name])
        steps_per_epoch = train_ds_size // batch_size

        perms = jax.random.permutation(rng, train_ds_size)
        perms = perms[:steps_per_epoch * batch_size]  # skip incomplete batch
        perms = perms.reshape((steps_per_epoch, batch_size))
        batch_metrics = []
        for perm in perms:
            batch = {k: v[perm, ...] for k, v in train_ds.items()}
            self.state, metrics = self.train_step(self.state, batch[inputs_name], batch[labels_name])
            batch_metrics.append(metrics)

        # compute mean of metrics across each batch in epoch.
        batch_metrics_np = jax.device_get(batch_metrics)
        epoch_metrics_np = {
          k: np.mean([metrics[k] for metrics in batch_metrics_np])
          for k in batch_metrics_np[0]}

        print('train epoch: %d, loss: %.4f, accuracy: %.2f' % (
          epoch, epoch_metrics_np['loss'], epoch_metrics_np['accuracy'] * 100))
        
    def eval_model(self, test_ds):
        metrics = self.eval_step(self.model_params, test_ds)
        metrics = jax.device_get(metrics)
        summary = jax.tree_map(lambda x: x.item(), metrics)
        return summary['loss'], summary['accuracy']
      

def test_parallel_transformer(input_shape: tuple):
    """Tests forward pass of a transformer on an arbitrary tensor.
    Args:
        input_shape: first dimension must be the batch size
    """
    print(f"Running transformer on input with shape: {input_shape}")
    vocab_size = 200
    test_config = TransformerConfig(input_vocab_size=vocab_size, output_size=vocab_size)
    
    parallel_transformer = ParallelizedTransformer(test_config, input_shape[1:])
    print("Successfully built model")
    
    rng = random.PRNGKey(42)
    input_rng, _ = random.split(rng)
    input_logits = random.uniform(input_rng, (*input_shape, vocab_size))
    inputs = random.categorical(input_rng, input_logits, axis=-1)
    
    for i in range(100):    
        y = parallel_transformer.forward(parallel_transformer.model_params, inputs)
        desired_shape = (*input_shape, vocab_size)
        assert y.shape == desired_shape, f"output shape is {y.shape} instead of {desired_shape}"
    print("All done!")
    

if __name__ == '__main__':    
    test_parallel_transformer((512, 300))
    test_parallel_transformer((512, 10, 10))
    test_parallel_transformer((128, 10, 10, 10)) 