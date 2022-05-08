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
from flax.training import train_state

import tensorflow_datasets as tfds

import numpy as np
import os
import pdb
from tqdm import tqdm

from tensor_model import *

# os.environ["NCCL_LAUNCH_MODE"] = "PARALLEL"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

PRNGKey = Any
Shape = Tuple[int]
Dtype = Any
Array = Any

class MnistViT(nn.Module):
    config: TransformerConfig
    
    def setup(self):
        self.transformer = Transformer(self.config)
        self.dense = nn.Dense(features=10)    

    def __call__(self, inputs):
        """Applies a Transformer, a mean pooling of output embddings for each pixel, followed by 
        a dense layer to predict the class of each image.
        Args:
            inputs: shape of [batch, 28, 28]
        Returns:
            logits of shape [batch, 10]
        """
        out = self.transformer(inputs) # shape: [batch, 28, 28, 32 /*output size */]
        out = jnp.mean(out, axis=(1, 2)).squeeze()
        out = self.dense(out)
        return out
    
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

class ParallelizedModel():
    def __init__(self, module_class, config, input_shape, n_gpus=2, seed=97, sample_input_batch_size=2):
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
        self.rng = random.PRNGKey(seed)
        self.init_rng, self.dropout_rng, self.input_rng = random.split(self.rng, num=3)
        
        # create dummy sample one-hot inputs used to initialize the model
        sample_input_shape = (sample_input_batch_size, *input_shape)
        sample_input_logits = random.uniform(self.input_rng, (*sample_input_shape, config.input_vocab_size))
        sample_inputs = random.categorical(self.input_rng, sample_input_logits, axis=-1)
        
        # create and initialize model
        self.model = module_class(config)
        self.model_params = self.model.init({"params": self.init_rng, "dropout": self.dropout_rng}, sample_inputs)
        
        # create pjit mesh to shard the model along the second input dimension (usually, the length of the sequence)
        self.mesh_shape = (1, n_gpus)
        self.devices = np.asarray(jax.devices()[:n_gpus]).reshape(*self.mesh_shape)
        self.mesh = maps.Mesh(self.devices, ('batch', 'length'))
        
        self.model_pjit = pjit(lambda model_params, inputs: self.model.apply(model_params, inputs, 
                                                                             rngs={"dropout": self.dropout_rng}),
                               in_axis_resources=[None, PartitionSpec('batch', 'length')],
                               out_axis_resources=PartitionSpec('batch'))
    
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
        for perm in tqdm(perms):
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
        
    def eval_model(self, test_ds, epoch, inputs_name="image", labels_name="label"):
        batch_metrics = []
        for i in range(len(test_ds[inputs_name])):
            batch_data = {k: v[[i], ...] for k, v in test_ds.items()}
            metrics = eval_step(params, batch_data)
            batch_metrics.append(metrics)

        print_metrics(batch_metrics, epoch, train=False)
        
    def eval_model(self, test_ds):
        metrics = self.eval_step(self.model_params, test_ds)
        metrics = jax.device_get(metrics)
        summary = jax.tree_map(lambda x: x.item(), metrics)
        return summary['loss'], summary['accuracy']
    
      
def get_MNIST_datasets():
    """Load MNIST train and test datasets into memory."""
    ds_builder = tfds.builder('mnist')
    ds_builder.download_and_prepare()
    train_ds = tfds.as_numpy(ds_builder.as_dataset(split='train', batch_size=-1))
    test_ds = tfds.as_numpy(ds_builder.as_dataset(split='test', batch_size=-1))
    return train_ds, test_ds
    
def MNIST_test():
    # get datasets
    train_ds, test_ds = get_MNIST_datasets()
    
    # create model
    config = TransformerConfig(
            input_vocab_size=256, 
            output_size=32,
            emb_dim=32,
            d_qkv=32,
            d_mlp = 64,
            n_layers=4,
            n_heads=4,
            dropout_rate=0.1
        )
    model = ParallelizedModel(MnistViT, config, (28, 28))
    
    # random seed for shuffling training dataset
    rng = jax.random.PRNGKey(0)
    rng, init_rng = jax.random.split(rng)
    
    # create initial train state
    optimizer = optax.sgd
    learning_rate = 0.1
    momentum = 0.9
    model.create_train_state(optimizer, learning_rate = learning_rate, momentum = momentum)
    
    num_epochs = 10
    batch_size = 128
    for epoch in range(1, num_epochs + 1):
        # Use a separate PRNG key to permute image data during shuffling
        rng, input_rng = jax.random.split(rng)
        # Run an optimization step over a training batch
        model.train_epoch(train_ds, batch_size, epoch, input_rng)
        # Evaluate on the test set after each training epoch
        test_loss, test_accuracy = model.eval_model(test_ds)
        print(' test epoch: %d, loss: %.2f, accuracy: %.2f' % (
          epoch, test_loss, test_accuracy * 100))
        

if __name__ == '__main__':
    MNIST_test()