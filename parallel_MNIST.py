"""
Parallelized version of the tensor_model transformer
"""

from doctest import OutputChecker
import functools
from typing import Any, Callable, Optional, Tuple
import copy
import dataclasses

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
os.environ["LD_LIBRARY_PATH"] = "/clusterfs/nilah/aniketh/conda_envs/deeplearning_cu113/lib64/"
# os.environ["XLA_FLAGS"] = "--xla_gpu_strict_conv_algorithm_picker=false"

PRNGKey = Any
Shape = Tuple[int]
Dtype = Any
Array = Any

class CNN(nn.Module):
    config: TransformerConfig
    """A simple CNN model."""

    @nn.compact
    def __call__(self, x):
        x = nn.Conv(features=32, kernel_size=(3, 3))(x)
        x = nn.relu(x)
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = nn.Conv(features=64, kernel_size=(3, 3))(x)
        x = nn.relu(x)
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = x.reshape((x.shape[0], -1))  # flatten
        x = nn.Dense(features=256)(x)
        x = nn.relu(x)
        x = nn.Dense(features=10)(x)
        return x

class MnistViT(nn.Module):
    config: TransformerConfig
    
    def setup(self):
        self.transformer = Transformer(self.config)
        self.dense = nn.Dense(features=10)    

    def __call__(self, inputs, deterministic=None):
        """Applies a Transformer, a mean pooling of output embddings for each pixel, followed by 
        a dense layer to predict the class of each image.
        Args:
            inputs: shape of [batch, 28, 28]
        Returns:
            logits of shape [batch, 10]
        """
        out = self.transformer(inputs, deterministic=deterministic) # shape: [batch, 28, 28, 32 /*output size */]
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
        if module_class == MnistViT:
            sample_input_shape = (sample_input_batch_size, *input_shape)
            sample_input_logits = random.uniform(self.input_rng, (*sample_input_shape, config.input_vocab_size))
            sample_inputs = random.categorical(self.input_rng, sample_input_logits, axis=-1)
        elif module_class == CNN:
            sample_input_shape = (sample_input_batch_size, *input_shape)
            sample_inputs = random.uniform(self.input_rng, sample_input_shape)
        
        # create and initialize model
        self.model = module_class(config)
        self.model_params = self.model.init({"params": self.init_rng, "dropout": self.dropout_rng}, sample_inputs, deterministic=True)
        
        # create pjit mesh to shard the model along the second input dimension (usually, the length of the sequence)
        self.mesh_shape = (1, n_gpus)
        self.devices = np.asarray(jax.devices()[:n_gpus]).reshape(*self.mesh_shape)
        self.mesh = maps.Mesh(self.devices, ('batch', 'length'))
        
        # define the train step pjit
        def train_step(state, dropout_rng, inputs, labels):
            """Train for a single step."""
            dropout_rng, new_dropout_rng = random.split(dropout_rng)            
            def loss_fn(params):
                logits = module_class(config).apply(params, inputs, deterministic=False, rngs={"dropout": dropout_rng})
                loss = cross_entropy_loss(logits=logits, labels=labels)            
                return loss, logits

            grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
            (_, logits), grads = grad_fn(state.params)
            state = state.apply_gradients(grads=grads)
            metrics = compute_metrics(logits=logits, labels=labels)
            return state, metrics, new_dropout_rng
        
        self.train_step_pjit = pjit(lambda state, dropout_rng, inputs, labels: train_step(state, dropout_rng, inputs, labels),\
                                    in_axis_resources=[None, None, PartitionSpec('batch', 'length'), PartitionSpec('batch')],
                                    out_axis_resources=None)
        
        # define the normal forward pass pjit for evaluation   
        self.eval_config = self.eval_config = dataclasses.replace(config, dropout_rate=0)
        self.forward_pjit = pjit(lambda model_params, inputs: module_class(self.eval_config).apply(model_params, inputs, deterministic=True),
                               in_axis_resources=[None, PartitionSpec('batch', 'length')],
                               out_axis_resources=PartitionSpec('batch'))

    def create_train_state(self, optimizer_module, **optimize_params):
        """Creates initial `TrainState`."""
        self.optimizer = optimizer_module(**optimize_params)
        self.state = train_state.TrainState.create(apply_fn=self.train_step_pjit, params=self.model_params, tx=self.optimizer)

    def eval_step(self, inputs, labels):
        with self.mesh as m:
            logits = self.forward_pjit(params, inputs)
        return compute_metrics(logits=logits, labels=labels)

    def train_epoch(self, train_ds, batch_size, epoch, rng, inputs_name="image", labels_name="label"):
        """Train for a single epoch."""
        train_ds_size = len(train_ds[inputs_name])
        steps_per_epoch = train_ds_size // batch_size
        
        rng, dropout_rng = random.split(rng)
        perms = jax.random.permutation(rng, train_ds_size)
        perms = perms[:steps_per_epoch * batch_size]  # skip incomplete batch
        perms = perms.reshape((steps_per_epoch, batch_size))
        batch_metrics = []
        for perm in tqdm(perms):
            batch = {k: v[perm, ...] for k, v in train_ds.items()}
            with self.mesh as m:
                self.state, metrics, dropout_rng = self.train_step_pjit(self.state, dropout_rng, batch[inputs_name], batch[labels_name])
            batch_metrics.append(metrics)

        # compute mean of metrics across each batch in epoch.
        batch_metrics_np = jax.device_get(batch_metrics)
        epoch_metrics_np = {
          k: np.mean([metrics[k] for metrics in batch_metrics_np])
          for k in batch_metrics_np[0]}

        print('train epoch: %d, loss: %.4f, accuracy: %.2f' % (
          epoch, epoch_metrics_np['loss'], epoch_metrics_np['accuracy'] * 100))

    def eval_model(self, test_ds, batch_size, epoch, inputs_name="image", labels_name="label"):
        # get all predictions for test_ds
        all_preds = []
        all_labels = []
        for i in tqdm(range(0, len(test_ds[inputs_name]), batch_size)):
            batch = {k: v[np.arange(i, min(i+batch_size, len(test_ds[inputs_name]))), ...] for k, v in test_ds.items()}
            with self.mesh as m:
                batch_preds = self.forward_pjit(self.state.params, batch[inputs_name])
                
            all_preds.append(jax.device_get(batch_preds))
            all_labels.append(batch[labels_name])
        
        # stack them
        all_preds = np.vstack(all_preds)
        all_labels = np.hstack(all_labels)
        
        # compute metrics
        epoch_metrics_np = compute_metrics(logits=all_preds, labels=all_labels)
        
        print('test epoch: %d, loss: %.4f, accuracy: %.2f' % (
          epoch, epoch_metrics_np['loss'], epoch_metrics_np['accuracy'] * 100))
        
    
def get_MNIST_datasets():
    """Load MNIST train and test datasets into memory."""
    ds_builder = tfds.builder('mnist')
    ds_builder.download_and_prepare()
    train_ds = tfds.as_numpy(ds_builder.as_dataset(split='train', batch_size=-1))
    test_ds = tfds.as_numpy(ds_builder.as_dataset(split='test', batch_size=-1))
#     train_ds['image'] = jnp.float32(train_ds['image']) / 255.
#     test_ds['image'] = jnp.float32(test_ds['image']) / 255.
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
#     model = ParallelizedModel(CNN, config, (28, 28, 1))
    
    # random seed for shuffling training dataset
    rng = jax.random.PRNGKey(0)
    rng, init_rng = jax.random.split(rng)
    
    # create initial train state
    optimizer = optax.sgd
    learning_rate = 1e-3
    momentum = 0.9
    model.create_train_state(optimizer, learning_rate = learning_rate, momentum = momentum)
    
    num_epochs = 10
    batch_size = 64
    for epoch in range(1, num_epochs + 1):
        # Use a separate PRNG key to permute image data during shuffling
        rng, input_rng = jax.random.split(rng)
        # Run an optimization step over a training batch
        model.train_epoch(train_ds, batch_size, epoch, input_rng)
        # Evaluate on the test set after each training epoch
        model.eval_model(test_ds, batch_size, epoch)        

if __name__ == '__main__':
    MNIST_test()