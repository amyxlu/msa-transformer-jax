import jax
from jax import random, numpy as jnp

from flax import linen as nn
from flax.training import train_state

import numpy as np
import optax
import tensorflow_datasets as tfds

import os
from tqdm import tqdm

import sys

sys.path.append("..")
import tensor_model as tm
from msa_transformer import MSATransformer
from configs import MSATransformerConfig

NUM_EPOCHS = 10
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
MOMENTUM = 0.9

cfg = MSATransformerConfig()

def get_datasets():
    """Load MNIST train and test datasets into memory."""
    ds_builder = tfds.builder('mnist')
    ds_builder.download_and_prepare()
    train_ds = tfds.as_numpy(ds_builder.as_dataset(split='train', batch_size=-1))
    test_ds = tfds.as_numpy(ds_builder.as_dataset(split='test', batch_size=-1))
    return train_ds, test_ds


def cross_entropy_loss(logits, labels):
    labels_onehot = jax.nn.one_hot(labels, num_classes=10)
    return optax.softmax_cross_entropy(logits=logits, labels=labels_onehot).mean()


def compute_metrics(logits, labels):
    loss = cross_entropy_loss(logits, labels)
    accuracy = jnp.mean(jnp.argmax(logits, axis=-1) == labels)
    return {'loss': loss, 'accuracy': accuracy}


@jax.jit
def train_step(state, batch):
    def loss_fn(params):
        model = MSATransformer(cfg)
        rng = random.PRNGKey(42)  # TODO: remove harcoded rng
        logits = model.apply({'params': params}, batch['image'], rngs={'dropout': rng})
        loss = cross_entropy_loss(logits, batch['label'])
        return loss, logits

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    logits, grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    metrics = compute_metrics(logits, batch['label'])
    return state, metrics


def print_metrics(batch_metrics, epoch, train=True):
    batch_metrics = jax.device_get(batch_metrics)
    # compute mean of metrics across each batch in epoch
    epoch_metrics = {
        k: np.mean([metrics[k] for metrics in batch_metrics])
        for k in batch_metrics[0]
    }

    print("{} epoch: {}, loss: {:.3f}, accuracy: {:.3f}".format(
        'train' if train else 'test',
        epoch,
        epoch_metrics['loss'],
        epoch_metrics['accuracy']
    ))


def train_epoch(state, train_ds, epoch, rng):
    """Train for a single epoch."""
    train_ds_size = len(train_ds['image'])
    steps_per_epoch = train_ds_size // BATCH_SIZE

    permutation = random.permutation(rng, train_ds_size)
    permutation = permutation[: steps_per_epoch * BATCH_SIZE]  # skip incomplete batch
    all_batch_indexes = permutation.reshape((steps_per_epoch, BATCH_SIZE))

    batch_metrics = []
    for batch_indexes in tqdm(all_batch_indexes):
        batch_data = {k: v[batch_indexes, ...] for k, v in train_ds.items()}
        state, metrics = train_step(state, batch_data)
        batch_metrics.append(metrics)

    print_metrics(batch_metrics, epoch, train=True)
    return state


@jax.jit
def eval_step(params, batch):
    model = MSATransformer(cfg)
    rng = random.PRNGKey(42)  # TODO: remove harcoded rng
    logits = model.apply({'params': params}, batch['image'], rngs={'dropout': rng})
    return compute_metrics(logits, batch['label'])


def eval_model(params, test_ds, epoch):
    batch_metrics = []
    for i in range(len(test_ds['image'])):
        batch_data = {k: v[[i], ...] for k, v in test_ds.items()}
        metrics = eval_step(params, batch_data)
        batch_metrics.append(metrics)

    print_metrics(batch_metrics, epoch, train=False)


if __name__ == "__main__":
    train_ds, test_ds = get_datasets()
    cfg = MSATransformerConfig()
    model = MSATransformer(cfg, "mnist")

    # Initialize model
    rng = random.PRNGKey(0)
    rng, init_rng = random.split(rng)
    rng, dropout_rng = random.split(rng)

    # dummy_input_shape = (BATCH_SIZE, 28, 28)
    dummy_input_shape = (4, 28, 28)
    dummy_input_logits = random.uniform(init_rng, (*dummy_input_shape, 255))
    dummy_input = random.categorical(init_rng, dummy_input_logits, axis=-1)
    params = model.init({'params': rng, 'dropout': dropout_rng}, dummy_input)#['params']
    print(jax.tree_map(lambda x: print(x.shape), params))

    # Create optimizer and state
    tx = optax.sgd(LEARNING_RATE, MOMENTUM)
    state = train_state.TrainState.create(apply_fn=model.apply, params=params['params'], tx=tx)

    # Train model and take snapshot on test data every epoch
    for epoch in tqdm(range(1, NUM_EPOCHS + 1)):
        # Use a separate PRNG key to permute image data during shuffle
        rng, input_rng = random.split(rng)

        # Run an optimization step over a training batch
        state = train_epoch(state, train_ds, epoch, input_rng)

        # Evaluate on the test set after each training epoch
        eval_model(state.params, test_ds, epoch)

