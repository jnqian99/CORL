import os
os.environ["XLA_FLAGS"] = '--xla_force_host_platform_device_count=8'

from typing import Optional, Any
import shutil

import numpy as np
import jax
from jax import random, numpy as jnp

import flax
from flax import linen as nn
from flax.training import checkpoints, train_state
from flax import struct, serialization
import orbax.checkpoint

import optax

ckpt_dir = '/tmp/flax_ckpt'

if os.path.exists(ckpt_dir):
    shutil.rmtree(ckpt_dir)  # Remove any existing checkpoints from the last notebook run.

# A simple model with one linear layer.
key = random.PRNGKey(0)
key1, key2 = random.split(key)
x1 = random.normal(key1, (5,))      # A simple JAX array.
model = nn.Dense(features=3)
variables = model.init(key2, x1)

# Flax's TrainState is a pytree dataclass and is supported in checkpointing.
# Define your class with `@flax.struct.dataclass` decorator to make it compatible.
tx = optax.sgd(learning_rate=0.001)      # An Optax SGD optimizer.
state = train_state.TrainState.create(
    apply_fn=model.apply,
    params=variables['params'],
    tx=tx)
# Perform a simple gradient update similar to the one during a normal training workflow.
state = state.apply_gradients(grads=jax.tree_util.tree_map(jnp.ones_like, state.params))

# Some arbitrary nested pytree with a dictionary and a NumPy array.
config = {'dimensions': np.array([5, 3])}

# Bundle everything together.
ckpt = {'model': state, 'config': config, 'data': [x1]}

print(ckpt)

shutil.rmtree('/tmp/chkpt/myenv_gru/halfcheetah/medium_v2')
orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
orbax_checkpointer.save('/tmp/chkpt/myenv_gru/halfcheetah/medium_v2', ckpt)


