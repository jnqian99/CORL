import jax
from jax import numpy as jnp
from flax import linen as nn

class GRU(nn.Module):
    hidden_size: int

    @nn.compact
    def __call__(self, inputs, state):
        update_gate = nn.sigmoid(nn.Dense(self.hidden_size, name='update_gate')(jnp.concatenate([inputs, state], axis=-1)))
        reset_gate = nn.sigmoid(nn.Dense(self.hidden_size, name='reset_gate')(jnp.concatenate([inputs, state], axis=-1)))
        candidate_state = jnp.tanh(nn.Dense(self.hidden_size, name='candidate_state')(jnp.concatenate([inputs, reset_gate * state], axis=-1)))
        new_state = state * (1 - update_gate) + candidate_state * update_gate
        return new_state

    @staticmethod
    def initialize_params(rng, input_shape, input_dtype=jnp.float32):
        input_size = input_shape[-1]
        output_size = input_shape[-1]  # output size same as input size for GRU
        rng, dropout_rng = jax.random.split(rng)
        dummy_input = jnp.ones(input_shape, input_dtype)
        initial_variables = GRU.partial(hidden_size=output_size).init({'params': rng}, dummy_input, state=None)
        return initial_variables['params']

# Example usage
input_shape = (32, 10)  # (batch_size, sequence_length)
gru = GRU(hidden_size=64)
params = GRU.initialize_params(jax.random.PRNGKey(0), input_shape)

# Dummy input and initial state
inputs = jnp.ones((32, 10, input_shape[-1]))
initial_state = jnp.zeros((32, gru.hidden_size))

# Applying GRU
new_state = gru.apply({'params': params}, inputs, state=initial_state)
