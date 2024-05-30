import os

#os.environ["XLA_FLAGS"] = '--xla_force_host_platform_device_count=8'
os.environ["TF_CUDNN_DETERMINISTIC"] = "1"  # For reproducibility

from typing import TypeVar, Generic, Tuple, Union, Optional, SupportsFloat

import math
import uuid
from copy import deepcopy
from dataclasses import asdict, dataclass
from functools import partial
from typing import Any, Callable, Dict, Sequence, Tuple, Union, List

import chex
import d4rl  # noqa
import flax.linen as nn
import gym
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pyrallis
import wandb
from flax.core import FrozenDict
from flax.training.train_state import TrainState
from tqdm.auto import trange
import random
import pickle
# use orbax 0.1.0
from flax.training import checkpoints
import shutil
import dataclasses

ObsType = TypeVar("ObsType")
ActType = TypeVar("ActType")

@dataclass
class Config:
    # wandb params
    project: str = "myenv"
    group: str = "myenv_gru"
    name: str = "myenv_gru"
    # model params
    hidden_dim: int = 256
    rewards_ln: bool = True
    dones_cutoff: float = 0.6
    dones_ln: bool = True
    rewards_n_hiddens: int = 3
    rewards_learning_rate: float = 1e-3
    normalize_reward: bool = False
    normalize_states: bool = False
    dones_learning_rate: float = 1e-3
    dones_n_hiddens: int = 3    
    gamma: float = 0.99
    policy_noise: float = 0.2
    noise_clip: float = 0.5
    dynamics_learning_rate: float = 1e-3
    tau: float = 0.005
    # training params
    dataset_name: str = "halfcheetah-medium-v2"
    num_epochs: int = 100
    num_updates_on_epoch: int = 100
    # evaluation params
    eval_episodes: int = 5
    eval_every: int = 20
    eval_steps: int = 50
    # general params
    train_seed: int = 0
    eval_seed: int = 42
    # whether to load the model again for further training
    reload_chkpt: bool = False
    time_steps: int = 5
    max_steps: int = 2048
    rb_stage_file: str = "halfcheetah-medium-v2"
    chkpt_dir: str = "/tmp/flax_ckpt"

    def __post_init__(self):
        self.name = f"{self.name}-{self.dataset_name}-{str(uuid.uuid4())[:8]}"

    def dict(self):
        return {k: v for k, v in asdict(self).items() if not isinstance(v,str) }

class MyEnv(Generic[ObsType, ActType]):
    states = []
    actions = []
    rewards = []
    dones = []

    dynamics_params : jax.Array
    dynamics_fn : Callable
    rewards_params : jax.Array
    rewards_fn : Callable
    dones_params : jax.Array
    dones_fn : Callable

    time_step : int = 5
    init_state : ObsType
    init_action : ActType

    def step(self, action: ActType,
             dones_cutoff: float = 0.6) -> Tuple[ObsType, float, bool, float]:
        """
        Run one timestep of the environment's dynamics. When end of
        episode is reached, you are responsible for calling `reset()`
        to reset this environment's state.

        Accepts an action and returns a tuple (observation, reward, done).

        Args:
            action (object): an action provided by the agent

        Returns:
            observation (object): agent's observation of the current environment
            reward (float) : amount of reward returned after previous action
            done (bool): whether the episode has ended, in which case further step() calls will return undefined results
        """
        self.actions.append(action)
        steps = min(len(self.states), self.time_step)
        states_ = jnp.stack([self.states[i] for i in range(len(self.states)-steps, len(self.states))], axis=0)
        actions_ = jnp.stack([self.actions[i] for i in range(len(self.actions)-steps, len(self.actions))], axis=0)
        states_ = states_[None, ...]
        actions_ = actions_[None, ...]
        next_state = self.dynamics_fn(self.dynamics_params, states_, actions_)
        state_1 = self.states[-1]
        action_1 = self.actions[-1]
        state_1 = state_1[None, ...]
        action_1 = action_1[None, ...]
        reward = self.rewards_fn(self.rewards_params, state_1, action_1, next_state)
        done = self.dones_fn(self.dones_params, state_1, action_1, next_state)
        next_state = next_state.squeeze(0)
        reward = reward.squeeze(0)
        done = done.squeeze(0)
        self.states.append(next_state)
        self.rewards.append(reward)
        self.dones.append(done)
        return next_state, reward, done > dones_cutoff, done
    
    def reset(
        self,
        obs: ObsType,
    ) -> ObsType:
        """Resets the environment to an initial state and returns an initial
        observation.

        Returns:
            observation (object): the initial observation.
        """
        self.states = [obs]
        self.actions = []
        self.rewards = []
        self.dones = []
        return obs
    
    def load(self, dynamics_params: jax.Array, rewards_params: jax.Array, dones_params: jax.Array,
             dynamics_fn: Callable, rewards_fn: Callable, dones_fn: Callable, time_steps: int) -> None:
             
        """Load the environment from a file.
           model can only be load once in its lifetime.

        Args:
            path (str): the path to the file.
        """
        self.dynamics_params = dynamics_params
        self.rewards_params = rewards_params
        self.dones_params = dones_params
        self.dynamics_fn = dynamics_fn
        self.rewards_fn = rewards_fn
        self.dones_fn = dones_fn
        self.time_step = time_steps

    def load_from_checkpoint(self, chk_path: str) -> None:
        """Load the environment from a checkpoint.

        Args:
            path (str): the path to the checkpoint
        """
        key = jax.random.PRNGKey(0)
        chkdata = checkpoints.restore_checkpoint(ckpt_dir=chk_path,
                            target=None,
                            step=0)
        conf_dict = chkdata["config"]
        init_state = chkdata["init_state"]
        init_action = chkdata["init_action"]

        dynamics_module_ = Dynamics(
            state_dim=init_state.shape[-1],
            action_dim=init_action.shape[-1],
            hidden_dim=conf_dict["hidden_dim"],
        )
        dynamics_ = DynamicsTrainState.create(
            apply_fn=dynamics_module_.apply,
            params=dynamics_module_.init(key, init_state, init_action),
            target_params=dynamics_module_.init(key, init_state, init_action),
            tx=optax.adam(learning_rate=conf_dict["dynamics_learning_rate"]),
        )
        rewards_module_ = Rewards(
            state_dim=init_state.shape[-1],
            action_dim=init_action.shape[-1],
            hidden_dim=conf_dict["hidden_dim"],
            layernorm=conf_dict["rewards_ln"],
            n_hiddens=conf_dict["rewards_n_hiddens"],
        )
        rewards_ = RewardsTrainState.create(
            apply_fn=rewards_module_.apply,
            params=rewards_module_.init(key, init_state[:,-1,:], init_action[:,-1,:], init_state[:,-1,:]),
            target_params=rewards_module_.init(key, init_state[:,-1,:], init_action[:,-1,:], init_state[:,-1,:]),
            tx=optax.adam(learning_rate=conf_dict["rewards_learning_rate"]),
        )
        dones_module_ = Dones(
            state_dim=init_state.shape[-1],
            action_dim=init_action.shape[-1],
            hidden_dim=conf_dict["hidden_dim"],
            layernorm=conf_dict["dones_ln"],
            n_hiddens=conf_dict["dones_n_hiddens"],
        )
        dones_ = DonesTrainState.create(
            apply_fn=dones_module_.apply,
            params=dones_module_.init(key, init_state[:,-1,:], init_action[:,-1,:], init_state[:,-1,:]),
            target_params=dones_module_.init(key, init_state[:,-1,:], init_action[:,-1,:], init_state[:,-1,:]),
            tx=optax.adam(learning_rate=conf_dict["dones_learning_rate"]),
        )
        target = {
            "dynamics": dynamics_,
            "rewards": rewards_,
            "dones": dones_,
            "config": conf_dict,
            "init_state": init_state,
            "init_action": init_action,
        }
        chkdata = checkpoints.restore_checkpoint(ckpt_dir=chk_path,
                            target=target,
                            step=0)
        dynamics = chkdata["dynamics"]
        rewards = chkdata["rewards"]
        dones = chkdata["dones"]
        conf_dict = chkdata["config"]
        init_state = chkdata["init_state"]
        init_action = chkdata["init_action"]
        self.load(dynamics_params=dynamics.params, rewards_params=rewards.params, dones_params=dones.params,
                    dynamics_fn=dynamics.apply_fn, rewards_fn=rewards.apply_fn, dones_fn=dones.apply_fn, 
                    time_steps=conf_dict["time_steps"])
        self.init_state = init_state
        self.init_action = init_action

    def get_state_size(self) -> int:
        return self.init_state.shape[-1]
    
    def get_action_size(self) -> int:
        return self.init_action.shape[-1]


def pytorch_init(fan_in: float) -> Callable:
    """
    Default init for PyTorch Linear layer weights and biases:
    https://pytorch.org/docs/stable/generated/torch.nn.Linear.html
    """
    bound = math.sqrt(1 / fan_in)

    def _init(key: jax.random.PRNGKey, shape: Tuple, dtype: type) -> jax.Array:
        return jax.random.uniform(
            key, shape=shape, minval=-bound, maxval=bound, dtype=dtype
        )

    return _init

def uniform_init(bound: float) -> Callable:
    def _init(key: jax.random.PRNGKey, shape: Tuple, dtype: type) -> jax.Array:
        return jax.random.uniform(
            key, shape=shape, minval=-bound, maxval=bound, dtype=dtype
        )

    return _init

def identity(x: Any) -> Any:
    return x

class Dynamics(nn.Module):
    action_dim: int
    state_dim: int
    hidden_dim: int = 256

    @nn.compact
    def __call__(self, state: jax.Array, action: jax.Array) -> jax.Array:
        s_d, a_d, h_d = self.state_dim, self.action_dim, self.hidden_dim
        h = jnp.zeros((state.shape[0], self.hidden_dim))
        for timestep in range(state.shape[1]):
            inputs = jnp.concatenate([state[:, timestep, :], action[:, timestep, :]], axis=-1)
            update_gate = nn.sigmoid(nn.Dense(self.hidden_dim, name=f'update_gate{timestep+1}',
                                            kernel_init=pytorch_init(s_d+a_d+h_d),
                                            #kernel_init=nn.initializers.constant(0.0),
                                            bias_init=nn.initializers.constant(0.1)
                                            )
                                            (jnp.concatenate([inputs, h], axis=-1)))
            reset_gate = nn.sigmoid(nn.Dense(self.hidden_dim, name=f'reset_gate{timestep+1}',
                                            kernel_init=pytorch_init(s_d+a_d+h_d),
                                            bias_init=nn.initializers.constant(0.1)
                                            )
                                            (jnp.concatenate([inputs, h], axis=-1)))
            candidate_state = jnp.tanh(nn.Dense(self.hidden_dim, name=f'candidate_state{timestep+1}',
                                            kernel_init=pytorch_init(s_d+a_d+h_d),
                                            bias_init=nn.initializers.constant(0.1)
                                            )
                                            (jnp.concatenate([inputs, reset_gate * h], axis=-1)))
            h = h * (1 - update_gate) + candidate_state * update_gate
        out = nn.Dense(self.state_dim,
                       kernel_init=uniform_init(3e-3), bias_init=uniform_init(3e-3)
                       )(h)
        #next_state = jnp.tanh(out)
        return out
    
class Rewards(nn.Module):
    state_dim: int
    action_dim: int
    hidden_dim: int = 256
    layernorm: bool = True
    n_hiddens: int = 3

    @nn.compact
    def __call__(self, state: jax.Array, action: jax.Array, next_state: jax.Array) -> jax.Array:
        s_d, a_d, ns_d, h_d = self.state_dim, self.action_dim, self.state_dim, self.hidden_dim
        # Initialization as in the EDAC paper
        # print("state.shape", state.shape, "action.shape", action.shape)
        layers = [
            nn.Dense(
                self.hidden_dim,
                kernel_init=pytorch_init(s_d + a_d + ns_d),
                bias_init=nn.initializers.constant(0.1),
            ),
            nn.relu,
            nn.LayerNorm() if self.layernorm else identity,
        ]
        for _ in range(self.n_hiddens - 1):
            layers += [
                nn.Dense(
                    self.hidden_dim,
                    kernel_init=pytorch_init(h_d),
                    bias_init=nn.initializers.constant(0.1),
                ),
                nn.relu,
                nn.LayerNorm() if self.layernorm else identity,
            ]
        layers += [
            nn.Dense(1, kernel_init=uniform_init(3e-3), bias_init=uniform_init(3e-3)),
        ]
        network = nn.Sequential(layers)
        state_action = jnp.concatenate([state, action, next_state], axis=-1)
        out = network(state_action)
        rewards = out.squeeze(-1)
        return rewards

class Dones(nn.Module):
    state_dim: int
    action_dim: int
    hidden_dim: int = 256
    layernorm: bool = True
    n_hiddens: int = 3

    @nn.compact
    def __call__(self, state: jax.Array, action: jax.Array, next_state: jax.Array) -> jax.Array:
        s_d, a_d, ns_d, h_d = self.state_dim, self.action_dim, self.state_dim, self.hidden_dim
        # Initialization as in the EDAC paper
        # print("state.shape", state.shape, "action.shape", action.shape)
        layers = [
            nn.Dense(
                self.hidden_dim,
                kernel_init=pytorch_init(s_d + a_d + ns_d),
                bias_init=nn.initializers.constant(0.1),
            ),
            nn.relu,
            nn.LayerNorm() if self.layernorm else identity,
        ]
        for _ in range(self.n_hiddens - 1):
            layers += [
                nn.Dense(
                    self.hidden_dim,
                    kernel_init=pytorch_init(h_d),
                    bias_init=nn.initializers.constant(0.1),
                ),
                nn.relu,
                nn.LayerNorm() if self.layernorm else identity,
            ]
        layers += [
            nn.Dense(1, kernel_init=uniform_init(3e-3), bias_init=uniform_init(3e-3)),
            nn.sigmoid
        ]
        network = nn.Sequential(layers)
        state_action = jnp.concatenate([state, action, next_state], axis=-1)
        out = network(state_action)
        dones = out.squeeze(-1)
        return dones
        
def qlearning_dataset(
    env: gym.Env,
    dataset: Dict = None,
    terminate_on_end: bool = False,
    **kwargs,
) -> Dict:
    if dataset is None:
        dataset = env.get_dataset(**kwargs)

    N = dataset["rewards"].shape[0]
    obs_ = []
    next_obs_ = []
    action_ = []
    next_action_ = []
    reward_ = []
    done_ = []
    episode_ = []

    # The newer version of the dataset adds an explicit
    # timeouts field. Keep old method for backwards compatability.
    use_timeouts = "timeouts" in dataset

    episode_step = 0
    for i in range(N - 1):
        obs = dataset["observations"][i].astype(np.float32)
        new_obs = dataset["observations"][i + 1].astype(np.float32)
        action = dataset["actions"][i].astype(np.float32)
        new_action = dataset["actions"][i + 1].astype(np.float32)
        reward = dataset["rewards"][i].astype(np.float32)
        done_bool = bool(dataset["terminals"][i])

        if use_timeouts:
            final_timestep = dataset["timeouts"][i]
        else:
            final_timestep = episode_step == env._max_episode_steps - 1
        if (not terminate_on_end) and final_timestep:
            # Skip this transition
            episode_step = 0
            continue
        if done_bool or final_timestep:
            episode_step = 0

        #print("obs", obs, "action", action, "reward", reward)
        obs_.append(obs)
        next_obs_.append(new_obs)
        action_.append(action)
        next_action_.append(new_action)
        reward_.append(reward)
        done_.append(done_bool)
        episode_.append(episode_step)        
        episode_step += 1

    start_ = []
    stop_ = []

    start = 0
    for i in range(len(episode_)):
        #print("i", i, "episode_[i]", episode_[i])
        if episode_[i] == 0:
            start = i
        start_.append(start)
    stop = len(episode_)
    for i in range(len(episode_)-1, -1, -1):
        if episode_[i] == 0:
            stop = i
        stop_.append(stop)
    stop_.reverse()

    return {
        "states": np.array(obs_),
        "actions": np.array(action_),
        "next_states": np.array(next_obs_),
        "next_actions": np.array(next_action_),
        "rewards": np.array(reward_),
        "dones": np.array(done_),
        "episode": np.array(episode_),
        "start": np.array(start_),
        "stop": np.array(stop_),
    }

def compute_mean_std(states: jax.Array, eps: float) -> Tuple[jax.Array, jax.Array]:
    mean = states.mean(0)
    std = states.std(0) + eps
    return mean, std

def normalize_states(states: jax.Array, mean: jax.Array, std: jax.Array) -> jax.Array:
    return (states - mean) / std

@partial(jax.jit, static_argnames=("time_steps",))
def sample_buffer_get(data: Dict[str, jax.Array], indices: jax.Array, time_steps: int) -> Dict[str, jax.Array]:
    batch = {}
    for key in data.keys():
        batch[key] = jnp.stack([data[key][indices+i] for i in range(time_steps)], axis=1)
    return batch

@chex.dataclass
class ReplayBuffer:
    data: Dict[str, jax.Array] = None
    mean: float = 0
    std: float = 1
    d4rl_data: Dict = None

    def create_from_d4rl(
        self,
        dataset_name: str,
        normalize_reward: bool = False,
        is_normalize: bool = False,
    ):
        d4rl_data = qlearning_dataset(gym.make(dataset_name))
        self.d4rl_data = d4rl_data
        buffer = {
            "states": jnp.asarray(d4rl_data["states"], dtype=jnp.float32),
            "actions": jnp.asarray(d4rl_data["actions"], dtype=jnp.float32),
            "rewards": jnp.asarray(d4rl_data["rewards"], dtype=jnp.float32),
            "next_states": jnp.asarray(
                d4rl_data["next_states"], dtype=jnp.float32
            ),
            "next_actions": jnp.asarray(d4rl_data["next_actions"], dtype=jnp.float32),
            "dones": jnp.asarray(d4rl_data["dones"], dtype=jnp.float32),
        }
        if is_normalize:
            self.mean, self.std = compute_mean_std(buffer["states"], eps=1e-3)
            buffer["states"] = normalize_states(buffer["states"], self.mean, self.std)
            buffer["next_states"] = normalize_states(
                buffer["next_states"], self.mean, self.std
            )
        if normalize_reward:
            buffer["rewards"] = ReplayBuffer.normalize_reward(
                dataset_name, buffer["rewards"]
            )
        self.data = buffer

    @property
    def size(self) -> int:
        # WARN: It will use len of the dataclass, i.e. number of fields.
        return self.data["states"].shape[0]
    
    '''
    def sample_batch(
        self, key: jax.random.PRNGKey, batch_size: int
    ) -> Dict[str, jax.Array]:
        indices = jax.random.randint(
            key, shape=(batch_size,), minval=0, maxval=self.size
        )
        batch = jax.tree_map(lambda arr: arr[indices], self.data)
        return batch
    '''

    def sample_batch(
        self, key: jax.random.PRNGKey, time_steps: int
    ) -> Dict[str, jax.Array]:
        
        index = random.randint(0, self.size)
        start = self.d4rl_data["start"][index]
        stop = self.d4rl_data["stop"][index]
        #print("start", start, "stop", stop)
        indices = jnp.arange(start, stop-time_steps+1)
        samples = sample_buffer_get(self.data, indices, time_steps)
        #print("samples", samples["states"].shape)
        return samples
    
    def sample_batch_1(
            self, key: jax.random.PRNGKey
        ) -> Dict[str, jax.Array]:
        index = random.randint(0, self.size)
        start = self.d4rl_data["start"][index]
        stop = self.d4rl_data["stop"][index]
        indices = jnp.arange(start, stop)
        sample = jax.tree_map(lambda arr: arr[indices], self.data)
        return sample

    def get_moments(self, modality: str) -> Tuple[jax.Array, jax.Array]:
        mean = self.data[modality].mean(0)
        std = self.data[modality].std(0)
        return mean, std

    @staticmethod
    def normalize_reward(dataset_name: str, rewards: jax.Array) -> jax.Array:
        if "antmaze" in dataset_name:
            return rewards * 100.0  # like in LAPO
        else:
            raise NotImplementedError(
                "Reward normalization is implemented only for AntMaze yet!"
            )

@chex.dataclass(frozen=True)
class Metrics:
    accumulators: Dict[str, Tuple[jax.Array, jax.Array]]

    @staticmethod
    def create(metrics: Sequence[str]) -> "Metrics":
        init_metrics = {key: (jnp.array([0.0]), jnp.array([0.0])) for key in metrics}
        return Metrics(accumulators=init_metrics)

    def update(self, updates: Dict[str, jax.Array]) -> "Metrics":
        new_accumulators = deepcopy(self.accumulators)
        for key, value in updates.items():
            acc, steps = new_accumulators[key]
            new_accumulators[key] = (acc + value, steps + 1)
            #new_accumulators[key] = (value, steps + 1)

        return self.replace(accumulators=new_accumulators)

    def compute(self) -> Dict[str, np.ndarray]:
        # cumulative_value / total_steps
        return {k: np.array(v[0] / v[1]) for k, v in self.accumulators.items()}
        #return {k: np.array(v[0]) for k, v in self.accumulators.items()}


def normalize(
    arr: jax.Array, mean: jax.Array, std: jax.Array, eps: float = 1e-8
) -> jax.Array:
    return (arr - mean) / (std + eps)


def make_env(env_name: str, seed: int) -> gym.Env:
    env = gym.make(env_name)
    env.seed(seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)
    return env


def wrap_env(
    env: gym.Env,
    state_mean: Union[np.ndarray, float] = 0.0,
    state_std: Union[np.ndarray, float] = 1.0,
    reward_scale: float = 1.0,
) -> gym.Env:
    # PEP 8: E731 do not assign a lambda expression, use a def
    def normalize_state(state: np.ndarray) -> np.ndarray:
        return (
            state - state_mean
        ) / state_std  # epsilon should be already added in std.

    def scale_reward(reward: float) -> float:
        # Please be careful, here reward is multiplied by scale!
        return reward_scale * reward

    env = gym.wrappers.TransformObservation(env, normalize_state)
    if reward_scale != 1.0:
        env = gym.wrappers.TransformReward(env, scale_reward)
    return env

def evaluate_offline(
    env: MyEnv,
    rb : ReplayBuffer,
    num_episodes: int,
    eval_steps: int,
    key: jax.random.PRNGKey,
):
    states_loss = []
    rewards_loss = []
    dones_loss = []
    for _ in trange(num_episodes, desc="Eval", leave=False):
        key, eval_key = jax.random.split(key)
        batch = rb.sample_batch_1(eval_key)
        obs = batch["states"][0,:]
        env.reset(obs)
        for i in range(max(eval_steps, batch["actions"].shape[0])):
            action = batch["actions"][i,:]
            obs, reward, bdone, done = env.step(action)
            if i > env.time_step:
                states_loss.append(((obs - batch["next_states"][i,:]) ** 2).mean())
                rewards_loss.append(((reward - batch["rewards"][i]) ** 2).mean())
                dones_loss.append(((done - batch["dones"][i]) ** 2).mean())

    return np.array(states_loss), np.array(rewards_loss), np.array(dones_loss), key


class DynamicsTrainState(TrainState):
    target_params: FrozenDict

class RewardsTrainState(TrainState):
    target_params: FrozenDict

class DonesTrainState(TrainState):
    target_params: FrozenDict    

def update_dynamics(
    key: jax.random.PRNGKey,
    dynamics: TrainState,
    batch: Dict[str, jax.Array],
    tau: float,
    metrics: Metrics,
) -> Tuple[jax.random.PRNGKey, TrainState, TrainState, Metrics]:
    key, random_state_key = jax.random.split(key, 2)

    def dynamics_loss_fn(params: jax.Array) -> Tuple[jax.Array, Metrics]:
        next_states = dynamics.apply_fn(params, batch["states"], batch["actions"])

        penalty = ((next_states - batch["next_states"][:,-1,:]) ** 2).sum(-1)

        loss = penalty.mean()

        new_metrics = metrics.update(
            {
                "dynamics_loss": loss,
            }
        )
        return loss, new_metrics

    #print("update_dynamics dynamics.params", dynamics.params["params"]["update_gate1"]["kernel"])
    grads, new_metrics = jax.grad(dynamics_loss_fn, has_aux=True)(dynamics.params)
    new_dynamics = dynamics.apply_gradients(grads=grads)
    '''
    new_dynamics = new_dynamics.replace(
        target_params=optax.incremental_update(dynamics.params, dynamics.target_params, tau)
    )
    '''
    return key, new_dynamics, new_metrics

def update_rewards(
    key: jax.random.PRNGKey,
    rewards: TrainState,
    batch: Dict[str, jax.Array],
    tau: float,
    metrics: Metrics,
) -> Tuple[jax.random.PRNGKey, TrainState, Metrics]:
    key, random_state_key = jax.random.split(key, 2)

    def rewards_loss_fn(params: jax.Array) -> Tuple[jax.Array, Metrics]:
        rewards_pred = rewards.apply_fn(params, batch["states"][:,-1,:], batch["actions"][:,-1,:], batch["next_states"][:,-1,:])

        penalty = ((rewards_pred - batch["rewards"][:,-1]) ** 2)

        loss = penalty.mean()

        new_metrics = metrics.update(
            {
                "rewards_loss": loss,
            }
        )
        return loss, new_metrics

    grads, new_metrics = jax.grad(rewards_loss_fn, has_aux=True)(rewards.params)
    new_rewards = rewards.apply_gradients(grads=grads)
    '''
    new_rewards = new_rewards.replace(
        target_params=optax.incremental_update(rewards.params, rewards.target_params, tau)
    )
    '''
    return key, new_rewards, new_metrics

def update_dones(
    key: jax.random.PRNGKey,
    dones: TrainState,
    batch: Dict[str, jax.Array],
    tau: float,
    metrics: Metrics,
) -> Tuple[jax.random.PRNGKey, TrainState, Metrics]:
    key, random_state_key = jax.random.split(key, 2)

    def dones_loss_fn(params: jax.Array) -> Tuple[jax.Array, Metrics]:
        dones_pred = dones.apply_fn(params, batch["states"][:,-1,:], batch["actions"][:,-1,:], batch["next_states"][:,-1,:])

        penalty = ((dones_pred - batch["dones"][:,-1]) ** 2)

        loss = penalty.mean()

        new_metrics = metrics.update(
            {
                "dones_loss": loss,
            }
        )
        return loss, new_metrics

    grads, new_metrics = jax.grad(dones_loss_fn, has_aux=True)(dones.params)
    new_dones = dones.apply_gradients(grads=grads)
    '''
    new_dones = new_dones.replace(
        target_params=optax.incremental_update(dones.params, dones.target_params, tau)
    )
    '''

    return key, new_dones, new_metrics

@pyrallis.wrap()
def main(config: Config):
    dict_config = asdict(config)
    dict_config["mlc_job_name"] = os.environ.get("PLATFORM_JOB_NAME")

    wandb.init(
        config=dict_config,
        project=config.project,
        group=config.group,
        name=config.name,
        id=str(uuid.uuid4()),
    )
    wandb.mark_preempting()
    buffer = ReplayBuffer()
    buffer.create_from_d4rl(
        config.dataset_name, config.normalize_reward, config.normalize_states
    )

    random.seed(config.train_seed)
    key = jax.random.PRNGKey(seed=config.train_seed)
    key, dynamics_key, rewards_key, dones_key = jax.random.split(key, 4)

    eval_env = make_env(config.dataset_name, seed=config.eval_seed)
    eval_env = wrap_env(eval_env, buffer.mean, buffer.std)
    #init_state = buffer.data["states"][0][None, ...]
    #init_action = buffer.data["actions"][0][None, ...]
    init_state = np.zeros((1, config.time_steps, buffer.d4rl_data["states"].shape[-1]))
    init_action = np.zeros((1, config.time_steps, buffer.d4rl_data["actions"].shape[-1]))

    dynamics_module_ = Dynamics(
        state_dim=init_state.shape[-1],
        action_dim=init_action.shape[-1],
        hidden_dim=config.hidden_dim,
    )
    dynamics_ = DynamicsTrainState.create(
        apply_fn=dynamics_module_.apply,
        params=dynamics_module_.init(dynamics_key, init_state, init_action),
        target_params=dynamics_module_.init(dynamics_key, init_state, init_action),
        tx=optax.adam(learning_rate=config.dynamics_learning_rate),
    )
    rewards_module_ = Rewards(
        state_dim=init_state.shape[-1],
        action_dim=init_action.shape[-1],
        hidden_dim=config.hidden_dim,
        layernorm=config.rewards_ln,
        n_hiddens=config.rewards_n_hiddens,
    )
    rewards_ = RewardsTrainState.create(
        apply_fn=rewards_module_.apply,
        params=rewards_module_.init(rewards_key, init_state[:,-1,:], init_action[:,-1,:], init_state[:,-1,:]),
        target_params=rewards_module_.init(rewards_key, init_state[:,-1,:], init_action[:,-1,:], init_state[:,-1,:]),
        tx=optax.adam(learning_rate=config.rewards_learning_rate),
    )
    dones_module_ = Dones(
        state_dim=init_state.shape[-1],
        action_dim=init_action.shape[-1],
        hidden_dim=config.hidden_dim,
        layernorm=config.dones_ln,
        n_hiddens=config.dones_n_hiddens,
    )
    dones_ = DonesTrainState.create(
        apply_fn=dones_module_.apply,
        params=dones_module_.init(dones_key, init_state[:,-1,:], init_action[:,-1,:], init_state[:,-1,:]),
        target_params=dones_module_.init(dones_key, init_state[:,-1,:], init_action[:,-1,:], init_state[:,-1,:]),
        tx=optax.adam(learning_rate=config.dones_learning_rate),
    )

    config_dict = config.dict()
    if config.reload_chkpt and os.path.exists(config.chkpt_dir):
        target = {
            "dynamics": dynamics_,
            "rewards": rewards_,
            "dones": dones_,
            "config": config_dict,
            "init_state": init_state,
            "init_action": init_action,
        }
        chkdata = checkpoints.restore_checkpoint(ckpt_dir=config.chkpt_dir,
                            target=target,
                            step=0)        
        dynamics = DynamicsTrainState.create(
            apply_fn=dynamics_module_.apply,
            params=chkdata["dynamics"].params,
            target_params=dynamics_module_.init(dynamics_key, init_state, init_action),
            tx=optax.adam(learning_rate=config.dynamics_learning_rate),
        )
        #print("restored dynamics.params", dynamics.params["params"]["update_gate1"]["kernel"])
        rewards = chkdata["rewards"]
        dones = chkdata["dones"]
        config_dict = chkdata["config"]
        init_state = chkdata["init_state"]
        init_action = chkdata["init_action"]
        print("Successfully loaded checkpoint from", config.chkpt_dir)
    else:
        dynamics = dynamics_
        rewards = rewards_
        dones = dones_

    #print("after restore dynamics.params", dynamics.params["params"]["update_gate1"]["kernel"])

    #print("after restore chkdata", chkdata["dynamics"].params["params"]["update_gate1"]["kernel"])
    # metrics
    bc_metrics_to_log = [
        "dynamics_loss",
        "rewards_loss",
        "dones_loss",
    ]

    @jax.jit
    def dynamics_fn(params: jax.Array, obs: jax.Array, act: jax.Array):
        return dynamics.apply_fn(params, obs, act)
    
    @jax.jit
    def rewards_fn(params: jax.Array, obs: jax.Array, act: jax.Array, next_obs: jax.Array):
        return rewards.apply_fn(params, obs, act, next_obs)
    
    @jax.jit
    def dones_fn(params: jax.Array, obs: jax.Array, act: jax.Array, next_obs: jax.Array):
        return dones.apply_fn(params, obs, act, next_obs)
    
    @jax.jit
    def td3_loop_update_step(i: int, carry: TrainState, batch : Dict[str, jax.Array]):
        key, new_dynamics, new_metrics = update_dynamics(
            carry["key"],
            carry["dynamics"],
            batch,
            config.tau,
            carry["metrics"],
        )        
        carry.update(key=key, dynamics=new_dynamics, metrics=new_metrics)
        key, new_rewards, new_metrics = update_rewards(
            carry["key"],
            carry["rewards"],
            batch,
            config.tau,
            carry["metrics"],
        )
        carry.update(key=key, rewards=new_rewards, metrics=new_metrics)
        key, new_dones, new_metrics = update_dones(
            carry["key"],
            carry["dones"],
            batch,
            config.tau,
            carry["metrics"],
        )
        carry.update(key=key, dones=new_dones, metrics=new_metrics)        
        return carry
    
    myenv = MyEnv()
    myenv.load(dynamics_params=dynamics.params, rewards_params=rewards.params, dones_params=dones.params,
                dynamics_fn=dynamics_fn, rewards_fn=rewards_fn, dones_fn=dones_fn, time_steps=config.time_steps)

    # shared carry for update loops
    update_carry = {
        "key": key,
        "dynamics": dynamics,
        "rewards": rewards,
        "dones": dones,
        "buffer": buffer,
    }
    for epoch in trange(config.num_epochs, desc="myenv_gru Epochs"):
        # metrics for accumulation during epoch and logging to wandb
        # we need to reset them every epoch
        update_carry["metrics"] = Metrics.create(bc_metrics_to_log)

        '''
        update_carry = jax.lax.fori_loop(
            lower=0,
            upper=config.num_updates_on_epoch,
            body_fun=td3_loop_update_step,
            init_val=update_carry,
        )
        '''
        
        for i in range(config.num_updates_on_epoch):
            key, batch_key = jax.random.split(update_carry["key"])
            batch = update_carry["buffer"].sample_batch(batch_key, time_steps=config.time_steps)
            update_carry["key"] = key
            update_carry = td3_loop_update_step(i, update_carry, batch)

        # log mean over epoch for each metric
        mean_metrics = update_carry["metrics"].compute()
        wandb.log(
            {"epoch": epoch, **{f"myenv_gru/{k}": v for k, v in mean_metrics.items()}}
        )
        '''
        if epoch % config.eval_every == 0 or epoch == config.num_epochs - 1:
            dynamics_loss, rewards_loss, dones_loss, key = evaluate_offline(
                myenv,
                buffer,
                config.eval_episodes,
                config.eval_steps,
                key=key,
            )
            #print("dynamics_loss", dynamics_loss)
            wandb.log(
                {
                    "epoch": epoch,
                    "eval/dynamics_loss_mean": np.mean(dynamics_loss),
                    "eval/dynamics_loss_std": np.std(dynamics_loss),
                    "eval/rewards_loss_mean": np.mean(rewards_loss),
                    "eval/rewards_loss_std": np.std(rewards_loss),
                    "eval/dones_loss_mean": np.mean(dones_loss),
                    "eval/dones_loss_std": np.std(dones_loss),
                }
            )
        '''

    # save to checkpoint
    shutil.rmtree(config.chkpt_dir, ignore_errors=True)
    ckpt = {
        "dynamics": update_carry["dynamics"],
        "rewards": update_carry["rewards"],
        "dones": update_carry["dones"],
        "config": config_dict,
        "init_state": init_state,
        "init_action": init_action,
    }
    #print("saved params", ckpt["dynamics"].params["params"]["update_gate1"]["kernel"]) # is OK
    checkpoints.save_checkpoint(ckpt_dir=config.chkpt_dir,
                            target=ckpt,
                            step=0,
                            overwrite=True,
                            keep=2)
    print("Successfully saved checkpoint to", config.chkpt_dir)


if __name__ == "__main__":
    main()
