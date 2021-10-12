import flax
from graphviz import Digraph
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from IPython.display import HTML
from pprint import pprint
import logging
import os
import gym
from dopamine.jax import networks
from dopamine.jax.agents.dqn import dqn_agent
import gin

from flax import linen as nn

env = gym.make("gym_fake:fake-v0")
#env=gym.make('CartPole-v1')

from dopamine.discrete_domains import atari_lib
class FakeDQNNetwork(nn.Module):
  """Jax DQN network for Cartpole."""
  num_actions: int

  @nn.compact
  def __call__(self, x):
    initializer = nn.initializers.xavier_uniform()
    x = x.astype(jnp.float32)
    print(1)
    x = x.reshape((-1))  # flatten
    #x -= np.array(np.ones(768)*(-10))
    #print(2)
    #x /= np.array(np.ones(768)*(-10)) - np.array(np.ones(768)*(-10))
    #x = 2.0 * x - 1.0  # Rescale in range [-1, 1].
    x = nn.Dense(features=512, kernel_init=initializer)(x)
    x = nn.relu(x)
    x = nn.Dense(features=512, kernel_init=initializer)(x)
    x = nn.relu(x)
    print(3)
    q_values = nn.Dense(features=self.num_actions, kernel_init=initializer)(x)
    print(4)
    return atari_lib.DQNNetworkType(q_values)

Fake_config = """
JaxDQNAgent.gamma = 0.99
JaxDQNAgent.update_horizon = 1
JaxDQNAgent.min_replay_history = 500
JaxDQNAgent.update_period = 4
JaxDQNAgent.target_update_period = 100
JaxDQNAgent.epsilon_fn = @dqn_agent.identity_epsilon

create_optimizer.name = 'adam'
create_optimizer.learning_rate = 0.001
create_optimizer.eps = 3.125e-4

OutOfGraphReplayBuffer.replay_capacity = 50000
OutOfGraphReplayBuffer.batch_size = 128
"""
gin.parse_config(Fake_config, skip_unknown=False)

dqn_agent = dqn_agent.JaxDQNAgent(num_actions=8,
                                  observation_shape=(1538, 1),
                                  observation_dtype=jnp.float64,
                                  stack_size=1,
                                  network=FakeDQNNetwork)
                 
max_steps_per_episode = 4  # @param {type:'slider', min:10, max:1000}
training_steps = 4  # @param {type:'slider', min:10, max:5000}
num_iterations = 4 #@param {type:"slider", min:10, max:200, step:1}

# First remove eval mode!
dqn_agent.eval_mode = False
average_returns = []
# Each iteration will consist of a number of episodes.
for iteration in range(num_iterations):
  step_count = 0
  num_episodes = 0
  sum_returns = 0.
  # This while loop will continue running episodes until we've done enough
  # training steps.
  while step_count < training_steps:
    episode_length = 0
    episode_rewards = 0.
    s = env.reset()
    a = int(dqn_agent.begin_episode(s))
    
    is_terminal = False
    # Run the episode until termination.
    while True:
      #r=env.step(a)[1]
      #s=env.step(a)[0]
      #done=env.step(a)[2]
      b=env.step(a)
      s, r, done, _ = b
      episode_rewards += r
      episode_length += 1
      if done or episode_length == max_steps_per_episode:
        # Stop the loop if the episode or ended or we've reached the max steps.
        break
      else:
        a = int(dqn_agent.step(r, s))
    dqn_agent.end_episode(r)
    step_count += episode_length
    sum_returns += episode_rewards
    num_episodes += 1
  average_return = sum_returns / num_episodes if num_episodes > 0 else 0.
  print(f'Iteration {iteration}: average return = {average_return:.4f}')
  average_returns.append(average_return)                 
