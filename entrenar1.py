import torch
import numpy as np
import os
from dopamine.discrete_domains import run_experiment
from dopamine.colab import utils as colab_utils
from absl import flags
import gin.tf
import tensorflow as tf
from dopamine.agents.dqn import dqn_agent

BASE_PATH = '/home/balam/balam/Fake_News/tmp/fisrt_dqn_test/'  # @param

# @title Load the configuration for DQN.
DQN_PATH = os.path.join(BASE_PATH, 'dqn3')
# Modified from dopamine/agents/dqn/config/dqn_cartpole.gin
dqn_config = """
# Hyperparameters for a simple DQN-style Cartpole agent. The hyperparameters
# chosen achieve reasonable performance.
import dopamine.discrete_domains.gym_lib
import dopamine.discrete_domains.run_experiment
import dopamine.agents.dqn.dqn_agent
import dopamine.replay_memory.circular_replay_buffer
import gin.tf.external_configurables
import gym_fake
import testConfiguration
import tensorflow


DQNAgent.observation_shape = (1538,1)
DQNAgent.observation_dtype = %diaspora.OBSERVATION_DTYPE
DQNAgent.stack_size = 1
DQNAgent.network = @testConfiguration.diaspora_dqn_network
DQNAgent.gamma = 0.99
DQNAgent.update_horizon = 1
DQNAgent.min_replay_history = 500
DQNAgent.update_period = 4
DQNAgent.target_update_period = 100
DQNAgent.epsilon_fn = @dqn_agent.identity_epsilon
DQNAgent.tf_device = '/gpu:0'  # use '/cpu:*' for non-GPU version
DQNAgent.optimizer = @tf.train.AdamOptimizer()

tf.train.AdamOptimizer.learning_rate = 0.001
tf.train.AdamOptimizer.epsilon = 0.0003125

create_gym_environment.environment_name = 'gym_fake'
create_gym_environment.version = 'v0'
create_agent.agent_name = 'dqn'
TrainRunner.create_environment_fn = @gym_lib.create_gym_environment
Runner.num_iterations = 4
Runner.training_steps = 4
Runner.evaluation_steps = 4
Runner.max_steps_per_episode = 4  # Default max episode length.

WrappedReplayBuffer.replay_capacity = 10
WrappedReplayBuffer.batch_size = 2
"""
gin.parse_config(dqn_config, skip_unknown=False)


# @title Train DQN on Cartpole
dqn_runner = run_experiment.create_runner(DQN_PATH, schedule='continuous_train')
print('Will train DQN agent, please be patient, may be a while...')


dqn_runner.run_experiment()
print('Done training!')

data = colab_utils.read_experiment(DQN_PATH, verbose=True,
                                   summary_keys=['train_episode_returns'])
data['agent'] = 'DQN'
data['run'] = 1



import seaborn as sns
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(16,8))
sns.tsplot(data=data, time='iteration', unit='run',
           condition='agent', value='train_episode_returns', ax=ax)
plt.title('Cartpole')
plt.show()

