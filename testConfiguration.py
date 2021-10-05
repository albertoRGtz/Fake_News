import itertools
import math
from dopamine.discrete_domains.gym_lib import BasicDiscreteDomainNetwork
from dopamine.discrete_domains import atari_lib

import torch
import gym
import numpy as np
import tensorflow as tf

import gin.tf
import gin


gin.constant('diaspora.OBSERVATION_DTYPE', tf.float64)

@gin.configurable
#def diaspora_dqn_network(num_actions, network_type, state):
#  """Builds the deep network used to compute the agent's Q-values.
#  It rescales the input features to a range that yields improved performance.
#  Args:
#    num_actions: int, number of actions.
#    network_type: namedtuple, collection of expected values to return.
#    state: `tf.Tensor`, contains the agent's current state.
#  Returns:
#    net: _network_type object containing the tensors output by the network.
#  """
#  q_values = BasicDiscreteDomainNetwork(
#       np.array([0., 0.]), np.array([99., 99.]), num_actions, state)
#  return network_type(q_values)
  
class diaspora_dqn_network(tf.keras.Model):
  """Keras DQN network for Cartpole."""

  def __init__(self, num_actions, name=None):
    """Builds the deep network used to compute the agent's Q-values.
    It rescales the input features so they lie in range [-1, 1].
    Args:
      num_actions: int, number of actions.
      name: str, used to create scope for network parameters.
    """
    super(diaspora_dqn_network, self).__init__(name=name)
    #t1 = tf.convert_to_tensor(np.array(np.ones(768)*(-10)),dtype=tf.float32)
    t1 = np.array(np.ones(1538)*(-10))
    #print('t1: ')
    #print(t1)
    #t2 = tf.convert_to_tensor(np.array(np.ones(768)*10),dtype=tf.float32)
    t2 = np.array(np.ones(1538)*10)
    #print('t2: ')
    #print(t2)
    self.net = BasicDiscreteDomainNetwork(t1, t2, num_actions)


  def call(self, state):
    """Creates the output tensor/op given the state tensor as input."""
    x = self.net(state)
    return  atari_lib.DQNNetworkType(x)
