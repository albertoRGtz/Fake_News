import gym
from gym import error, spaces, utils
from gym.utils import seeding
import pandas as pd
import json
import re
import numpy as np
import itertools
import tensorflow as tf
from collections import Counter
from keras.preprocessing.sequence import pad_sequences
import string
from transformers import pipeline
import torch
from transformers import AutoModel, AutoTokenizer, BertTokenizer, BertForSequenceClassification, AutoModelForSequenceClassification, BertModel, AutoConfig
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertTokenizer, BertConfig

env = gym.make("gym_fake:fake-v0")

print(env.observation_space)

print(env.action_space)

print(env.reward_range)

print(env.metadata)

print(env.action_space.n)



env.reset()


