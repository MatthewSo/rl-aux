import gymnasium as gym
import torch
import torch.nn as nn
from stable_baselines3 import SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torchvision import models
import torch.nn as nn
import torch.optim as optim
import numpy as np
from gymnasium import spaces
from torch.utils.data import DataLoader, RandomSampler, SubsetRandomSampler
import copy
import random
import sys
from stable_baselines3 import PPO
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from sklearn.metrics import f1_score
import sys

class SeededSubsetRandomSampler(SubsetRandomSampler):
    # Custom sample that uses a seed for shuffling. Used in the training loop.
    def __init__(self, indices, seed=None):
        self.seed = seed
        super().__init__(indices)

    def __iter__(self):
        if self.seed is not None:
            # Use local random state to shuffle indices based on the given seed
            np_random_state = np.random.RandomState(self.seed)
            indices = list(self.indices)
            np_random_state.shuffle(indices)
            return iter(indices)
        else:
            # Default behavior if no seed is provided
            return super().__iter__()