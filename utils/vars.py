import gym
import torch
import torch.nn as nn
from stable_baselines3 import SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torchvision import models
import torch.nn as nn
import torch.optim as optim
import numpy as np
from gym import spaces
from torch.utils.data import DataLoader, RandomSampler, SubsetRandomSampler
from tqdm import tqdm
import copy
import random
import sys
from stable_baselines3 import PPO
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from create_dataset import CIFAR100,ImbalancedDatasetWrapper
from torch.utils.data import Dataset
from sklearn.metrics import f1_score
import sys


softmax= nn.Softmax(dim=-1)