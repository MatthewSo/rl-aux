import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torchvision import models
import torch
import torch.nn as nn
import numpy as np
from gymnasium import spaces
from stable_baselines3 import PPO
from networks.primary.vgg import VGG16

class CustomFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim):
        super(CustomFeatureExtractor, self).__init__(observation_space, features_dim)
        self.net = VGG16(primary_task_output=features_dim, auxiliary_task_output=features_dim)

    def forward(self, observations):
        observation=observations['image']
        x = self.net(observation)[0]
        return x

class ActionNet(nn.Module):
    def __init__(self, feature_dim, auxiliary_dim):
        super(ActionNet, self).__init__()
        self.auxiliary_dim = auxiliary_dim
        self.feature_dim = feature_dim
        self.fc = nn.Linear(self.feature_dim, auxiliary_dim)

    def forward(self, x):
        x=x.reshape(-1,self.feature_dim)
        x = self.fc(x)
        x = x.reshape(-1, self.auxiliary_dim)
        return x

class ValueNet(nn.Module):
    def __init__(self, feature_dim):
        super(ValueNet, self).__init__()
        self.feature_dim = feature_dim
        self.fc = nn.Linear(feature_dim, 1)

    def forward(self, x):
        x=x.reshape(-1,self.feature_dim)
        x=self.fc(x)
        x=x.squeeze(-1).mean(dim=-1)
        return x