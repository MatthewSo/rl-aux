import gymnasium as gym
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch
import torch.nn as nn
import numpy as np
from networks.primary.vgg import VGG16

class CustomFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim, input_shape=(3, 32, 32)):
        super(CustomFeatureExtractor, self).__init__(observation_space, features_dim)
        self.net = VGG16(primary_task_output=features_dim, auxiliary_task_output=features_dim, input_shape=input_shape)

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

class WeightNet(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()
        self.fc = nn.Linear(feature_dim, 1)

    # def forward(self, x):
    #     x = x.reshape(-1, self.fc.in_features)
    #     return torch.sigmoid(self.fc(x)).squeeze(-1)

    def forward(self, x):
        x = x.view(-1, self.fc.in_features)
        x = self.fc(x)
        return x.squeeze(-1)

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

class ContinuousToMultiDiscrete(gym.ActionWrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)

        assert isinstance(env.action_space, gym.spaces.MultiDiscrete)

        self.sizes = env.action_space.nvec           # e.g. [100, 21]
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(len(self.sizes),), dtype=np.float32
        )

    def _cont_to_discrete(self, a: np.ndarray) -> np.ndarray:
        scaled   = (a + 1.0) * 0.5                    # -> [0, 1]
        indices  = np.round(scaled * (self.sizes - 1)).astype(np.int64)
        return np.clip(indices, 0, self.sizes - 1)

    def action(self, act: np.ndarray) -> np.ndarray:
        return self._cont_to_discrete(act)
