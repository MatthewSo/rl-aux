import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torchvision import models
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
        return self.fc(x).reshape(-1, self.auxiliary_dim)

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

def get_ppo_agent( env, feature_dim, auxiliary_dim, device, learning_rate=0.001,ent_coef=0.01,n_steps=79, n_epochs=10, batch_size=64):
    # Set up the RL PPO agent (of course other agent types may make sense too)
    policy_kwargs = {
        "features_extractor_class": CustomFeatureExtractor, #CustomFeatureExtractor,
        "features_extractor_kwargs": {"features_dim": feature_dim},  # Dimensionality of the output features
        "net_arch":[],
    }

    model = PPO("MultiInputPolicy", env, verbose=0, policy_kwargs=policy_kwargs,batch_size=batch_size, learning_rate=learning_rate,ent_coef=ent_coef,n_steps=n_steps,n_epochs=n_epochs)
    action_net = ActionNet(feature_dim, auxiliary_dim)
    action_net = action_net.to(device)
    model.policy.action_net = action_net

    value_net = ValueNet(feature_dim)
    value_net = value_net.to(device)
    model.policy.value_net = value_net

    return model