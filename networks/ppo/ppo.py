import os

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch
import torch.nn as nn
import numpy as np
from gymnasium import spaces
from stable_baselines3 import PPO

from networks.common import CustomFeatureExtractor, ActionNet, ValueNet



def get_ppo_agent( env, feature_dim, auxiliary_dim, weight_bins, device, batch_size, learning_rate=0.001,ent_coef=0.01,n_steps=79, n_epochs=10):
    # Set up the RL PPO agent (of course other agent types may make sense too)
    policy_kwargs = {
        "features_extractor_class": CustomFeatureExtractor, #CustomFeatureExtractor,
        "features_extractor_kwargs": {"features_dim": feature_dim},  # Dimensionality of the output features
        "net_arch":[],
    }

    model = PPO("MultiInputPolicy", env, verbose=0, policy_kwargs=policy_kwargs,batch_size=batch_size, learning_rate=learning_rate,ent_coef=ent_coef,n_steps=n_steps,n_epochs=n_epochs, device=device)
    action_net = ActionNet(feature_dim, auxiliary_dim + weight_bins)
    action_net = action_net.to(device)
    model.policy.action_net = action_net

    value_net = ValueNet(feature_dim)
    value_net = value_net.to(device)
    model.policy.value_net = value_net

    return model


def get_fast_dummy_ppo_agent(
        env,
        device,
        learning_rate=3e-4,
        hidden_size=32,
        n_steps=16,
        batch_size=16,
        n_epochs=1,
):
    class FastDictExtractor(BaseFeaturesExtractor):


        def __init__(self, observation_space: spaces.Dict):
            flat_dim = sum(int(np.prod(space.shape))
                           for space in observation_space.spaces.values())
            super().__init__(observation_space, features_dim=flat_dim)
            self.flatten = nn.Flatten()

        def forward(self, obs):
            parts = [self.flatten(t) for t in obs.values()]
            return torch.cat(parts, dim=1)

    policy_kwargs = dict(
        features_extractor_class=FastDictExtractor,
        net_arch=[dict(pi=[hidden_size], vf=[hidden_size])],
    )

    model = PPO(
        policy="MultiInputPolicy",
        env=env,
        device=device,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=0.99,
        ent_coef=0.0,
        verbose=0,
        policy_kwargs=policy_kwargs,
    )
    return model

class _HeadLinear(nn.Module):
    def __init__(self, in_dim: int, out_dim: int = 1):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_dim, in_dim))
        self.bias   = nn.Parameter(torch.empty(out_dim))
        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.bias)

    def forward(self, x):
        x = x.reshape(-1, self.weight.shape[1])
        return nn.functional.linear(x, self.weight, self.bias)


CompatActionNet = lambda in_dim, out_dim: _HeadLinear(in_dim, out_dim)
CompatValueNet  = lambda in_dim:          _HeadLinear(in_dim, 1)


def load_ppo_labeler(checkpoint_dir: str,
                     device: str | torch.device = "cpu") -> PPO:
    path = os.path.join(checkpoint_dir, "agent")
    agent = PPO.load(
        path,
        device=device,
        custom_objects=dict(
            CustomFeatureExtractor=CustomFeatureExtractor,
            ActionNet=CompatActionNet,
            ValueNet = CompatValueNet,
        ),
        print_system_info=False,
    )
    agent.policy.eval()
    return agent