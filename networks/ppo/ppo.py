from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch
import torch.nn as nn
import numpy as np
from gymnasium import spaces
from stable_baselines3 import PPO

from networks.common import CustomFeatureExtractor, ActionNet, ValueNet


def get_ppo_agent( env, feature_dim, auxiliary_dim, weight_bins, device, batch_size, learning_rate=0.001,ent_coef=0.01,n_steps=79, n_epochs=10, input_shape=(3, 32, 32)):
    '''
    This function creates a PPO agent with a custom feature extractor and action/value networks.
    '''
    # Set up the RL PPO agent (of course other agent types may make sense too)
    policy_kwargs = {
        "features_extractor_class": CustomFeatureExtractor, #CustomFeatureExtractor,
        "features_extractor_kwargs": {"features_dim": feature_dim, "input_shape": input_shape},
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
    """
    Returns a dummy PPO agent with a fast feature extractor.
    Not intended for actual use, but for testing purposes.
    """
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