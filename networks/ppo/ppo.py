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

import os
from collections import OrderedDict

import torch
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.save_util import load_from_zip_file

from networks.common import CustomFeatureExtractor                      # backbone
from networks.common import ActionNet, ValueNet                         # original heads


def _rename_fc_keys(state_dict: dict) -> dict:
    """turn  action_net.fc.* → action_net.*   and   value_net.fc.* → value_net.*"""
    new_sd = OrderedDict()
    for k, v in state_dict.items():
        if k.startswith("action_net.fc."):
            new_sd[k.replace("action_net.fc.", "action_net.")] = v
        elif k.startswith("value_net.fc."):
            new_sd[k.replace("value_net.fc.", "value_net.")] = v
        else:
            new_sd[k] = v
    return new_sd


def load_ppo_labeler(checkpoint_dir: str,
                     obs_shape=(3, 32, 32),
                     device: str | torch.device = "cpu") -> PPO:
    """
    Restore a PPO labeler whose checkpoint contains `fc.*` sub-keys.

    Parameters
    ----------
    checkpoint_dir : str
        Folder that contains `agent.zip` (saved via `env.save(..., dir)`).
    obs_shape : tuple[int]
        Image shape so we can create a dummy env for policy instantiation.
    device : str | torch.device
        Destination device.

    Returns
    -------
    PPO
        Fully loaded PPO object.
    """
    agent_path = os.path.join(checkpoint_dir, "agent")

    # ------------------------------------------------------------------ #
    # 1) make a dummy single-step env just to build the policy structure
    # ------------------------------------------------------------------ #
    class _DummyEnv(gym.Env):
        def __init__(self):
            super().__init__()
            self.observation_space = spaces.Dict(
                {"image": spaces.Box(0, 1, shape=obs_shape, dtype=float)}
            )
            self.action_space = spaces.MultiDiscrete([1])          # placeholder
        def reset(self, *, seed=None, options=None): return {"image": torch.zeros(obs_shape)}, {}
        def step(self, action): return self.reset()[0], 0.0, True, False, {}

    dummy_env = _DummyEnv()

    policy_kwargs = dict(
        features_extractor_class=CustomFeatureExtractor,
        features_extractor_kwargs=dict(features_dim=512),          # value never used
        net_arch=[],
    )
    empty_agent = PPO("MultiInputPolicy", dummy_env,
                      policy_kwargs=policy_kwargs,
                      device=device, verbose=0)

    # ------------------------------------------------------------------ #
    # 2) read raw checkpoint tensors, rename keys, inject them
    # ------------------------------------------------------------------ #
    params, _, _ = load_from_zip_file(agent_path, device=device)
    renamed = _rename_fc_keys(params["policy"])
    empty_agent.policy.load_state_dict(renamed, strict=True)
    empty_agent.policy.eval()
    return empty_agent