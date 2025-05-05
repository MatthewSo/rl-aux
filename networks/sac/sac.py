import gymnasium as gym
from stable_baselines3 import SAC
from networks.common import CustomFeatureExtractor, ActionNet, ValueNet, ContinuousToMultiDiscrete


def get_sac_agent(
        env: gym.Env,
        feature_dim: int,
        auxiliary_dim: int,
        weight_bins: int,
        device: str,
        batch_size: int,
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        ent_coef: str | float = "auto",
):
    '''
    This is an experimental implementation of a SAC version of the task selection agent.
    '''
    if isinstance(env.action_space, gym.spaces.MultiDiscrete):
        env = ContinuousToMultiDiscrete(env)

    policy_kwargs = dict(
        features_extractor_class=CustomFeatureExtractor,
        features_extractor_kwargs=dict(features_dim=feature_dim),
        net_arch=[],
    )

    model = SAC(
        policy="MultiInputPolicy",
        env=env,
        device=device,
        learning_rate=learning_rate,
        batch_size=batch_size,
        gamma=gamma,
        tau=tau,
        buffer_size=1_000_000,
        ent_coef=ent_coef,
        verbose=0,
        policy_kwargs=policy_kwargs,
    )

    action_net = ActionNet(feature_dim, auxiliary_dim + weight_bins).to(device)
    value_net = ValueNet(feature_dim).to(device)

    model.policy.action_net = action_net
    model.policy.value_net = value_net

    return model