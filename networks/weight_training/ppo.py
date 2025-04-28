from networks.common import CustomFeatureExtractor, WeightNet, ValueNet
from stable_baselines3 import PPO


def get_weight_training_ppo_agent(env, feature_dim, device, batch_size,
                  learning_rate=3e-4, ent_coef=0.01,
                  n_steps=2048, n_epochs=10):

    policy_kwargs = dict(
        features_extractor_class=CustomFeatureExtractor,
        features_extractor_kwargs=dict(features_dim=feature_dim),
        net_arch=[],
    )

    model = PPO("MultiInputPolicy",
                env,
                verbose=0,
                policy_kwargs=policy_kwargs,
                batch_size=batch_size,
                learning_rate=learning_rate,
                ent_coef=ent_coef,
                n_steps=n_steps,
                n_epochs=n_epochs,
                device=device)

    model.policy.action_net = WeightNet(feature_dim).to(device)
    model.policy.value_net  = ValueNet(feature_dim).to(device)

    return model