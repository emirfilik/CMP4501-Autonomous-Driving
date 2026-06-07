import gymnasium as gym
from stable_baselines3 import PPO

# wrapping the PPO model setup
def create_model(env: gym.Env, learning_rate: float, batch_size: int) -> PPO:
    # standard mlp policy for kinematic data
    model = PPO(
        "MlpPolicy", 
        env, 
        learning_rate=learning_rate,
        batch_size=batch_size,
        verbose=1
    )
    return model