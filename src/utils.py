import gymnasium as gym
import highway_env  # noqa: F401

# helper to create the environment
def make_env(render_mode: str = "rgb_array") -> gym.Env:
    env = gym.make("highway-fast-v0", render_mode=render_mode)
    return env