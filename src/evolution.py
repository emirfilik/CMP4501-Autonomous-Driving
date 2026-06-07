import os
import highway_env
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
from stable_baselines3 import PPO

def record_model(model_path: str, video_folder: str) -> None:
    env = gym.make("highway-fast-v0", render_mode="rgb_array")
    env = RecordVideo(env, video_folder=video_folder, episode_trigger=lambda e: True)
    
    model = PPO.load(model_path)
    obs, info = env.reset()
    done = False
    truncated = False
    
    while not (done or truncated):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        
    env.close()

def main() -> None:
    print("recording untrained agent (stage 1)...")
    record_model("assets/ppo_untrained", "videos/stage1_untrained")
    
    print("recording half-trained agent (stage 2)...")
    record_model("assets/ppo_half_10000_steps", "videos/stage2_half")
    
    print("recording fully trained agent (stage 3)...")
    record_model("assets/ppo_fully_trained", "videos/stage3_full")
    
    print("all 3 evolution stages recorded! check the videos folder.")

if __name__ == "__main__":
    main()