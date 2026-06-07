import os
import highway_env
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
from stable_baselines3 import PPO
from config import MODEL_SAVE_PATH, VIDEO_SAVE_PATH

def main() -> None:
    print("evaluating the agent...")
    
    os.makedirs(VIDEO_SAVE_PATH, exist_ok=True)
    env = gym.make("highway-fast-v0", render_mode="rgb_array")
    
    # wrap env to record video
    env = RecordVideo(env, video_folder=VIDEO_SAVE_PATH, episode_trigger=lambda e: True)
    
    # load our trained model
    model = PPO.load(MODEL_SAVE_PATH)
    
    obs, info = env.reset()
    done = False
    truncated = False
    
    while not (done or truncated):
        # predict the best action
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        
    env.close()
    print("eval done, check videos folder")

if __name__ == "__main__":
    main()