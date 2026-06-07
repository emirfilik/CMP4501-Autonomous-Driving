import os
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback
from config import TOTAL_TIMESTEPS, LEARNING_RATE, BATCH_SIZE
from utils import make_env
from model import create_model

def main() -> None:
    print("starting training process...")
    os.makedirs("assets", exist_ok=True)
    
    env = make_env()
    # monitor wraps the env to save reward history to assets/monitor.csv
    env = Monitor(env, "assets/")
    
    model = create_model(env, LEARNING_RATE, BATCH_SIZE)
    
    # save untrained model (stage 1)
    model.save("assets/ppo_untrained")
    
    # callback to save half-trained model at 10000 steps (stage 2)
    checkpoint_callback = CheckpointCallback(save_freq=10000, save_path='./assets/', name_prefix='ppo_half')
    
    print(f"training for {TOTAL_TIMESTEPS} timesteps")
    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=checkpoint_callback)
    
    # save fully trained model (stage 3)
    model.save("assets/ppo_fully_trained")
    print("training finished and all stages saved.")
    
    env.close()

if __name__ == "__main__":
    main()