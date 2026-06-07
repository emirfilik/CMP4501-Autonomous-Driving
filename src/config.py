# hyperparams for the highway environment
# keeping it light so it doesn't melt the cpu

ENV_NAME: str = "highway-fast-v0"
TOTAL_TIMESTEPS: int = 20000 
LEARNING_RATE: float = 5e-4
BATCH_SIZE: int = 64
MODEL_SAVE_PATH: str = "assets/ppo_highway"
VIDEO_SAVE_PATH: str = "videos"