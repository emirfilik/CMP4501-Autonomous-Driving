import pandas as pd
import matplotlib.pyplot as plt
import os

def main() -> None:
    print("generating reward graph...")
    # read the monitor file created during training
    df = pd.read_csv("assets/monitor.csv", skiprows=1)
    
    plt.figure(figsize=(10, 5))
    # rolling mean smooths out the graph lines
    plt.plot(df['r'].rolling(window=10).mean(), color='blue', label='Reward (10-ep rolling mean)')
    plt.xlabel('Episodes')
    plt.ylabel('Reward')
    plt.title('Training Performance - CMP4501')
    plt.legend()
    plt.grid(True)
    
    os.makedirs("assets", exist_ok=True)
    plt.savefig("assets/reward_plot.png")
    print("graph saved as assets/reward_plot.png")

if __name__ == "__main__":
    main()