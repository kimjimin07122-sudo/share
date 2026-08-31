import matplotlib.pyplot as plt
import numpy as np

def plot_reconstruction_error(time_sec, errors, save_path="reconstruction_error.png"):
    plt.figure(figsize=(10, 6))
    plt.plot(time_sec, errors, color='black', linewidth=0.5)
    plt.xlabel("time (sec)")
    plt.ylabel("reconstruction error")
    plt.grid(False)
    plt.savefig(save_path)
    plt.close()

def plot_channel_identification(channel_scores, save_path="channel_id.png"):
    channels = np.arange(1, len(channel_scores) + 1)
    plt.figure(figsize=(10, 5))
    plt.bar(channels, channel_scores, width=0.6, color="#005b96")
    plt.xticks(channels)
    plt.xlabel("channel number")
    plt.ylabel("mean identification score (Softmax)")
    plt.savefig(save_path)
    plt.close()