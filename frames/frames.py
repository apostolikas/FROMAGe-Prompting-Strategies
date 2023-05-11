import requests
from copy import copy

from PIL import Image
import matplotlib.pyplot as plt


def download_gif(gif_url: str, save_path: str) -> None:
    with open(save_path, 'wb') as f:
        f.write(requests.get(gif_url).content)


def extract_frames(gif_path: str, num_frames: int) -> list[Image.Image]:
    
    gif = Image.open(gif_path)
    step = gif.n_frames // (num_frames - 1) # take bigger step to include first and last frames
    frames = []
   
    for i in range(num_frames):
        gif.seek(min(i*step, gif.n_frames - 1)) # prevent i*step == gif.n_frames because IndexOutOfBounds
        frame = copy(gif)
        frame = frame.resize((224, 224))
        frame = frame.convert('RGB')
        frames.append(frame)
    
    return frames


def save_frames(frames: list[Image.Image], save_path: str) -> None:
    _, axs = plt.subplots(nrows=1, ncols=len(frames), figsize=(15,3))
    for i, frame in enumerate(frames):
        axs[i].imshow(frame)
        axs[i].axis('off')
    plt.savefig(save_path, bbox_inches='tight')
