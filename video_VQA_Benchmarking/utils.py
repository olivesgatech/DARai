import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import random
import numpy as np
import torch
import os


def set_seed(seed: int):
    """
    Sets the seed for reproducibility across random, numpy, and PyTorch.

    Args:
        seed (int): The seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    # For additional reproducibility in PyTorch
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def display_frames_grid(frames, title="Video Sample Frames" , fps = 16):
    """
    Displays the selected video frames in a grid (4x4).

    Args:
        frames (np.ndarray): Array of frames with shape (T, H, W, C).
        title (str): Title of the grid.
    """
    num_frames = frames.shape[0]
    grid_size = int(np.ceil(np.sqrt(num_frames)))  # Approximate square grid

    fig, axes = plt.subplots(grid_size, grid_size, figsize=(10, 10))
    fig.suptitle(title, fontsize=16)

    for i, ax in enumerate(axes.flat):
        if i < num_frames:
            ax.imshow(frames[i])
            ax.axis('off')
        else:
            ax.set_visible(False)

    plt.show()

def create_gif_from_frames(frames,title, fps=1):
    """
    Returns:
        PIL.Image: Animated GIF image.
    """
    pil_images = [Image.fromarray(frame) for frame in frames]
    gif_path = f"./{title}.gif"
    pil_images[0].save(gif_path, save_all=True, append_images=pil_images[1:], duration=int(1000/fps), loop=0)
    return gif_path

