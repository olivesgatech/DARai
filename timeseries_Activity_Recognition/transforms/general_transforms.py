import numpy as np
import torch

class Normalize:
    """
    Normalizes based on mean and std. Used by skeleton and inertial modalities
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, x):
        return (x - self.mean) / self.std

class ToTensor:
    def __call__(self, x):
        return torch.tensor(x)

class Permute:
    def __init__(self, shape):
        self.shape = shape

    def __call__(self, x):
        return x.permute(self.shape)

class ToFloat:
    def __call__(self, x):
        return x.float()
    
class scale:
    def __call__(self, x):
        min_vals = np.min(x, axis=0, keepdims=True)
        max_vals = np.max(x, axis=0, keepdims=True)

        # Avoid division by zero by setting zero ranges to 1
        ranges = max_vals - min_vals
        ranges[ranges == 0] = 1
        # Normalize to [-1, 1]
        x_normalized = 2 * ((x - min_vals) / ranges) - 1
        return x_normalized