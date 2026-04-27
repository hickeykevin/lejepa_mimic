import torch
import numpy as np
import torchvision.transforms as transforms
from PIL import Image
from typing import Any, List, Optional, Tuple

class MultiViewTransform:
    """
    Generates V randomly augmented views of the same image.
    Enables the standard LeJEPA symmetric invariance objective.
    """
    def __init__(self, img_size: int = 224, V: int = 4):
        self.V = V
        self.aug = transforms.Compose([
            transforms.RandomResizedCrop(img_size, scale=(0.2, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __call__(self, img) -> torch.Tensor:
        # Apply augmentation V times independently → [V, C, H, W]
        return torch.stack([self.aug(img) for _ in range(self.V)])

    def get_collate_fn(self):
        """Standard stack collation is fine for MultiView as views are same size."""
        return None


class RawSourceTransform:
    """
    Transform for models that perform their own internal cropping (e.g. DINOLeJEPA).
    Resizes the image to a fixed square size (default 512x512) so they can be 
    stacked into a single tensor in the Universal DataModule.
    """
    def __init__(self, size: int = 512):
        self.resize = transforms.Resize((size, size))
        self.to_tensor = transforms.ToTensor()

    def __call__(self, img) -> torch.Tensor:
        img = self.resize(img)
        return self.to_tensor(img)

    def get_collate_fn(self):
        """
        Since RawSource output depends on aspect ratio, images have different shapes.
        We must use the list-based RawImageCollate.
        """
        from .dataset import RawImageCollate
        return RawImageCollate(list_keys=["image"])


def get_cxr_spatial_prior(img_size: int = 224) -> torch.Tensor:
    """
    Returns a [1, H, W] importance map emphasizing the central lungs and हृदय.
    Used for certain JEPA masking strategies.
    """
    x = torch.linspace(-1, 1, img_size)
    y = torch.linspace(-1, 1, img_size)
    grid_y, grid_x = torch.meshgrid(y, x, indexing='ij')
    
    # Gaussian centered loosely on the chest
    prior = torch.exp(-(grid_x**2 / 0.5 + (grid_y + 0.1)**2 / 0.7))
    return prior.unsqueeze(0)
