import numpy as np
import pytest
import torch
from PIL import Image

# Ensure backward compatibility for NumPy 2.0 with older dependencies (e.g., wtfml)
if not hasattr(np, "Inf"):
    np.Inf = np.inf
if not hasattr(np, "PINF"):
    np.PINF = np.inf
if not hasattr(np, "NINF"):
    np.NINF = -np.inf


@pytest.fixture(autouse=True)
def set_random_seed():
    np.random.seed(42)
    torch.manual_seed(42)


@pytest.fixture
def dummy_image_tensor():
    # Batch size 2, 3 color channels, 224x224
    return torch.randn(2, 3, 224, 224)


@pytest.fixture
def dummy_targets_tensor():
    # Target binary labels for batch size 2
    return torch.tensor([0.0, 1.0])


@pytest.fixture
def dummy_pil_image():
    # Create a simple RGB PIL image
    img_array = np.uint8(np.random.rand(224, 224, 3) * 255)
    return Image.fromarray(img_array)
