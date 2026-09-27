"""The ONE preprocessing function, shared by training, evaluation and the service.

If anything in here changes, bump PREPROCESS_VERSION. It is stored with every
prediction so we can tell which results were produced by which pipeline.
"""
import torch
from PIL import Image
from torchvision.transforms import functional as F

PREPROCESS_VERSION = "v1-rgb-resize224-imagenet-norm"

INPUT_SIZE = 224  # ResNet18 was pretrained on 224x224 ImageNet images
# ImageNet channel statistics: the pretrained backbone expects inputs normalised this way.
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def preprocess(img: Image.Image) -> torch.Tensor:
    """PIL image -> float tensor of shape (3, 224, 224), ready for the model."""
    img = img.convert("RGB")
    img = F.resize(img, [INPUT_SIZE, INPUT_SIZE], interpolation=F.InterpolationMode.BILINEAR)
    tensor = F.to_tensor(img)  # uint8 [0, 255] -> float [0, 1], shape (C, H, W)
    return F.normalize(tensor, IMAGENET_MEAN, IMAGENET_STD)
