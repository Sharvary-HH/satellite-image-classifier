"""Model loading and inference.

Anything with a `predict(batch) -> probabilities` method and a `classes` list
counts as a Classifier, so tests can swap in a fake one without loading torch weights.
"""
import hashlib
import json
from pathlib import Path
from typing import Protocol

import numpy as np
import torch
from torchvision.models import resnet18


class Classifier(Protocol):
    classes: list[str]
    # Provenance, stored with every prediction.
    model_name: str
    model_version: str
    weights_sha: str
    preprocess_version: str

    def predict(self, batch: torch.Tensor) -> np.ndarray:
        """(N, 3, H, W) tensor -> (N, num_classes) array of probabilities summing to 1."""
        ...


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_resnet18(num_classes: int) -> torch.nn.Module:
    """ResNet18 with the 1000-way ImageNet head replaced by a num_classes head.

    weights=None: no download. Real weights are loaded from our own file afterwards.
    """
    model = resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    return model


class ResNetClassifier:
    """Fine-tuned ResNet18 loaded from models/manifest.json. Checksum is verified on load."""

    def __init__(self, manifest_path: Path):
        manifest = json.loads(manifest_path.read_text())
        weights_path = manifest_path.parent / manifest["weights_file"]

        # Refuse to start with weights that don't match what training produced.
        actual_sha = sha256_file(weights_path)
        if actual_sha != manifest["weights_sha256"]:
            raise RuntimeError(
                f"Weights checksum mismatch for {weights_path}: "
                f"expected {manifest['weights_sha256']}, got {actual_sha}"
            )

        self.manifest = manifest
        self.classes = manifest["classes"]  # index i in the output == classes[i]
        self.model_name = manifest["model_name"]
        self.model_version = manifest["model_version"]
        self.weights_sha = actual_sha
        self.preprocess_version = manifest["preprocess_version"]

        self.model = build_resnet18(len(self.classes))
        self.model.load_state_dict(torch.load(weights_path, map_location="cpu", weights_only=True))
        self.model.eval()  # inference mode: BatchNorm uses stored stats, no dropout

    @torch.no_grad()
    def predict(self, batch: torch.Tensor) -> np.ndarray:
        logits = self.model(batch)
        return torch.softmax(logits, dim=1).numpy()
