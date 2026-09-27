"""Fine-tune an ImageNet-pretrained ResNet18 on the labelled candidate tiles (CPU).

Usage (from the project root):  python -m scripts.train

- Split: candidate_tiles -> 80% train / 20% val, stratified per class, fixed seed.
  eval_set (with eval_labels.csv) is the held-out test set; see scripts/evaluate.py.
- Trained: layer4 (last residual block) + fc (new 7-class head). Everything else frozen.
- Output: models/resnet18_eurosat.pt + models/manifest.json (classes, SHA-256, versions).

This is the only step that touches the network: torchvision downloads the ImageNet
weights once (~45 MB, cached in ~/.cache/torch). The service only loads our own .pt file.
"""
import json
import random
from datetime import datetime, timezone
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.models import ResNet18_Weights, resnet18
from torchvision.transforms import functional as F

from app.model import sha256_file
from app.preprocess import PREPROCESS_VERSION, preprocess

SEED = 42
DATA_DIR = Path("data/raw/candidate_tiles")
MODELS_DIR = Path("models")
WEIGHTS_FILE = "resnet18_eurosat.pt"
VAL_FRACTION = 0.2
EPOCHS = 5
BATCH_SIZE = 32


def stratified_split(data_dir: Path):
    """Return (train, val, classes). Each item is (path, class_index)."""
    classes = sorted(p.name for p in data_dir.iterdir() if p.is_dir())  # alphabetical = stable order
    rng = random.Random(SEED)
    train, val = [], []
    for idx, name in enumerate(classes):
        files = sorted((data_dir / name).glob("*.png"))  # sort first so the shuffle is reproducible
        rng.shuffle(files)
        n_val = int(len(files) * VAL_FRACTION)
        val += [(f, idx) for f in files[:n_val]]
        train += [(f, idx) for f in files[n_val:]]
    return train, val, classes


class TileDataset(Dataset):
    def __init__(self, items, augment: bool):
        self.items = items
        self.augment = augment

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        path, label = self.items[i]
        img = Image.open(path).convert("RGB")
        if self.augment:
            # Overhead imagery has no "up", so flips give free, realistic extra examples.
            # Applied BEFORE preprocess(), so the preprocessing itself stays identical to serving.
            if random.random() < 0.5:
                img = F.hflip(img)
            if random.random() < 0.5:
                img = F.vflip(img)
        return preprocess(img), label


def evaluate_accuracy(model, loader) -> float:
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            correct += (model(x).argmax(dim=1) == y).sum().item()
            total += len(y)
    return correct / total


def main():
    random.seed(SEED)
    torch.manual_seed(SEED)

    train_items, val_items, classes = stratified_split(DATA_DIR)
    print(f"classes={classes} train={len(train_items)} val={len(val_items)}")

    train_loader = DataLoader(TileDataset(train_items, augment=True), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TileDataset(val_items, augment=False), batch_size=BATCH_SIZE)

    # Pretrained backbone, new 7-class head.
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    model.fc = torch.nn.Linear(model.fc.in_features, len(classes))

    # Freeze everything, then unfreeze only the last block and the head.
    for p in model.parameters():
        p.requires_grad = False
    for p in list(model.layer4.parameters()) + list(model.fc.parameters()):
        p.requires_grad = True

    # New head learns fast; the pretrained block gets a smaller learning rate so it adapts gently.
    optimizer = torch.optim.Adam([
        {"params": model.fc.parameters(), "lr": 1e-3},
        {"params": model.layer4.parameters(), "lr": 1e-4},
    ])
    loss_fn = torch.nn.CrossEntropyLoss()

    MODELS_DIR.mkdir(exist_ok=True)
    weights_path = MODELS_DIR / WEIGHTS_FILE
    best_val_acc = -1.0

    for epoch in range(1, EPOCHS + 1):
        # eval() everywhere keeps the frozen layers' BatchNorm stats at their ImageNet values;
        # only layer4 (which we are training) goes into train mode.
        model.eval()
        model.layer4.train()
        running_loss = 0.0
        for x, y in train_loader:
            optimizer.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * len(y)

        val_acc = evaluate_accuracy(model, val_loader)
        print(f"epoch {epoch}: train_loss={running_loss / len(train_items):.4f} val_acc={val_acc:.4f}")

        # Keep the epoch with the best validation accuracy.
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), weights_path)

    manifest = {
        "model_name": "resnet18-eurosat",
        "model_version": datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S"),
        "weights_file": WEIGHTS_FILE,
        "weights_sha256": sha256_file(weights_path),
        "classes": classes,
        "preprocess_version": PREPROCESS_VERSION,
        "seed": SEED,
        "epochs": EPOCHS,
        "train_size": len(train_items),
        "val_size": len(val_items),
        "best_val_accuracy": round(best_val_acc, 4),
    }
    (MODELS_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
