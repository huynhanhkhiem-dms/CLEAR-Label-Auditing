#!/usr/bin/env python3
"""Build frozen ImageNet ResNet-18 embeddings for the CIFAR-10 training set.

The output is a (50000, 512) float32 NumPy array in the official torchvision
CIFAR-10 training order. The classifier head is replaced by Identity and the
preprocessing transform bundled with ResNet18_Weights.IMAGENET1K_V1 is used.
"""
from __future__ import annotations
import argparse, hashlib, json, platform
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision.models import resnet18, ResNet18_Weights


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="CIFAR-10N run directory")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    root = Path(args.root).expanduser().resolve()
    features = root / "features"
    cache = root / "torchvision_data"
    features.mkdir(parents=True, exist_ok=True)
    cache.mkdir(parents=True, exist_ok=True)

    weights = ResNet18_Weights.IMAGENET1K_V1
    ds = CIFAR10(root=str(cache), train=True, download=True, transform=weights.transforms())
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.workers, pin_memory=args.device.startswith("cuda"))
    model = resnet18(weights=weights)
    model.fc = nn.Identity()
    model.eval().to(args.device)

    chunks = []
    with torch.inference_mode():
        for images, _ in loader:
            chunks.append(model(images.to(args.device, non_blocking=True)).cpu().numpy().astype(np.float32))
    arr = np.concatenate(chunks, axis=0)
    if arr.shape != (50000, 512):
        raise RuntimeError(f"Unexpected embedding shape: {arr.shape}")
    out = features / "resnet18_train_embeddings.npy"
    np.save(out, arr)
    meta = {
        "model": "torchvision.models.resnet18",
        "weights": "IMAGENET1K_V1",
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
        "sha256": sha256(out),
        "torch_version": torch.__version__,
        "torchvision_preprocessing": "ResNet18_Weights.IMAGENET1K_V1.transforms()",
        "device": args.device,
        "platform": platform.platform(),
        "note": "Floating-point hashes can differ across software/hardware stacks; shape, ordering, model weights and preprocessing define the intended representation."
    }
    (features / "embedding_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))

if __name__ == "__main__":
    main()
