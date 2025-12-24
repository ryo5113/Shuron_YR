from __future__ import annotations

import os
from pathlib import Path
from typing import List

import numpy as np
import torch
import torch.nn as nn

# =====================
# User settings (edit)
# =====================
CONFIG = {
    # Path to checkpoint produced by the training script
    "checkpoint_path": "PLY/ML3D/2type/best_model.pth",

    # A single .npz file path OR a directory that contains .npz files
    "input_path": "PLY/ml/mouth_voxel64_rgb/NotU.npz",

    # If input_path is a directory, whether to search recursively
    "recursive": True,

    # Threshold on sigmoid(logit) to decide 'u'
    "threshold": 0.7,

    # Device: "cuda" or "cpu" ("cuda" will be used if available)
    "device": "cuda",
}


class Small3DCNN(nn.Module):
    """Same architecture as in training script (must match)."""

    def __init__(self, in_ch: int = 3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv3d(in_ch, 16, kernel_size=3, padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2),

            nn.Conv3d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2),

            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2),

            nn.Conv3d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool3d((1, 1, 1)),
        )
        self.classifier = nn.Linear(128, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.flatten(1)
        return self.classifier(x).squeeze(1)  # (N,)


def load_voxel_npz(path: Path) -> np.ndarray:
    data = np.load(path, allow_pickle=False)
    if "voxel" not in data:
        raise KeyError(f"{path} does not contain key 'voxel'")
    v = data["voxel"]
    if v.shape != (3, 64, 64, 64):
        raise ValueError(f"Unexpected voxel shape {v.shape} in {path} (expected (3,64,64,64))")
    if v.dtype != np.float32:
        v = v.astype(np.float32)
    return v


def collect_npz_files(input_path: Path, recursive: bool) -> List[Path]:
    if input_path.is_file() and input_path.suffix.lower() == ".npz":
        return [input_path]
    if input_path.is_dir():
        pattern = "**/*.npz" if recursive else "*.npz"
        return sorted(input_path.glob(pattern))
    raise FileNotFoundError(f"input_path not found: {input_path}")


def main() -> None:
    ckpt_path = Path(CONFIG["checkpoint_path"])
    input_path = Path(CONFIG["input_path"])
    recursive = bool(CONFIG["recursive"])
    threshold = float(CONFIG["threshold"])

    device = CONFIG["device"]
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    if not ckpt_path.exists():
        raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")

    # Build model + load weights
    model = Small3DCNN(in_ch=3)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    files = collect_npz_files(input_path, recursive)
    if not files:
        raise FileNotFoundError(f"No .npz files found under: {input_path}")

    print(f"checkpoint: {ckpt_path}")
    print(f"device: {device}")
    print(f"num_files: {len(files)}\n")

    with torch.no_grad():
        for p in files:
            v = load_voxel_npz(p)
            x = torch.from_numpy(v).unsqueeze(0).to(device)  # (1,3,64,64,64)
            logit = model(x)
            prob_u = torch.sigmoid(logit).item()
            pred = "u" if prob_u >= threshold else "not_u"
            print(f"{p} -> prob_u={prob_u:.4f} pred={pred}")


if __name__ == "__main__":
    main()
