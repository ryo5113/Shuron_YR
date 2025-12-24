from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# =====================
# User settings (edit)
# =====================
CONFIG = {
    # Dataset root. Example structure:
    #   mouth_voxel64_rgb_dataset/
    #     u/*.npz
    #     a/*.npz
    #     i/*.npz
    #     e/*.npz
    #     o/*.npz
    # or:
    #     u/*.npz
    #     not_u/*.npz
    "DATA_ROOT": "PLY/ml/mouth_voxel64_rgb",

    # Parent directory names treated as positive class ('u')
    "POSITIVE_DIR_NAMES": {"u", "U", "う"},

    # Output
    "OUT_DIR": "PLY/ML3D",

    # Training
    "EPOCHS": 30,
    "BATCH_SIZE": 8,
    "LEARNING_RATE": 1e-3,
    "WEIGHT_DECAY": 1e-4,
    "VAL_RATIO": 0.3,
    "SEED": 42,

    # Hardware
    "DEVICE": "cuda" if torch.cuda.is_available() else "cpu",

    # Data
    "NUM_WORKERS": 0,
    "PIN_MEMORY": True,
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@dataclass
class Sample:
    path: Path
    label: int


def collect_samples(data_root: Path, positive_names: set[str]) -> List[Sample]:
    if not data_root.exists():
        raise FileNotFoundError(f"DATA_ROOT not found: {data_root}")

    files = sorted(data_root.rglob("*.npz"))
    if not files:
        raise FileNotFoundError(f"No .npz files found under: {data_root}")

    samples: List[Sample] = []
    for fp in files:
        parent = fp.parent.name
        label = 1 if parent in positive_names else 0
        samples.append(Sample(path=fp, label=label))

    # sanity: need both classes
    n_pos = sum(s.label == 1 for s in samples)
    n_neg = len(samples) - n_pos
    if n_pos == 0 or n_neg == 0:
        raise ValueError(
            f"Need both positive and negative samples. Found pos={n_pos}, neg={n_neg}. "
            f"Check folder names vs POSITIVE_DIR_NAMES={sorted(list(positive_names))}"
        )
    return samples


def stratified_split(samples: List[Sample], val_ratio: float, seed: int) -> Tuple[List[Sample], List[Sample]]:
    pos = [s for s in samples if s.label == 1]
    neg = [s for s in samples if s.label == 0]

    rng = random.Random(seed)
    rng.shuffle(pos)
    rng.shuffle(neg)

    n_pos_val = max(1, int(round(len(pos) * val_ratio)))
    n_neg_val = max(1, int(round(len(neg) * val_ratio)))

    val = pos[:n_pos_val] + neg[:n_neg_val]
    train = pos[n_pos_val:] + neg[n_neg_val:]

    rng.shuffle(train)
    rng.shuffle(val)
    return train, val


class VoxelNPZDataset(Dataset):
    def __init__(self, samples: List[Sample]):
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        with np.load(s.path) as z:
            v = z["voxel"]  # (3,64,64,64)

        # Ensure float32
        if v.dtype != np.float32:
            v = v.astype(np.float32)

        # Convert to torch (C,D,H,W)
        x = torch.from_numpy(v)

        # Optional: normalize to [0,1] already; keep as-is
        y = torch.tensor([float(s.label)], dtype=torch.float32)  # BCE expects float targets
        return x, y, str(s.path)


class Small3DCNN(nn.Module):
    """A compact 3D CNN for (3,64,64,64) input -> 1 logit."""

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv3d(3, 16, kernel_size=3, padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2),  # 32

            nn.Conv3d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2),  # 16

            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2),  # 8

            nn.Conv3d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool3d((1, 1, 1)),
        )
        self.classifier = nn.Linear(128, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N,3,64,64,64)
        f = self.features(x).flatten(1)
        logit = self.classifier(f)  # (N,1)
        return logit


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: str) -> Tuple[float, float]:
    model.eval()
    total = 0
    correct = 0
    total_loss = 0.0
    criterion = nn.BCEWithLogitsLoss()

    for x, y, _ in loader:
        x = x.to(device)
        y = y.to(device)  # (N,1)
        logit = model(x)
        loss = criterion(logit, y)

        prob = torch.sigmoid(logit)
        pred = (prob >= 0.5).float()

        total += y.numel()
        correct += (pred == y).sum().item()
        total_loss += loss.item() * y.size(0)

    acc = correct / max(1, total)
    avg_loss = total_loss / max(1, len(loader.dataset))
    return avg_loss, acc


def main() -> None:
    set_seed(int(CONFIG["SEED"]))

    data_root = Path(CONFIG["DATA_ROOT"]).expanduser()
    out_dir = Path(CONFIG["OUT_DIR"]).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    samples = collect_samples(data_root, set(CONFIG["POSITIVE_DIR_NAMES"]))
    train_s, val_s = stratified_split(samples, float(CONFIG["VAL_RATIO"]), int(CONFIG["SEED"]))

    # Save label rule metadata
    meta = {
        "task": "binary_u_vs_not_u",
        "positive_dir_names": sorted(list(CONFIG["POSITIVE_DIR_NAMES"])),
        "data_root": str(data_root),
        "input_shape": [3, 64, 64, 64],
        "split": {
            "train": len(train_s),
            "val": len(val_s),
        },
        "config": {k: (sorted(list(v)) if isinstance(v, set) else v) for k, v in CONFIG.items()},
    }
    (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    train_ds = VoxelNPZDataset(train_s)
    val_ds = VoxelNPZDataset(val_s)

    train_loader = DataLoader(
        train_ds,
        batch_size=int(CONFIG["BATCH_SIZE"]),
        shuffle=True,
        num_workers=int(CONFIG["NUM_WORKERS"]),
        pin_memory=bool(CONFIG["PIN_MEMORY"]),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(CONFIG["BATCH_SIZE"]),
        shuffle=False,
        num_workers=int(CONFIG["NUM_WORKERS"]),
        pin_memory=bool(CONFIG["PIN_MEMORY"]),
    )

    device = str(CONFIG["DEVICE"])
    model = Small3DCNN().to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(CONFIG["LEARNING_RATE"]),
        weight_decay=float(CONFIG["WEIGHT_DECAY"]),
    )

    best_val_acc = -1.0

    log_path = out_dir / "train_log.csv"
    if not log_path.exists():
        log_path.write_text("epoch,train_loss,val_loss,val_acc\n", encoding="utf-8")

    for epoch in range(1, int(CONFIG["EPOCHS"]) + 1):
        model.train()
        running = 0.0

        for x, y, _ in train_loader:
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad(set_to_none=True)
            logit = model(x)
            loss = criterion(logit, y)
            loss.backward()
            optimizer.step()

            running += loss.item() * y.size(0)

        train_loss = running / max(1, len(train_loader.dataset))
        val_loss, val_acc = evaluate(model, val_loader, device)

        with log_path.open("a", encoding="utf-8") as f:
            f.write(f"{epoch},{train_loss:.6f},{val_loss:.6f},{val_acc:.6f}\n")

        # Save last
        torch.save({
            "model_state_dict": model.state_dict(),
            "config": {k: (sorted(list(v)) if isinstance(v, set) else v) for k, v in CONFIG.items()},
        }, out_dir / "last_model.pth")

        # Save best
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "model_state_dict": model.state_dict(),
                "best_val_acc": best_val_acc,
                "config": {k: (sorted(list(v)) if isinstance(v, set) else v) for k, v in CONFIG.items()},
            }, out_dir / "best_model.pth")

        print(f"[Epoch {epoch:03d}] train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.3f} best={best_val_acc:.3f}")

    print(f"Done. Best val_acc={best_val_acc:.3f}  (saved: {out_dir/'best_model.pth'})")


if __name__ == "__main__":
    main()
