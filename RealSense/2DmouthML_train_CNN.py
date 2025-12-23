# -----------------------------------------
# 機能（soundML_train_CNN.py と同じ構成）:
#  1) 教師データ画像（フォルダ配下）を読み込む
#  2) CNNで学習（train/test を固定比率で分割）
#  3) model.pt（state_dict）, meta.json, metrics.json, learning_curve を保存
# -----------------------------------------

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# ====== 設定（必要ならここだけ編集） ======
CONFIG = {
    # データセット（label別フォルダ）
    # 例:
    #   ML2D/2type/U/*.png
    #   ML2D/2type/NotU/*.png
    "DATASET_ROOT": "ML2D/2type",

    # 「う」とみなすフォルダ名（親フォルダ名）
    "POSITIVE_DIR_NAMES": {"U", "u", "う"},

    # 学習に使うビュー（ファイル名末尾が "__XY.png" のようになっている想定）
    # None なら全pngを使用
    "USE_VIEWS": ["XY", "ZY"],   # 例: ["XY","ZY"] / ["XY","ZY","XZ"] / None

    # 入力サイズ
    "IMG_H": 224,
    "IMG_W": 224,

    # 画像の扱い
    # PLY由来のscatter画像がRGBなら False 推奨
    "GRAYSCALE": False,

    # 学習設定
    "RANDOM_STATE": 42,
    "TEST_SIZE": 0.2,
    "BATCH_SIZE": 32,
    "EPOCHS": 60,
    "LR": 1e-3,

    # 保存先
    "MODEL_DIR": "ML2D/trained_cnn_model_u_notu",
}
# =========================================


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _view_filter_ok(path: Path, use_views: Optional[List[str]]) -> bool:
    if use_views is None:
        return True
    name = path.name
    # 例: stem__XY.png
    for v in use_views:
        if name.endswith(f"__{v}.png"):
            return True
    return False


def load_image_as_tensor(img_path: Path, img_h: int, img_w: int, grayscale: bool) -> torch.Tensor:
    """
    1枚PNG -> Tensor(C, H, W) / 0..1
    学習・推論で必ず同じ前処理にする。
    """
    img = Image.open(img_path)
    img = img.convert("L") if grayscale else img.convert("RGB")
    img = img.resize((img_w, img_h), resample=Image.BILINEAR)

    arr = np.asarray(img, dtype=np.float32) / 255.0
    if grayscale:
        arr = arr[None, :, :]          # (1,H,W)
    else:
        arr = arr.transpose(2, 0, 1)   # (3,H,W)
    return torch.from_numpy(arr)


@dataclass
class Sample:
    path: Path
    label_idx: int  # 0=NotU, 1=U


class ImageFolderDataset(Dataset):
    def __init__(self, samples: List[Sample], img_h: int, img_w: int, grayscale: bool):
        self.samples = samples
        self.img_h = img_h
        self.img_w = img_w
        self.grayscale = grayscale

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        x = load_image_as_tensor(s.path, self.img_h, self.img_w, self.grayscale)
        y = torch.tensor(s.label_idx, dtype=torch.long)
        return x, y


def collect_samples_binary(dataset_root: Path, positive_names: set[str], use_views: Optional[List[str]]) -> List[Sample]:
    """
    dataset_root 配下の png を再帰探索し、
      親フォルダ名が positive_names に含まれれば label=1(U)
      それ以外は label=0(NotU)
    """
    if not dataset_root.exists():
        raise FileNotFoundError(f"dataset_root not found: {dataset_root}")

    pngs = sorted(dataset_root.rglob("*.png"))
    if use_views is not None:
        pngs = [p for p in pngs if _view_filter_ok(p, use_views)]

    if not pngs:
        raise ValueError("No images found. Check dataset_root and extension (*.png).")

    samples: List[Sample] = []
    for p in pngs:
        parent = p.parent.name
        label = 1 if parent in positive_names else 0
        samples.append(Sample(path=p, label_idx=label))

    n_pos = sum(s.label_idx == 1 for s in samples)
    n_neg = len(samples) - n_pos
    if n_pos == 0 or n_neg == 0:
        raise ValueError(f"Need both classes. pos={n_pos}, neg={n_neg}. Check folder names vs POSITIVE_DIR_NAMES.")
    return samples


class SimpleCNN(nn.Module):
    """
    小さめCNN（入力: (C, IMG_H, IMG_W)）
    出力は logits（softmaxしない）
    """
    def __init__(self, in_ch: int, n_classes: int, img_h: int, img_w: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_ch, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

        with torch.no_grad():
            dummy = torch.zeros(1, in_ch, img_h, img_w)
            feat = self.features(dummy)
            feat_dim = int(np.prod(feat.shape[1:]))

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(feat_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(256, n_classes),  # n_classes=2（NotU/U）
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x  # logits


def train_one_epoch(model, loader, optimizer, criterion, device) -> float:
    model.train()
    losses = []
    for x, y in loader:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
    return float(np.mean(losses)) if losses else 0.0


@torch.no_grad()
def eval_accuracy(model, loader, device) -> float:
    model.eval()
    ys = []
    yhats = []
    for x, y in loader:
        x = x.to(device)
        logits = model(x)
        pred = torch.argmax(logits, dim=1).cpu().numpy()
        ys.append(y.numpy())
        yhats.append(pred)
    y_true = np.concatenate(ys, axis=0)
    y_pred = np.concatenate(yhats, axis=0)
    return float(accuracy_score(y_true, y_pred))


@torch.no_grad()
def eval_model(model, loader, device) -> Tuple[float, np.ndarray, np.ndarray]:
    model.eval()
    ys = []
    yhats = []
    for x, y in loader:
        x = x.to(device)
        logits = model(x)
        pred = torch.argmax(logits, dim=1).cpu().numpy()
        ys.append(y.numpy())
        yhats.append(pred)
    y_true = np.concatenate(ys, axis=0)
    y_pred = np.concatenate(yhats, axis=0)
    acc = float(accuracy_score(y_true, y_pred))
    return acc, y_true, y_pred


def plot_learning_curve(epochs, train_acc, val_acc, out_path: Path) -> None:
    plt.figure()
    plt.plot(epochs, train_acc, label="train_acc")
    plt.plot(epochs, val_acc, label="val_acc")
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", type=str, default=CONFIG["DATASET_ROOT"])
    parser.add_argument("--model_dir", type=str, default=CONFIG["MODEL_DIR"])
    args = parser.parse_args()

    set_seed(int(CONFIG["RANDOM_STATE"]))

    dataset_root = Path(args.dataset_root)
    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    img_h = int(CONFIG["IMG_H"])
    img_w = int(CONFIG["IMG_W"])
    grayscale = bool(CONFIG["GRAYSCALE"])
    in_ch = 1 if grayscale else 3

    # ラベルは固定（2値）
    label_names = ["NotU", "U"]
    n_classes = 2

    samples = collect_samples_binary(
        dataset_root=dataset_root,
        positive_names=set(CONFIG["POSITIVE_DIR_NAMES"]),
        use_views=CONFIG["USE_VIEWS"],
    )

    y_all = np.array([s.label_idx for s in samples], dtype=int)
    idx_all = np.arange(len(samples))

    idx_train, idx_test = train_test_split(
        idx_all,
        test_size=float(CONFIG["TEST_SIZE"]),
        random_state=int(CONFIG["RANDOM_STATE"]),
        stratify=y_all
    )

    train_samples = [samples[i] for i in idx_train]
    test_samples = [samples[i] for i in idx_test]

    train_loader = DataLoader(
        ImageFolderDataset(train_samples, img_h, img_w, grayscale),
        batch_size=int(CONFIG["BATCH_SIZE"]),
        shuffle=True
    )
    test_loader = DataLoader(
        ImageFolderDataset(test_samples, img_h, img_w, grayscale),
        batch_size=int(CONFIG["BATCH_SIZE"]),
        shuffle=False
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = SimpleCNN(in_ch=in_ch, n_classes=n_classes, img_h=img_h, img_w=img_w).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(CONFIG["LR"]))
    criterion = nn.CrossEntropyLoss()  # 2クラスでもCrossEntropyでOK

    train_eval_loader = DataLoader(
        ImageFolderDataset(train_samples, img_h, img_w, grayscale),
        batch_size=int(CONFIG["BATCH_SIZE"]),
        shuffle=False
    )

    epochs_hist = []
    train_acc_hist = []
    val_acc_hist = []
    train_loss_hist = []

    for epoch in range(1, int(CONFIG["EPOCHS"]) + 1):
        tr_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)

        tr_acc = eval_accuracy(model, train_eval_loader, device)
        va_acc = eval_accuracy(model, test_loader, device)

        epochs_hist.append(epoch)
        train_loss_hist.append(tr_loss)
        train_acc_hist.append(tr_acc)
        val_acc_hist.append(va_acc)

        print(f"[epoch {epoch:03d}/{int(CONFIG['EPOCHS'])}] loss={tr_loss:.4f} train_acc={tr_acc:.4f} val_acc={va_acc:.4f}")

    plot_learning_curve(
        epochs_hist,
        train_acc_hist,
        val_acc_hist,
        model_dir / "learning_curve.png"
    )

    with open(model_dir / "learning_curve.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "epochs": epochs_hist,
                "train_acc": train_acc_hist,
                "val_acc": val_acc_hist,
                "train_loss": train_loss_hist,
            },
            f, ensure_ascii=False, indent=2
        )

    acc, y_true, y_pred = eval_model(model, test_loader, device)

    cm = confusion_matrix(y_true, y_pred).tolist()
    report = classification_report(y_true, y_pred, target_names=label_names)

    metrics = {
        "accuracy": acc,
        "labels": label_names,
        "confusion_matrix": cm,
        "classification_report": report,
        "n_samples": int(len(samples)),
        "n_train": int(len(train_samples)),
        "n_test": int(len(test_samples)),
        "img_h": img_h,
        "img_w": img_w,
        "grayscale": grayscale,
        "test_size": float(CONFIG["TEST_SIZE"]),
        "random_state": int(CONFIG["RANDOM_STATE"]),
        "epochs": int(CONFIG["EPOCHS"]),
        "batch_size": int(CONFIG["BATCH_SIZE"]),
        "lr": float(CONFIG["LR"]),
        "device": str(device),
        "use_views": CONFIG["USE_VIEWS"],
        "positive_dir_names": sorted(list(CONFIG["POSITIVE_DIR_NAMES"])),
    }

    ckpt = {
        "model_state_dict": model.state_dict(),
        "label_names": label_names,
        "img_h": img_h,
        "img_w": img_w,
        "grayscale": grayscale,
        "use_views": CONFIG["USE_VIEWS"],
        "positive_dir_names": sorted(list(CONFIG["POSITIVE_DIR_NAMES"])),
    }
    torch.save(ckpt, model_dir / "model.pt")

    with open(model_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "label_names": label_names,
                "img_h": img_h,
                "img_w": img_w,
                "grayscale": grayscale,
                "use_views": CONFIG["USE_VIEWS"],
                "positive_dir_names": sorted(list(CONFIG["POSITIVE_DIR_NAMES"])),
            },
            f, ensure_ascii=False, indent=2
        )

    with open(model_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print("=== TRAIN/EVAL DONE (MOUTH 2D CNN, U vs NotU) ===")
    print("model_dir:", model_dir.resolve())
    print("accuracy :", metrics["accuracy"])
    print("labels   :", metrics["labels"])
    print("confusion_matrix:", metrics["confusion_matrix"])
    print(metrics["classification_report"])


if __name__ == "__main__":
    main()
