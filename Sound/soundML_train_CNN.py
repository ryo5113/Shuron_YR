# -----------------------------------------
# 機能:
#  1) 教師データ画像（ラベル別フォルダ）を読み込む
#  2) CNNで学習（train/test を固定比率で分割）
#  3) model.pt（state_dict）, meta.json, metrics.json を保存
# -----------------------------------------

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report  # 評価出力 :contentReference[oaicite:4]{index=4}


# ====== 設定（必要ならここだけ編集） ======
IMG_H = 64
IMG_W = 256
GRAYSCALE = True

RANDOM_STATE = 42
TEST_SIZE = 0.3          # 7:3 など（あなたの運用に合わせる）
BATCH_SIZE = 32
EPOCHS = 50
LR = 1e-3
# =========================================


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_image_as_tensor(img_path: Path) -> torch.Tensor:
    """
    [機能] 1枚PNG -> Tensor(1, H, W) / 0..1
    学習・推論で必ず同じ前処理にする。
    """
    img = Image.open(img_path)
    img = img.convert("L") if GRAYSCALE else img.convert("RGB")
    # torchvision.transforms.Resize 相当の固定リサイズ（H,W）:contentReference[oaicite:5]{index=5}
    img = img.resize((IMG_W, IMG_H), resample=Image.BILINEAR)

    arr = np.asarray(img, dtype=np.float32) / 255.0
    if GRAYSCALE:
        arr = arr[None, :, :]  # (1,H,W)
    else:
        arr = arr.transpose(2, 0, 1)  # (C,H,W)
    return torch.from_numpy(arr)


@dataclass
class Sample:
    path: Path
    label_idx: int


class ImageFolderDataset(Dataset):
    def __init__(self, samples: List[Sample]):
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        x = load_image_as_tensor(s.path)      # (C,H,W)
        y = torch.tensor(s.label_idx, dtype=torch.long)
        return x, y


def list_samples(dataset_root: Path) -> Tuple[List[Sample], List[str]]:
    """
    [機能] dataset_root/ラベル名/*.png からサンプル一覧を作る
    戻り値:
      samples: (path, label_idx) の配列
      labels : idx->label_name の対応表
    """
    if not dataset_root.exists():
        raise FileNotFoundError(f"dataset_root not found: {dataset_root}")

    label_dirs = sorted([p for p in dataset_root.iterdir() if p.is_dir()])
    if not label_dirs:
        raise ValueError(f"No label folders found under: {dataset_root}")

    labels = [d.name for d in label_dirs]
    label_to_idx = {name: i for i, name in enumerate(labels)}

    samples: List[Sample] = []
    for d in label_dirs:
        for p in sorted([x for x in d.glob("*.png") if x.is_file()]):
            samples.append(Sample(path=p, label_idx=label_to_idx[d.name]))

    if not samples:
        raise ValueError("No images found. Check dataset structure and extension (*.png).")

    return samples, labels


class SimpleCNN(nn.Module):
    """
    [機能] 小さめCNN（入力: 1x64x256 を想定）
    - 出力は logits（softmaxしない）
      CrossEntropyLoss は logits を受け取る前提 :contentReference[oaicite:6]{index=6}
    """
    def __init__(self, in_ch: int, n_classes: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_ch, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 32x128

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 16x64

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 8x32
        )

        # flatten次元を動的に確定
        with torch.no_grad():
            dummy = torch.zeros(1, in_ch, IMG_H, IMG_W)
            feat = self.features(dummy)
            feat_dim = int(np.prod(feat.shape[1:]))

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(feat_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(256, n_classes),
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
    parser.add_argument("--dataset_root", type=str, default="ML/_ml_dataset",
                        help="教師データフォルダ（label/*.png 構造）")
    parser.add_argument("--model_dir", type=str, default="ML/trained_cnn_model",
                        help="学習済みモデルの保存先フォルダ")
    args = parser.parse_args()

    set_seed(RANDOM_STATE)

    dataset_root = Path(args.dataset_root)
    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    samples, label_names = list_samples(dataset_root)
    n_classes = len(label_names)
    in_ch = 1 if GRAYSCALE else 3

    # train/test split（ファイル単位で分割）
    y_all = np.array([s.label_idx for s in samples], dtype=int)
    idx_all = np.arange(len(samples))

    idx_train, idx_test = train_test_split(
        idx_all,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y_all
    )

    train_samples = [samples[i] for i in idx_train]
    test_samples = [samples[i] for i in idx_test]

    train_loader = DataLoader(ImageFolderDataset(train_samples), batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(ImageFolderDataset(test_samples), batch_size=BATCH_SIZE, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = SimpleCNN(in_ch=in_ch, n_classes=n_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()  # logits入力 :contentReference[oaicite:7]{index=7}

    # 評価用：train側も shuffle=False で見る（精度計算用）
    train_eval_loader = DataLoader(
        ImageFolderDataset(train_samples),
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    epochs_hist = []
    train_acc_hist = []
    val_acc_hist = []
    train_loss_hist = []

    for epoch in range(1, EPOCHS + 1):
        tr_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)

        tr_acc = eval_accuracy(model, train_eval_loader, device)
        va_acc = eval_accuracy(model, test_loader, device)

        epochs_hist.append(epoch)
        train_loss_hist.append(tr_loss)
        train_acc_hist.append(tr_acc)
        val_acc_hist.append(va_acc)

        print(f"[epoch {epoch:03d}/{EPOCHS}] loss={tr_loss:.4f} train_acc={tr_acc:.4f} val_acc={va_acc:.4f}")

    # 学習曲線を保存
    plot_learning_curve(
        epochs_hist,
        train_acc_hist,
        val_acc_hist,
        model_dir / "learning_curve.png"
    )

    # （任意）数値も保存して後で確認できるように
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

    # 評価指標（sklearn）:contentReference[oaicite:8]{index=8}
    cm = confusion_matrix(y_true, y_pred).tolist()
    report = classification_report(
        y_true, y_pred,
        target_names=label_names
    )

    metrics = {
        "accuracy": acc,
        "labels": label_names,
        "confusion_matrix": cm,
        "classification_report": report,
        "n_samples": int(len(samples)),
        "n_train": int(len(train_samples)),
        "n_test": int(len(test_samples)),
        "img_h": IMG_H,
        "img_w": IMG_W,
        "grayscale": GRAYSCALE,
        "test_size": TEST_SIZE,
        "random_state": RANDOM_STATE,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "lr": LR,
        "device": str(device),
    }

    # 保存（state_dict推奨）:contentReference[oaicite:9]{index=9}
    ckpt = {
        "model_state_dict": model.state_dict(),
        "label_names": label_names,
        "img_h": IMG_H,
        "img_w": IMG_W,
        "grayscale": GRAYSCALE,
    }
    torch.save(ckpt, model_dir / "model.pt")

    with open(model_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump({"label_names": label_names, "img_h": IMG_H, "img_w": IMG_W, "grayscale": GRAYSCALE},
                  f, ensure_ascii=False, indent=2)

    with open(model_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print("=== TRAIN/EVAL DONE (CNN) ===")
    print("model_dir:", model_dir.resolve())
    print("accuracy :", metrics["accuracy"])
    print("labels   :", metrics["labels"])
    print("confusion_matrix:", metrics["confusion_matrix"])
    print(metrics["classification_report"])


if __name__ == "__main__":
    main()
