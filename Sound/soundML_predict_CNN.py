# -----------------------------------------
# 機能:
#  1) trained_cnn_model/model.pt を読み込む
#  2) 入力（png 1枚 or フォルダ）を読み込み、学習時と同じ前処理
#  3) 予測ラベル + 確率(softmax) をCSV出力
# -----------------------------------------

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image

import torch
import torch.nn as nn


# ====== 学習時と同じ設定にする（meta.json から上書きされます） ======
IMG_H = 64
IMG_W = 256
GRAYSCALE = True
# ======================================================================


def load_image_as_tensor(img_path: Path, img_h: int, img_w: int, grayscale: bool) -> torch.Tensor:
    img = Image.open(img_path)
    img = img.convert("L") if grayscale else img.convert("RGB")
    img = img.resize((img_w, img_h), resample=Image.BILINEAR)

    arr = np.asarray(img, dtype=np.float32) / 255.0
    if grayscale:
        arr = arr[None, :, :]          # (1,H,W)
    else:
        arr = arr.transpose(2, 0, 1)   # (C,H,W)
    return torch.from_numpy(arr)


def collect_pngs(p: Path) -> list[Path]:
    if p.is_file():
        return [p]
    if p.is_dir():
        return sorted([x for x in p.glob("*.png") if x.is_file()])
    raise FileNotFoundError(f"input not found: {p}")


class SimpleCNN(nn.Module):
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
            nn.Linear(256, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x  # logits


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, default="ML/trained_cnn_model",
                        help="train_cnnで保存した model.pt / meta.json のあるフォルダ")
    parser.add_argument("--input", type=str, required=True,
                        help="分類したいpng（1枚） or pngフォルダ")
    parser.add_argument("--out_csv", type=str, default="predictions_cnn.csv",
                        help="推論結果CSV")
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    model_path = model_dir / "model.pt"
    meta_path = model_dir / "meta.json"
    if not model_path.exists():
        raise FileNotFoundError(f"model not found: {model_path}")
    if not meta_path.exists():
        raise FileNotFoundError(f"meta not found: {meta_path}")

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    label_names = meta["label_names"]
    img_h = int(meta.get("img_h", IMG_H))
    img_w = int(meta.get("img_w", IMG_W))
    grayscale = bool(meta.get("grayscale", GRAYSCALE))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    in_ch = 1 if grayscale else 3
    n_classes = len(label_names)

    model = SimpleCNN(in_ch=in_ch, n_classes=n_classes, img_h=img_h, img_w=img_w).to(device)

    # checkpoint読み込み（state_dict）:contentReference[oaicite:10]{index=10}
    ckpt = torch.load(model_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    inputs = collect_pngs(Path(args.input))
    if not inputs:
        raise ValueError("No png files found in input.")

    X = torch.stack([load_image_as_tensor(p, img_h, img_w, grayscale) for p in inputs], dim=0).to(device)

    with torch.no_grad():
        logits = model(X)                           # (N,C)
        proba = torch.softmax(logits, dim=1).cpu().numpy()  # 確率として出す（softmax）

    pred_idx = np.argmax(proba, axis=1)
    pred_label = [label_names[i] for i in pred_idx]

    out_csv = Path(args.out_csv)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        header = ["path", "pred_label"] + [f"proba_{c}" for c in label_names]
        writer.writerow(header)

        for i, p in enumerate(inputs):
            row = [str(p), str(pred_label[i])] + [float(x) for x in proba[i]]
            writer.writerow(row)

    print("=== PREDICT DONE (CNN) ===")
    print("out_csv:", out_csv.resolve())
    for p, yhat in zip(inputs, pred_label):
        print(p.name, "->", yhat)


if __name__ == "__main__":
    main()
