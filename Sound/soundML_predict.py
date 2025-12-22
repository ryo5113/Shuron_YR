# -----------------------------------------
# 機能:
#  1) 学習済み model.joblib を読み込む
#  2) 入力（png 1枚 or フォルダ）を読み込み、学習時と同じ前処理でベクトル化
#  3) 予測ラベル + 確率(predict_proba) をCSV出力
# -----------------------------------------

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from PIL import Image
import joblib


# ====== 学習時と同じ設定にする（train側と一致させる） ======
IMG_H = 64
IMG_W = 256
GRAYSCALE = True
# ============================================================


def load_image_as_vector(img_path: Path) -> np.ndarray:
    """
    [機能] 推論用：1枚PNGを学習時と同じ方法でベクトル化
    """
    img = Image.open(img_path)
    img = img.convert("L") if GRAYSCALE else img.convert("RGB")
    img = img.resize((IMG_W, IMG_H), resample=Image.BILINEAR)

    arr = np.asarray(img, dtype=np.float32) / 255.0
    return arr.reshape(-1)


def collect_pngs(p: Path) -> list[Path]:
    """
    [機能] 入力がファイルなら1枚、フォルダなら中の *.png を全部集める
    """
    if p.is_file():
        return [p]
    if p.is_dir():
        return sorted([x for x in p.glob("*.png") if x.is_file()])
    raise FileNotFoundError(f"input not found: {p}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, default="ML/trained_lr_model",
                        help="trainで保存した model.joblib のあるフォルダ")
    parser.add_argument("--input", type=str, required=True,
                        help="分類したいpng（1枚） or pngフォルダ")
    parser.add_argument("--out_csv", type=str, default="predictions.csv",
                        help="推論結果CSV")
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    model_path = model_dir / "model.joblib"
    if not model_path.exists():
        raise FileNotFoundError(f"model not found: {model_path}")

    model = joblib.load(model_path)

    inputs = collect_pngs(Path(args.input))
    if not inputs:
        raise ValueError("No png files found in input.")

    X = np.stack([load_image_as_vector(p) for p in inputs], axis=0)

    pred = model.predict(X)
    proba = model.predict_proba(X) if hasattr(model, "predict_proba") else None
    classes = list(model.classes_) if hasattr(model, "classes_") else []

    out_csv = Path(args.out_csv)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        header = ["path", "pred_label"] + [f"proba_{c}" for c in classes]
        writer.writerow(header)

        for i, p in enumerate(inputs):
            row = [str(p), str(pred[i])]
            if proba is not None and classes:
                row += [float(x) for x in proba[i]]
            writer.writerow(row)

    print("=== PREDICT DONE ===")
    print("out_csv:", out_csv.resolve())
    for p, yhat in zip(inputs, pred):
        print(p.name, "->", yhat)


if __name__ == "__main__":
    main()
