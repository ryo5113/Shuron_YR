# -----------------------------------------
# 機能:
#  1) 教師データ画像（ラベル別フォルダ）を読み込む
#  2) 画像を固定サイズにリサイズし、画素ベクトル(X)へ変換
#  3) 教師9:評価1 (= test_size=0.1) で分割して学習
#  4) 学習済みモデルとメタ情報を保存
# -----------------------------------------

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib


# ====== 設定（必要ならここだけ編集） ======
# 画像入力は特徴次元が大きくなりやすいので、固定サイズに落として学習します
IMG_H = 64
IMG_W = 256
GRAYSCALE = True
RANDOM_STATE = 42
TEST_SIZE = 0.1  # 教師9：評価1 固定
MAX_ITER = 2000
# =========================================


def load_image_as_vector(img_path: Path) -> np.ndarray:
    """
    [機能] 1枚のPNGを数値ベクトルへ変換（Xの1サンプル）
    手順:
      - 画像読込
      - グレースケール化（GRAYSCALE=Trueのとき）
      - 固定サイズ(IMG_H, IMG_W)にリサイズ
      - 0..255 -> 0..1 に正規化
      - flattenして1次元ベクトル化
    """
    img = Image.open(img_path)
    img = img.convert("L") if GRAYSCALE else img.convert("RGB")
    img = img.resize((IMG_W, IMG_H), resample=Image.BILINEAR)

    arr = np.asarray(img, dtype=np.float32) / 255.0
    return arr.reshape(-1)


def load_dataset(dataset_root: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    [機能] dataset_root/ラベル名/*.png から X, y を生成
    戻り値:
      X: (n_samples, n_features)
      y: (n_samples,)
      files: 各サンプルのファイルパス（デバッグ用）
    """
    if not dataset_root.exists():
        raise FileNotFoundError(f"dataset_root not found: {dataset_root}")

    label_dirs = sorted([p for p in dataset_root.iterdir() if p.is_dir()])
    if not label_dirs:
        raise ValueError(f"No label folders found under: {dataset_root}")

    X_list: list[np.ndarray] = []
    y_list: list[str] = []
    files: list[str] = []

    for label_dir in label_dirs:
        label = label_dir.name
        img_paths = sorted([p for p in label_dir.glob("*.png") if p.is_file()])
        for img_path in img_paths:
            X_list.append(load_image_as_vector(img_path))
            y_list.append(label)
            files.append(str(img_path))

    if not X_list:
        raise ValueError("No images found. Check dataset structure and extension (*.png).")

    X = np.stack(X_list, axis=0)
    y = np.array(y_list, dtype=object)
    return X, y, files


def build_model() -> Pipeline:
    """
    [機能] 学習器（前処理 + ロジスティック回帰）を構築
    - StandardScaler: 特徴量のスケールを揃える
    - LogisticRegression: 多クラス分類（正則化はデフォルトで有効）
    """
    return Pipeline(steps=[
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            max_iter=MAX_ITER,
            multi_class="auto"
        ))
    ])


def train_and_eval(X: np.ndarray, y: np.ndarray) -> tuple[Pipeline, dict]:
    """
    [機能] 教師9：評価1で分割し、学習・評価を実行
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    model = build_model()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = float(accuracy_score(y_test, y_pred))

    labels = [str(x) for x in np.unique(y)]
    cm = confusion_matrix(y_test, y_pred, labels=np.unique(y)).tolist()
    report = classification_report(y_test, y_pred)

    metrics = {
        "accuracy": acc,
        "labels": labels,
        "confusion_matrix": cm,
        "classification_report": report,
        "n_samples": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "img_h": IMG_H,
        "img_w": IMG_W,
        "test_size": TEST_SIZE,
        "random_state": RANDOM_STATE,
    }
    return model, metrics


def save_artifacts(model: Pipeline, metrics: dict, out_dir: Path) -> None:
    """
    [機能] 学習済みモデルとメタ情報を保存
    - model.joblib
    - metrics.json
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out_dir / "model.joblib")
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", type=str, default="ML/_ml_dataset",
                        help="教師データフォルダ（label/*.png 構造）")
    parser.add_argument("--model_dir", type=str, default="ML/trained_lr_model",
                        help="学習済みモデルの保存先フォルダ")
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)
    model_dir = Path(args.model_dir)

    X, y, _ = load_dataset(dataset_root)
    model, metrics = train_and_eval(X, y)
    save_artifacts(model, metrics, model_dir)

    print("=== TRAIN/EVAL DONE ===")
    print("model_dir:", model_dir.resolve())
    print("accuracy :", metrics["accuracy"])
    print("labels   :", metrics["labels"])
    print("confusion_matrix:", metrics["confusion_matrix"])
    print(metrics["classification_report"])


if __name__ == "__main__":
    main()
