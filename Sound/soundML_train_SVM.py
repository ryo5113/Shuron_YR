# soundML_train_SVM.py（npz入力版）
# -----------------------------------------
# 機能:
#  1) dataset.npz から X,y,labels を読み込む（wav/FFT計算はしない）
#  2) train/test split（7:3固定）して SVM（SVC）学習
#  3) 評価（accuracy, confusion_matrix, classification_report）を出力
#  4) model.joblib と meta.json を保存
# -----------------------------------------

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from joblib import dump

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# ====== 設定（必要ならここだけ編集） ======
RANDOM_STATE = 42
TEST_SIZE = 0.3   # 固定 7:3
# =========================================


def load_npz_dataset(npz_path: Path) -> tuple[np.ndarray, np.ndarray, list[str], dict]:
    """
    dataset.npz から X, y, labels を読む
    期待するキー:
      - X: (N, D) float
      - y: (N,) int
      - labels: (K,) object(str)
    """
    if not npz_path.exists():
        raise FileNotFoundError(f"dataset not found: {npz_path}")

    data = np.load(npz_path, allow_pickle=True)
    if "X" not in data or "y" not in data or "labels" not in data:
        raise ValueError(f"dataset.npz must contain keys: X, y, labels (got: {list(data.keys())})")

    X = data["X"].astype(np.float32)
    y = data["y"].astype(int)
    labels = [str(x) for x in data["labels"].tolist()]

    # メタ（あれば一緒に持っておく：freqs等）
    meta_extra = {}
    for k in data.keys():
        if k not in ("X", "y", "labels"):
            # freqsなどが入っていれば保存用に持つ（必須ではない）
            meta_extra[k] = data[k].tolist() if isinstance(data[k], np.ndarray) else data[k]

    return X, y, labels, meta_extra


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_npz", type=str, default="ML_fft_dataset/dataset.npz",
                        help="教師データ npz（X,y,labels を含む）")
    parser.add_argument("--model_dir", type=str, default="ML/trained_svm_fft_model",
                        help="保存先フォルダ")
    args = parser.parse_args()

    dataset_npz = Path(args.dataset_npz)
    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    X, y, label_names, meta_extra = load_npz_dataset(dataset_npz)

    # train/test split（層化）
    idx = np.arange(len(y))
    idx_tr, idx_te = train_test_split(
        idx,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    X_tr, X_te = X[idx_tr], X[idx_te]
    y_tr, y_te = y[idx_tr], y[idx_te]

    # SVC：多クラスは one-vs-one で扱われる :contentReference[oaicite:3]{index=3}
    # probability=True で predict_proba が使える（学習はやや重くなる） :contentReference[oaicite:4]{index=4}
    clf = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=RANDOM_STATE, break_ties=True)),
    ])

    clf.fit(X_tr, y_tr)

    y_pred = clf.predict(X_te)
    acc = float(accuracy_score(y_te, y_pred))
    cm = confusion_matrix(y_te, y_pred)
    report = classification_report(y_te, y_pred, target_names=label_names, digits=4)

    # 保存
    dump(clf, model_dir / "model.joblib")

    meta = {
        "label_names": label_names,
        "random_state": int(RANDOM_STATE),
        "test_size": float(TEST_SIZE),
        "dataset_npz": str(dataset_npz.resolve()),
        "n_samples": int(X.shape[0]),
        "feature_dim": int(X.shape[1]),
        # dataset側に freqs 等が入っていれば残す（必須ではない）
        **({"dataset_extra": meta_extra} if meta_extra else {}),
    }
    (model_dir / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    print("=== TRAIN/EVAL DONE (SVM + FFT from NPZ) ===")
    print("model_dir:", str(model_dir.resolve()))
    print("dataset  :", str(dataset_npz.resolve()))
    print("accuracy :", acc)
    print("labels   :", label_names)
    print("confusion_matrix:", cm.tolist())
    print(report)


if __name__ == "__main__":
    main()
