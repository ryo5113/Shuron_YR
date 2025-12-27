# soundML_train_SVM.py（FFT_LENで全時間を反映 + ①〜③のCSV/曲線出力）
# -----------------------------------------
# 機能:
#  1) dataset_root/label/*.wav を読み込む（wav=1サンプル）
#  2) 全wavの最大長 FFT_LEN を求め、0埋めして「全時間を反映」したFFT特徴量を作る
#  3) train/test split（固定）してSVM学習
#  4) model.joblib と meta.json を保存
#  5) diagnostics 出力
#     ① eval_predictions.csv（評価データ：正解/予測/クラス別確率）
#     ② validation_curve_C.(png/csv), validation_curve_gamma.(png/csv)
#     ③ learning_curve.(png/csv)
# -----------------------------------------

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
from joblib import dump

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.model_selection import learning_curve, validation_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# ====== 設定（必要ならここだけ編集） ======
RANDOM_STATE = 42

TEST_SIZE = 0.2  # train:test = 8:2

# FFT設定
TARGET_SR = 48000
N_FFT = 4096
FMAX = 2000.0
USE_LOG1P = True
WINDOW = "hann"

# 前処理
ZERO_MEAN = True
APPLY_DENOISE = False  # ノイズ処理済みならFalseのままでOK

# ①〜③ 診断出力
DO_DIAGNOSTICS = True
CV_FOLDS = 5  # データが少なく不安定なら 3 に下げても可（出力要件外なのでここは任意）

# 検証曲線の探索範囲（SVMの代表パラメータ）
C_RANGE = np.logspace(0, 2, 12)       # 0.01 ... 1000
GAMMA_RANGE = np.logspace(-4, -2.7, 12)   # 1e-4 ... 10

# 学習曲線の train_sizes（割合）
TRAIN_SIZES = np.linspace(0.2, 1.0, 5)  # 20%,40%,60%,80%,100%
# =========================================


@dataclass
class Sample:
    path: Path
    label_idx: int


def read_wav_mono(path: Path) -> Tuple[np.ndarray, int]:
    """PCM 16/32bit wav を想定して読み込み（モノラル化）"""
    import wave
    with wave.open(str(path), "rb") as wf:
        n_ch = wf.getnchannels()
        sr = wf.getframerate()
        sampwidth = wf.getsampwidth()
        n_frames = wf.getnframes()
        raw = wf.readframes(n_frames)

    if sampwidth == 2:
        x = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    elif sampwidth == 4:
        x = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
    else:
        raise ValueError(f"Unsupported sampwidth={sampwidth}. Use PCM 16-bit or 32-bit wav.")

    if n_ch > 1:
        x = x.reshape(-1, n_ch).mean(axis=1)

    return x, sr


def pad_to_len(x: np.ndarray, n: int) -> np.ndarray:
    """不足は0埋め。超過は許容しない（FFT_LEN=maxなので基本的に超過しない想定）"""
    if len(x) > n:
        raise ValueError(f"Input longer than pad length: len(x)={len(x)} > n={n}")
    y = np.zeros(n, dtype=np.float32)
    y[:len(x)] = x.astype(np.float32, copy=False)
    return y


def make_window(n: int, kind: str) -> np.ndarray:
    k = kind.lower()
    if k == "hann":
        return np.hanning(n).astype(np.float32)
    if k == "hamming":
        return np.hamming(n).astype(np.float32)
    return np.ones(n, dtype=np.float32)


def wav_to_fft_feature(wav_path: Path, fft_len: int) -> Tuple[np.ndarray, dict]:
    """
    wav -> FFT特徴量（|rfft|の f<=FMAX を1次元ベクトル化）
    ※ fft_len=FFT_LEN まで0埋めして、切り出しwavの全時間を反映させる
    """
    x, sr = read_wav_mono(wav_path)

    if TARGET_SR is not None and int(sr) != int(TARGET_SR):
        raise ValueError(f"SR mismatch: wav={sr}, expected={TARGET_SR} (file={wav_path})")

    x = x.astype(np.float32)

    if ZERO_MEAN:
        x = x - float(np.mean(x))

    # ノイズ処理済み前提なら False のままでOK
    if APPLY_DENOISE:
        import noisereduce as nr
        x = nr.reduce_noise(y=x, sr=sr, stationary=False).astype(np.float32)

    # ★切らない：FFT_LENまで0埋めのみ
    x = pad_to_len(x, fft_len)

    w = make_window(fft_len, WINDOW)
    X = np.fft.rfft(x * w, n=fft_len)
    mag = np.abs(X).astype(np.float32)

    freqs = np.fft.rfftfreq(fft_len, d=1.0 / sr)
    mask = freqs <= float(FMAX)

    feat = mag[mask]
    if USE_LOG1P:
        feat = np.log1p(feat)

    info = {
        "sr": int(sr),
        "n_fft": int(fft_len),
        "fmax": float(FMAX),
        "use_log1p": bool(USE_LOG1P),
        "window": str(WINDOW),
        "zero_mean": bool(ZERO_MEAN),
        "apply_denoise": bool(APPLY_DENOISE),
        "feature_dim": int(feat.shape[0]),
    }
    return feat, info


def list_samples(dataset_root: Path) -> Tuple[List[Sample], List[str]]:
    """dataset_root/label/*.wav を列挙（フォルダ名=ラベル名）"""
    if not dataset_root.exists():
        raise FileNotFoundError(f"dataset_root not found: {dataset_root}")

    label_dirs = sorted([p for p in dataset_root.iterdir() if p.is_dir()])
    if not label_dirs:
        raise ValueError(f"No label folders found under: {dataset_root}")

    labels = [d.name for d in label_dirs]
    label_to_idx = {name: i for i, name in enumerate(labels)}

    samples: List[Sample] = []
    for d in label_dirs:
        wavs = sorted([x for x in d.glob("*.wav") if x.is_file()])
        for w in wavs:
            samples.append(Sample(path=w, label_idx=label_to_idx[d.name]))

    if not samples:
        raise ValueError("No wav files found. Check dataset_root/label/*.wav")

    return samples, labels


def build_clf() -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(
            kernel="rbf",
            C=10, # rbfカーネルで使う初期値(MAX 0.79)
            probability=True,          # ①のために必要
            class_weight="balanced",
            random_state=RANDOM_STATE,
            break_ties=True,
        )),
    ])


def save_eval_predictions_csv(
    out_csv: Path,
    samples_te: List[Sample],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    proba: np.ndarray,
    label_names: List[str],
):
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    header = ["path", "y_true", "y_pred"] + [f"proba_{name}" for name in label_names]
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        for s, yt, yp, pr in zip(samples_te, y_true, y_pred, proba):
            row = [str(s.path), label_names[int(yt)], label_names[int(yp)]] + [float(x) for x in pr.tolist()]
            w.writerow(row)


def save_validation_curve(
    diag_dir: Path,
    X: np.ndarray,
    y: np.ndarray,
    clf: Pipeline,
    param_name: str,
    param_range: np.ndarray,
    cv,
    title: str,
    png_name: str,
    csv_name: str,
):
    train_scores, val_scores = validation_curve(
        clf, X, y,
        param_name=param_name,
        param_range=param_range,
        cv=cv,
        scoring="accuracy",
        n_jobs=None,
    )

    tr_mean = train_scores.mean(axis=1)
    tr_std = train_scores.std(axis=1)
    va_mean = val_scores.mean(axis=1)
    va_std = val_scores.std(axis=1)

    # CSV
    out_csv = diag_dir / csv_name
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["param_value", "train_mean", "train_std", "val_mean", "val_std"])
        for pv, a, b, c, d in zip(param_range, tr_mean, tr_std, va_mean, va_std):
            w.writerow([float(pv), float(a), float(b), float(c), float(d)])

    # PNG
    import matplotlib.pyplot as plt
    plt.figure()
    plt.title(title)
    plt.xlabel(param_name)
    plt.ylabel("accuracy")
    plt.semilogx(param_range, tr_mean, marker="o", label="train")
    plt.semilogx(param_range, va_mean, marker="o", label="val")
    plt.legend()
    plt.tight_layout()
    plt.savefig(diag_dir / png_name, dpi=200)
    plt.close()


def save_learning_curve(
    diag_dir: Path,
    X: np.ndarray,
    y: np.ndarray,
    clf: Pipeline,
    cv,
    train_sizes: np.ndarray,
    png_name: str,
    csv_name: str,
):
    sizes_abs, train_scores, val_scores = learning_curve(
        clf, X, y,
        train_sizes=train_sizes,
        cv=cv,
        scoring="accuracy",
        shuffle=True,
        random_state=RANDOM_STATE,
        n_jobs=None,
    )

    tr_mean = train_scores.mean(axis=1)
    tr_std = train_scores.std(axis=1)
    va_mean = val_scores.mean(axis=1)
    va_std = val_scores.std(axis=1)

    # CSV
    out_csv = diag_dir / csv_name
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["train_size_abs", "train_mean", "train_std", "val_mean", "val_std"])
        for s, a, b, c, d in zip(sizes_abs, tr_mean, tr_std, va_mean, va_std):
            w.writerow([int(s), float(a), float(b), float(c), float(d)])

    # PNG
    import matplotlib.pyplot as plt
    plt.figure()
    plt.title("Learning curve (SVM)")
    plt.xlabel("train size (abs)")
    plt.ylabel("accuracy")
    plt.plot(sizes_abs, tr_mean, marker="o", label="train")
    plt.plot(sizes_abs, va_mean, marker="o", label="val")
    plt.legend()
    plt.tight_layout()
    plt.savefig(diag_dir / png_name, dpi=200)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", type=str, default="ML_wav_dataset",
                        help="教師データ（label/*.wav）。wavは切り出し済みを想定")
    parser.add_argument("--model_dir", type=str, default="ML/trained_svm_fft_model",
                        help="保存先フォルダ")
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)
    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    samples, label_names = list_samples(dataset_root)
    y_all = np.array([s.label_idx for s in samples], dtype=int)

    # ★FFT_LEN（全wavの最大長）を計測
    max_len = 0
    for s in samples:
        x_tmp, sr_tmp = read_wav_mono(s.path)
        if TARGET_SR is not None and int(sr_tmp) != int(TARGET_SR):
            raise ValueError(f"SR mismatch: wav={sr_tmp}, expected={TARGET_SR} (file={s.path})")
        max_len = max(max_len, len(x_tmp))

    FFT_LEN = int(max(max_len, N_FFT))
    print(f"[INFO] FFT_LEN (max wav length) = {FFT_LEN} samples")

    # 特徴量作成
    X_list = []
    meta_info = None
    for s in samples:
        feat, info = wav_to_fft_feature(s.path, FFT_LEN)
        X_list.append(feat)
        if meta_info is None:
            meta_info = info
        else:
            if feat.shape[0] != int(meta_info["feature_dim"]):
                raise ValueError(
                    f"feature_dim mismatch: {feat.shape[0]} vs {meta_info['feature_dim']} (file={s.path})"
                )

    X = np.stack(X_list, axis=0).astype(np.float32)  # (N, D)

    # train/test split（層化）
    idx = np.arange(len(samples))
    idx_tr, idx_te = train_test_split(
        idx, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y_all
    )
    X_tr, X_te = X[idx_tr], X[idx_te]
    y_tr, y_te = y_all[idx_tr], y_all[idx_te]

    clf = build_clf()
    clf.fit(X_tr, y_tr)

    y_pred = clf.predict(X_te)
    acc = float(accuracy_score(y_te, y_pred))
    cm = confusion_matrix(y_te, y_pred)
    report = classification_report(y_te, y_pred, target_names=label_names, digits=4)

    # 保存（モデル）
    dump(clf, model_dir / "model.joblib")

    meta = {
        "label_names": label_names,
        "random_state": int(RANDOM_STATE),
        "test_size": float(TEST_SIZE),
        "dataset_root": str(dataset_root.resolve()),
        "n_samples": int(X.shape[0]),
        "feature_dim": int(X.shape[1]),
        **(meta_info or {}),
    }
    (model_dir / "meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )

    # ①〜③ 診断出力
    if DO_DIAGNOSTICS:
        diag_dir = model_dir / "diagnostics"
        diag_dir.mkdir(parents=True, exist_ok=True)

        # ① 評価データ：クラス別確率をCSVへ
        proba = clf.predict_proba(X_te)
        samples_te = [samples[i] for i in idx_te]
        save_eval_predictions_csv(
            out_csv=diag_dir / "eval_predictions.csv",
            samples_te=samples_te,
            y_true=y_te,
            y_pred=y_pred,
            proba=proba,
            label_names=label_names,
        )

        # ② 検証曲線（C, gamma）
        cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

        save_validation_curve(
            diag_dir=diag_dir,
            X=X, y=y_all,
            clf=build_clf(),
            param_name="svm__C",
            param_range=C_RANGE,
            cv=cv,
            title="Validation curve: svm__C",
            png_name="validation_curve_C.png",
            csv_name="validation_curve_C.csv",
        )

        save_validation_curve(
            diag_dir=diag_dir,
            X=X, y=y_all,
            clf=build_clf(),
            param_name="svm__gamma",
            param_range=GAMMA_RANGE,
            cv=cv,
            title="Validation curve: svm__gamma",
            png_name="validation_curve_gamma.png",
            csv_name="validation_curve_gamma.csv",
        )

        # ③ 学習曲線
        save_learning_curve(
            diag_dir=diag_dir,
            X=X, y=y_all,
            clf=build_clf(),
            cv=cv,
            train_sizes=TRAIN_SIZES,
            png_name="learning_curve.png",
            csv_name="learning_curve.csv",
        )

    print("=== TRAIN/EVAL DONE (SVM + FFT from WAV) ===")
    print("model_dir:", str(model_dir.resolve()))
    print("dataset  :", str(dataset_root.resolve()))
    print("accuracy :", acc)
    print("labels   :", label_names)
    print("confusion_matrix:", cm.tolist())
    print(report)
    if DO_DIAGNOSTICS:
        print("diagnostics_dir:", str((model_dir / "diagnostics").resolve()))
        print("  - eval_predictions.csv")
        print("  - validation_curve_C.(png/csv)")
        print("  - validation_curve_gamma.(png/csv)")
        print("  - learning_curve.(png/csv)")


if __name__ == "__main__":
    main()
