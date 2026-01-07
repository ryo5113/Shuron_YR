# soundML_train_SVM.py
# ---------------------------------------------------------
# [目的] ラベル別に wav を集めて、FFT特徴量を作り、
#       SVMを学習してモデルを保存
#       （本版では「複数のバンド幅」を1回の実行で検証できる）
# ---------------------------------------------------------
import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt

import numpy as np
import joblib

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


# ===========================
# 実験設定（スクリプト内で編集）
# ===========================
# データ分割（教師:評価 = 7:3）
TEST_SIZE = 0.2
RANDOM_STATE = 42

# FFT特徴量
FMIN = 0
FMAX = 8000  # 特徴量化する最大周波数 [Hz]
WINDOW = "hann"  # hann / hamming / rect
ZERO_MEAN = True
USE_LOG1P = True

# 【固定FFT条件】（ユーザー提示の条件に合わせる）
TARGET_SR = 16000
FIXED_NFFT = 65536

# 【複数バンド幅を一括検証】
MULTI_BAND_EVAL = True
# 候補: 1, 5, 10, 20, 25, 30..100 (5刻み)
BAND_HZ_LIST = [1, 2, 3, 4, 5, 10, 20, 25, 30, 40, 50, 60, 70, 80, 100] 

# SVM（poly固定）
SVM_KERNEL = "rbf"  # linear / rbf / poly
SVM_DEGREE = 2
SVM_C = 3
SVM_GAMMA = "scale"
SVM_CLASS_WEIGHT = "balanced"
SVM_PROBABILITY = True


@dataclass
class Sample:
    path: Path
    label: str


def make_window(n: int, name: str) -> np.ndarray:
    name = name.lower()
    if name == "hann":
        return np.hanning(n).astype(np.float32)
    if name == "hamming":
        return np.hamming(n).astype(np.float32)
    if name == "rect":
        return np.ones(n, dtype=np.float32)
    raise ValueError(f"Unknown window: {name}")


def mag_to_equal_band_features(
    mag: np.ndarray,
    freqs: np.ndarray,
    fmin: float,
    fmax: float,
    band_hz: float,
) -> np.ndarray:
    """
    [機能] FFT振幅スペクトル mag を、fmin〜fmax を band_hz 等間隔で区切って
          各バンド内の「平均振幅」を特徴量として返す。
    """
    edges = np.arange(float(fmin), float(fmax) + float(band_hz), float(band_hz), dtype=np.float32)
    n_bands = int(len(edges) - 1)
    feat = np.zeros(n_bands, dtype=np.float32)

    for i in range(n_bands):
        lo = float(edges[i])
        hi = float(edges[i + 1])

        # 最終バンドだけ上端(hi)を含める
        if i == n_bands - 1:
            sel = (freqs >= lo) & (freqs <= hi)
        else:
            sel = (freqs >= lo) & (freqs < hi)

        if np.any(sel):
            feat[i] = float(np.mean(mag[sel]))
        else:
            feat[i] = 0.0

    return feat


def collect_labeled_wavs(wav_root: Path) -> List[Sample]:
    """
    [機能] wav_root/label/*.wav を収集（label=サブフォルダ名）
    """
    samples: List[Sample] = []
    if not wav_root.exists():
        raise FileNotFoundError(f"wav_root not found: {wav_root}")

    for label_dir in sorted([p for p in wav_root.iterdir() if p.is_dir()]):
        label = label_dir.name
        for wav_path in sorted(label_dir.glob("*.wav")):
            samples.append(Sample(path=wav_path, label=label))

    if len(samples) == 0:
        raise RuntimeError(f"No wav files found under: {wav_root}")

    return samples


def read_wav_mono_float32(wav_path: Path) -> Tuple[np.ndarray, int]:
    """
    [機能] wave標準ライブラリで読み込み（モノラル化してfloat32へ）
    16bit/32bit PCM を想定。
    """
    import wave

    with wave.open(str(wav_path), "rb") as wf:
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        sr = wf.getframerate()
        n_frames = wf.getnframes()
        raw = wf.readframes(n_frames)

    if sampwidth == 2:
        x = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    elif sampwidth == 4:
        x = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
    else:
        raise ValueError(f"Unsupported sample width: {sampwidth} bytes (file={wav_path})")

    if n_channels > 1:
        x = x.reshape(-1, n_channels).mean(axis=1)

    return x.astype(np.float32), int(sr)


def wav_to_fft_mag(wav_path: Path, nfft: int, sr: int) -> np.ndarray:
    """
    [機能] wav全区間から rFFT 振幅(mag) を作る（nfft固定）
      - 長さが短い: ゼロ埋め
      - 長さが長い: エラー（特徴次元が崩れるため）
    """
    x, sr_read = read_wav_mono_float32(wav_path)
    if sr_read != sr:
        raise ValueError(f"sr mismatch: {sr_read} vs {sr} (file={wav_path})")

    if ZERO_MEAN:
        x = x - float(np.mean(x))

    if len(x) > nfft:
        raise ValueError(f"Input longer than NFFT. len={len(x)} > nfft={nfft} (file={wav_path})")

    x_pad = np.zeros(nfft, dtype=np.float32)
    x_pad[:len(x)] = x

    w = make_window(nfft, WINDOW)
    X = np.fft.rfft(x_pad * w, n=nfft)
    mag = np.abs(X).astype(np.float32)
    return mag


def build_clf() -> Pipeline:
    """
    [機能] StandardScaler + SVM を組む
    """
    clf = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(
            kernel=SVM_KERNEL,
            degree=SVM_DEGREE,
            C=SVM_C,
            gamma=SVM_GAMMA,
            probability=SVM_PROBABILITY,
            class_weight=SVM_CLASS_WEIGHT,
            random_state=RANDOM_STATE,
            break_ties=True,
        )),
    ])
    return clf


def save_run_outputs(
    out_dir: Path,
    clf: Pipeline,
    label_names: List[str],
    meta: Dict,
    report: str,
    cm: np.ndarray,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    # モデル保存
    joblib.dump({"model": clf, "label_names": label_names}, out_dir / "model.joblib")

    # メタ情報
    with (out_dir / "meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    # レポート
    with (out_dir / "report.txt").open("w", encoding="utf-8") as f:
        f.write(report)

    # 混同行列（画像）
    try:
        disp = ConfusionMatrixDisplay(cm, display_labels=label_names)
        disp.plot(values_format="d")
        plt.tight_layout()
        plt.savefig(out_dir / "confusion_matrix.png", dpi=200)
        plt.close()
    except Exception as e:
        # matplotlibが無い/描画失敗時は画像保存をスキップ
        with (out_dir / "confusion_matrix_error.txt").open("w", encoding="utf-8") as f:
            f.write(str(e))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wav_root", type=str, default="SVM_wav_dataset_re",
                        help="ラベル別にwavが入っているルートフォルダ")
    parser.add_argument("--model_dir", type=str, default="word/trained_svm_model",
                        help="出力先フォルダ（各バンド幅の結果をサブフォルダに保存）")
    args = parser.parse_args()

    wav_root = Path(args.wav_root)
    model_root = Path(args.model_dir)
    model_root.mkdir(parents=True, exist_ok=True)

    # --- データ収集 ---
    samples = collect_labeled_wavs(wav_root)

    # ラベルID化（固定順）
    label_names = sorted(list({s.label for s in samples}))
    label_to_id = {lab: i for i, lab in enumerate(label_names)}
    y = np.array([label_to_id[s.label] for s in samples], dtype=np.int64)

    # --- FFT（mag）を全サンプルで一度だけ計算 ---
    sr = TARGET_SR
    nfft = FIXED_NFFT
    freqs = np.fft.rfftfreq(nfft, d=1.0 / sr).astype(np.float32)

    mags: List[np.ndarray] = []
    for s in samples:
        mag = wav_to_fft_mag(s.path, nfft=nfft, sr=sr)
        mags.append(mag)

    # --- バンド幅ごとに特徴量化 → 学習/評価 ---
    results: List[Dict] = []

    band_list = BAND_HZ_LIST if MULTI_BAND_EVAL else [BAND_HZ_LIST[-1]]

    for band_hz in band_list:
        # 特徴量作成
        feats = []
        for mag in mags:
            feat = mag_to_equal_band_features(
                mag=mag,
                freqs=freqs,
                fmin=FMIN,
                fmax=float(FMAX),
                band_hz=float(band_hz),
            )
            if USE_LOG1P:
                feat = np.log1p(feat)
            feats.append(feat)

        X = np.stack(feats, axis=0).astype(np.float32)

        # 分割（教師:評価 = 7:3 固定）
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
        )

        clf = build_clf()
        clf.fit(X_tr, y_tr)

        y_pred = clf.predict(X_te)

        acc = float(accuracy_score(y_te, y_pred))
        cm = confusion_matrix(y_te, y_pred)
        report = classification_report(y_te, y_pred, target_names=label_names, digits=4)

        meta = {
            "wav_root": str(wav_root),
            "n_samples": int(len(samples)),
            "labels": label_names,
            "test_size": float(TEST_SIZE),
            "random_state": int(RANDOM_STATE),
            "sr": int(sr),
            "nfft": int(nfft),
            "fmin": float(FMIN),
            "fmax": float(FMAX),
            "band_hz": float(band_hz),
            "window": WINDOW,
            "zero_mean": bool(ZERO_MEAN),
            "use_log1p": bool(USE_LOG1P),
            "feature_dim": int(X.shape[1]),
            "svm": {
                "kernel": SVM_KERNEL,
                "degree": int(SVM_DEGREE),
                "C": float(SVM_C),
                "gamma": str(SVM_GAMMA),
                "class_weight": str(SVM_CLASS_WEIGHT),
                "probability": bool(SVM_PROBABILITY),
            },
            "accuracy": acc,
        }

        # 保存先（バンド幅ごと）
        out_dir = model_root / f"band_{int(band_hz):03d}Hz"
        save_run_outputs(out_dir, clf, label_names, meta, report, cm)

        results.append({
            "band_hz": float(band_hz),
            "feature_dim": int(X.shape[1]),
            "accuracy": acc,
        })

        print(f"[band_hz={band_hz:>5}] acc={acc:.4f} feat_dim={X.shape[1]} -> {out_dir}")

    # --- まとめ保存 ---
    # JSON
    with (model_root / "band_sweep_results.json").open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # CSV（簡易）
    with (model_root / "band_sweep_results.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["band_hz", "feature_dim", "accuracy"])
        for r in results:
            w.writerow([r["band_hz"], r["feature_dim"], r["accuracy"]])

    print(f"\nSaved summary: {model_root / 'band_sweep_results.json'}")
    print(f"Saved summary: {model_root / 'band_sweep_results.csv'}")


if __name__ == "__main__":
    main()
