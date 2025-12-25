# soundML_predict_SVM.py
# -----------------------------------------
# 機能:
#  1) model.joblib / meta.json を読み込む
#  2) 入力wav（1本 or フォルダ）をFFT特徴量に変換（学習時と同じ設定）
#  3) 予測ラベル + クラス別確率（predict_proba）をCSV出力
# -----------------------------------------

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Tuple

import numpy as np
from joblib import load


def read_wav_mono(path: Path) -> Tuple[np.ndarray, int]:
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


def crop_or_pad(x: np.ndarray, n: int) -> np.ndarray:
    if len(x) >= n:
        return x[:n]
    y = np.zeros(n, dtype=np.float32)
    y[:len(x)] = x
    return y


def make_window(n: int, kind: str) -> np.ndarray:
    if kind.lower() == "hann":
        return np.hanning(n).astype(np.float32)
    if kind.lower() == "hamming":
        return np.hamming(n).astype(np.float32)
    return np.ones(n, dtype=np.float32)


def wav_to_fft_feature(wav_path: Path, meta: dict) -> np.ndarray:
    """
    [機能] 推論用: 学習時metaと同じパラメータでFFT特徴量を作る
    """
    x, sr = read_wav_mono(wav_path)

    duration_sec = float(meta["duration_sec"])
    n_fft = int(meta["n_fft"])
    fmax = float(meta["fmax"])
    use_log = bool(meta["use_log"])
    window = str(meta.get("window", "hann"))

    n_samples = int(duration_sec * sr)
    x = crop_or_pad(x, n_samples)

    w = make_window(len(x), window)
    X = np.fft.rfft(x * w, n=n_fft)
    mag = np.abs(X).astype(np.float32)

    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr)
    mask = freqs <= fmax

    feat = mag[mask]
    if use_log:
        feat = np.log1p(feat)

    return feat


def collect_wavs(p: Path) -> list[Path]:
    if p.is_file():
        return [p]
    if p.is_dir():
        return sorted([x for x in p.glob("*.wav") if x.is_file()])
    raise FileNotFoundError(f"input not found: {p}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, default="ML/trained_svm_fft_model",
                        help="train側で保存した model.joblib / meta.json のあるフォルダ")
    parser.add_argument("--input", type=str, required=True,
                        help="分類したいwav（1本） or wavフォルダ")
    parser.add_argument("--out_csv", type=str, default="ML_fft_dataset/predictions_svm_fft.csv",
                        help="推論結果CSV")
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    model_path = model_dir / "model.joblib"
    meta_path = model_dir / "meta.json"
    if not model_path.exists():
        raise FileNotFoundError(f"model not found: {model_path}")
    if not meta_path.exists():
        raise FileNotFoundError(f"meta not found: {meta_path}")

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    label_names = meta["label_names"]

    clf = load(model_path)

    wavs = collect_wavs(Path(args.input))
    if not wavs:
        raise ValueError("No wav files found in input.")

    # 特徴量
    X = np.stack([wav_to_fft_feature(w, meta) for w in wavs], axis=0)

    # クラス別スコア（確率）
    # SVCは probability=True のとき predict_proba が有効 :contentReference[oaicite:8]{index=8}
    proba = clf.predict_proba(X)
    pred_idx = np.argmax(proba, axis=1)
    pred_label = [label_names[i] for i in pred_idx]

    out_csv = Path(args.out_csv)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        header = ["path", "pred_label"] + [f"proba_{c}" for c in label_names]
        writer.writerow(header)
        for i, w in enumerate(wavs):
            row = [str(w), pred_label[i]] + [float(x) for x in proba[i]]
            writer.writerow(row)

    print("=== PREDICT DONE (SVM + FFT) ===")
    print("out_csv:", out_csv.resolve())
    for w, yhat in zip(wavs, pred_label):
        print(w.name, "->", yhat)


if __name__ == "__main__":
    main()
