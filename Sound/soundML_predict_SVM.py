# soundML_predict_SVM.py（wav入力・学習metaに合わせてFFT→推論）
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


def pad_to_len(x: np.ndarray, n: int) -> np.ndarray:
    if len(x) > n:
        raise ValueError(
            f"Input wav is longer than trained n_fft, so it cannot keep feature_dim while using full duration. "
            f"len(x)={len(x)} > n_fft={n} (file={wav_path})"
        )
    y = np.zeros(n, dtype=np.float32)
    y[:len(x)] = x.astype(np.float32, copy=False)
    return y


def make_window(n: int, kind: str) -> np.ndarray:
    if kind.lower() == "hann":
        return np.hanning(n).astype(np.float32)
    if kind.lower() == "hamming":
        return np.hamming(n).astype(np.float32)
    return np.ones(n, dtype=np.float32)


def wav_to_fft_feature(wav_path: Path, meta: dict) -> np.ndarray:
    x, sr = read_wav_mono(wav_path)
    x = x.astype(np.float32)

    # 学習側条件に合わせる
    expected_sr = int(meta["sr"])
    if int(sr) != expected_sr:
        raise ValueError(f"SR mismatch: wav={sr}, expected={expected_sr} (file={wav_path})")

    n_fft = int(meta["n_fft"])
    fmax = float(meta["fmax"])
    use_log1p = bool(meta["use_log1p"])
    window = str(meta.get("window", "hann"))
    feature_dim = int(meta["feature_dim"])

    if bool(meta.get("zero_mean", False)):
        x = x - float(np.mean(x))

    # N_FFTサンプルに固定してFFT（duration_secは使わない）
    x = pad_to_len(x, n_fft)

    w = make_window(n_fft, window)
    X = np.fft.rfft(x * w, n=n_fft)
    mag = np.abs(X).astype(np.float32)

    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr)
    mask = freqs <= fmax

    feat = mag[mask]
    if use_log1p:
        feat = np.log1p(feat)

    if feat.shape[0] != feature_dim:
        raise ValueError(f"feature_dim mismatch: got={feat.shape[0]}, expected={feature_dim} (file={wav_path})")

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
                        help="model.joblib / meta.json のあるフォルダ")
    parser.add_argument("--input", type=str, required=True,
                        help="分類したいwav（1本） or wavフォルダ")
    parser.add_argument("--out_csv", type=str, default="predictions_svm_fft.csv",
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

    X = np.stack([wav_to_fft_feature(w, meta) for w in wavs], axis=0)

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

    print("=== PREDICT DONE (SVM + FFT from WAV) ===")
    print("out_csv:", out_csv.resolve())
    for w, yhat in zip(wavs, pred_label):
        print(w.name, "->", yhat)


if __name__ == "__main__":
    main()
