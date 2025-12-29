# soundML_predict_SVM.py
# ---------------------------------------------------------
# 機能:
#  1) model.joblib と meta.json を読み込む
#  2) 入力wav（1つ or フォルダ）から学習時と同じNFFTでFFT特徴量を作る
#  3) 予測ラベル + 確率（predict_proba）をCSVに出す
# ---------------------------------------------------------

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import List, Tuple

import numpy as np
import joblib


# train側と同じ前処理（meta.json の値に従う）
USE_LOG1P_DEFAULT = True
ZERO_MEAN_DEFAULT = True
WINDOW_DEFAULT = "hann"


def read_wav_mono_float32(wav_path: Path) -> Tuple[np.ndarray, int]:
    import wave

    with wave.open(str(wav_path), "rb") as wf:
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        fr = wf.getframerate()
        n_frames = wf.getnframes()
        raw = wf.readframes(n_frames)

    if sampwidth == 2:
        x = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    elif sampwidth == 4:
        x = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
    else:
        raise ValueError(f"Unsupported sampwidth={sampwidth} (file={wav_path})")

    if n_channels > 1:
        x = x.reshape(-1, n_channels).mean(axis=1)

    return x.astype(np.float32), int(fr)


def make_window(n: int, name: str) -> np.ndarray:
    if name == "hann":
        return np.hanning(n).astype(np.float32)
    if name == "hamming":
        return np.hamming(n).astype(np.float32)
    if name == "rect":
        return np.ones(n, dtype=np.float32)
    raise ValueError(f"Unknown window: {name}")


def wav_to_fft_feature(
    wav_path: Path,
    nfft: int,
    sr: int,
    fmax: float,
    use_log1p: bool,
    zero_mean: bool,
    window: str,
) -> np.ndarray:
    x, sr_read = read_wav_mono_float32(wav_path)
    if sr_read != sr:
        raise ValueError(f"sr mismatch: {sr_read} vs {sr} (file={wav_path})")

    if zero_mean:
        x = x - float(np.mean(x))

    if len(x) > nfft:
        raise ValueError(f"Input longer than NFFT. len={len(x)} > nfft={nfft} (file={wav_path})")

    x_pad = np.zeros(nfft, dtype=np.float32)
    x_pad[:len(x)] = x

    w = make_window(nfft, window)
    X = np.fft.rfft(x_pad * w, n=nfft)
    mag = np.abs(X).astype(np.float32)

    freqs = np.fft.rfftfreq(nfft, d=1.0 / sr)
    mask = freqs <= float(fmax)
    feat = mag[mask]

    if use_log1p:
        feat = np.log1p(feat)

    return feat.astype(np.float32)


def collect_wavs(p: Path) -> List[Path]:
    if p.is_file():
        return [p]
    if p.is_dir():
        return sorted([x for x in p.rglob("*.wav") if x.is_file()])
    raise FileNotFoundError(f"input not found: {p}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, default="ML_SVM/trained_svm_model",
                        help="trainで保存した model.joblib / meta.json のあるフォルダ")
    parser.add_argument("--input", type=str, required=True,
                        help="分類したい wav（1つ）or wavフォルダ")
    parser.add_argument("--out_csv", type=str, default="ML_SVM/predictions_sa.csv",
                        help="推論結果CSV")
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    model_path = model_dir / "model.joblib"
    meta_path = model_dir / "meta.json"
    if not model_path.exists():
        raise FileNotFoundError(f"model not found: {model_path}")
    if not meta_path.exists():
        raise FileNotFoundError(f"meta not found: {meta_path}")

    clf = joblib.load(model_path)
    meta = json.loads(meta_path.read_text(encoding="utf-8"))

    label_names = meta["label_names"]
    sr = int(meta.get("sr", 48000))
    if "nfft" not in meta:
        raise KeyError(f"meta.json に 'nfft' がありません。学習側で sr/nfft を保存するよう修正してください: {meta_path}")

    nfft = int(meta["nfft"])
    fmax = float(meta.get("fmax", 2000.0))
    use_log1p = bool(meta.get("use_log1p", USE_LOG1P_DEFAULT))
    zero_mean = bool(meta.get("zero_mean", ZERO_MEAN_DEFAULT))
    window = str(meta.get("window", WINDOW_DEFAULT))

    wavs = collect_wavs(Path(args.input))
    if not wavs:
        raise ValueError("No wav files found.")

    X = []
    for w in wavs:
        feat = wav_to_fft_feature(
            w, nfft=nfft, sr=sr, fmax=fmax,
            use_log1p=use_log1p, zero_mean=zero_mean, window=window
        )
        X.append(feat)
    X = np.stack(X, axis=0)

    pred_id = clf.predict(X)
    # probability=True のとき predict_proba が使える（あなたの旧SVMスクリプトでも同様）:contentReference[oaicite:6]{index=6}
    proba = clf.predict_proba(X) if hasattr(clf, "predict_proba") else None

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        header = ["path", "pred_label"] + [f"proba_{name}" for name in label_names]
        w.writerow(header)

        for i, wav_path in enumerate(wavs):
            pid = int(pred_id[i])
            row = [str(wav_path), label_names[pid]]
            if proba is None:
                row += [""] * len(label_names)
            else:
                row += [float(v) for v in proba[i].tolist()]
            w.writerow(row)

    print("=== PRED DONE (SVM + FFT from WAV) ===")
    print("model_dir:", str(model_dir.resolve()))
    print("n_files  :", len(wavs))
    print("out_csv  :", str(out_csv.resolve()))


if __name__ == "__main__":
    main()
