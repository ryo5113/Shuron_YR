# soundML_predict_SVM.py
# ---------------------------------------------------------
# [目的] soundML_train_SVM.py で作成した model.joblib / meta.json を用いて
#       wav入力 -> 学習時と同じFFT特徴量化（等間隔バンド平均） -> 推論する
# ---------------------------------------------------------
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Tuple, List, Any, Dict

import numpy as np
import joblib


def read_wav_mono_float32(wav_path: Path) -> Tuple[np.ndarray, int]:
    """wav読み込み（モノラル化・float32化）"""
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
    name = name.lower()
    if name == "hann":
        return np.hanning(n).astype(np.float32)
    if name == "hamming":
        return np.hamming(n).astype(np.float32)
    if name == "rect":
        return np.ones(n, dtype=np.float32)
    # 学習側が Unknown window をエラーにしているので同様に
    raise ValueError(f"Unknown window: {name}")


def mag_to_equal_band_features(
    mag: np.ndarray,
    freqs: np.ndarray,
    fmin: float,
    fmax: float,
    band_hz: float,
) -> np.ndarray:
    """
    学習側と同じ：fmin〜fmax を band_hz 等間隔で区切り、各バンドの平均振幅を特徴量にする
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
            #feat[i] = float(np.mean(mag[sel])) # 平均振幅に変更
            feat[i] = float(np.sum(mag[sel])) # バンド和に変更
        else:
            feat[i] = 0.0

    return feat


def wav_to_fft_feature(wav_path: Path, meta: Dict[str, Any]) -> np.ndarray:
    """
    学習側(meta.json)に合わせて wav -> FFT特徴量（等間隔バンド平均）を作る
    """
    x, sr = read_wav_mono_float32(wav_path)

    expected_sr = int(meta["fft"]["sr"])
    if int(sr) != expected_sr:
        raise ValueError(f"SR mismatch: wav={sr}, expected={expected_sr} (file={wav_path})")

    nfft = int(meta["fft"]["nfft"])
    fmin = float(meta["fft"].get("fmin", 0.0))
    fmax = float(meta["fft"]["fmax"])

    band_hz = float(meta["feature"]["band_hz"])
    use_log1p = bool(meta["fft"]["use_log1p"])
    zero_mean = bool(meta["fft"].get("zero_mean", False))
    window = str(meta["fft"].get("window", "hann"))
    feature_dim = int(meta["feature"]["feature_dim"])

    if zero_mean:
        x = x - float(np.mean(x))

    if len(x) > nfft:
        # 学習側と同じ理由（次元が崩れる）でエラーにする
        raise ValueError(f"Input longer than NFFT. len={len(x)} > nfft={nfft} (file={wav_path})")

    x_pad = np.zeros(nfft, dtype=np.float32)
    x_pad[:len(x)] = x.astype(np.float32, copy=False)

    w = make_window(nfft, window)
    X = np.fft.rfft(x_pad * w, n=nfft)
    mag = np.abs(X).astype(np.float32)

    freqs = np.fft.rfftfreq(nfft, d=1.0 / expected_sr).astype(np.float32)

    feat = mag_to_equal_band_features(
        mag=mag,
        freqs=freqs,
        fmin=fmin,
        fmax=fmax,
        band_hz=band_hz,
    )

    if use_log1p:
        feat = np.log1p(feat)

    if feat.shape[0] != feature_dim:
        raise ValueError(f"feature_dim mismatch: got={feat.shape[0]}, expected={feature_dim} (file={wav_path})")

    return feat.astype(np.float32)


def collect_wavs(p: Path) -> List[Path]:
    if p.is_file():
        return [p]
    if p.is_dir():
        return sorted([x for x in p.glob("*.wav") if x.is_file()])
    raise FileNotFoundError(f"input not found: {p}")


def load_model_any(model_path: Path):
    """
    学習側(model.joblib)は {"model": clf, "label_names": ...} を保存している。
    互換のため、Pipelineそのものが保存されているケースも受ける。
    """
    obj = joblib.load(model_path)
    if isinstance(obj, dict) and "model" in obj:
        return obj["model"], obj.get("label_names", None)
    return obj, None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_dir",
        type=str,
        default="word_Ex1/trained_all_svm_model_band/sum73/BEST_band_020Hz",
        help="model.joblib / meta.json のあるフォルダ",
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="分類したいwav（1本） or wavフォルダ",
    )
    parser.add_argument(
        "--out_csv",
        type=str,
        default="word_Ex1/trained_all_svm_model_band/sum73/BEST_band_020Hz/predictions_svm_word6.csv",
        help="推論結果CSV",
    )
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    model_path = model_dir / "model.joblib"
    meta_path = model_dir / "meta.json"
    if not model_path.exists():
        raise FileNotFoundError(f"model not found: {model_path}")
    if not meta_path.exists():
        raise FileNotFoundError(f"meta not found: {meta_path}")

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    label_names = meta["labels"]

    clf, label_names_in_model = load_model_any(model_path)
    # label_names は meta.json を正として使う（学習スクリプトがここに保存しているため）
    # ただし model.joblib 側にもある場合だけ軽く整合チェックする
    if label_names_in_model is not None and list(label_names_in_model) != list(label_names):
        raise ValueError("label_names mismatch between model.joblib and meta.json")

    wavs = collect_wavs(Path(args.input))
    if not wavs:
        raise ValueError("No wav files found in input.")

    X = np.stack([wav_to_fft_feature(w, meta) for w in wavs], axis=0)

    proba = clf.predict_proba(X)
    proba_pct = proba * 100.0
    pred_idx = np.argmax(proba, axis=1)
    pred_label = [label_names[i] for i in pred_idx]

    out_csv = Path(args.out_csv)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        header = ["path", "pred_label"] + [f"pct_{c}" for c in label_names]
        writer.writerow(header)
        for i, w in enumerate(wavs):
            row = [str(w), pred_label[i]] + [round(float(x), 1) for x in proba_pct[i]]
            writer.writerow(row)

    print("=== PREDICT DONE (SVM + Equal-band FFT from WAV) ===")
    print("out_csv:", out_csv.resolve())
    for i, (w, yhat) in enumerate(zip(wavs, pred_label)):
        order = np.argsort(-proba_pct[i])  # 降順
        topk = []
        for j in order[:3]:
            topk.append(f"{label_names[j]}: {proba_pct[i][j]:.1f}%")
        print(f"{w.name} -> {yhat}  (top3: " + ", ".join(topk) + ")")

if __name__ == "__main__":
    main()
