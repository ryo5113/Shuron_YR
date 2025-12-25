import json
import csv
import numpy as np
import librosa
import noisereduce as nr
from pathlib import Path

# =============================
# 入力設定（allDenoise_CNN.py と同形式）
# ※あなたの allDenoise_CNN.py の FILE_CONFIGS をコピペしてOK
# =============================
FILE_CONFIGS = [
    {
        "label": "sa",
        "paths": [
            "sata_ML/sa.wav", "sata_ML2/sa.wav", "sata_ML3/sa.wav", "sata_ML4/sa.wav", "sata_ML5/sa.wav", "sata_ML7/sa.wav"
        ],
        "tone_ranges": [
            (1.0, 2.0), (2.0, 3.0), (3.0, 4.0), (4.0, 5.0), (5.0, 6.0),
            (6.0, 7.0), (7.0, 8.0), (8.0, 9.0), (9.0, 10.0), (10.0, 11.0),
        ], 
    },
    {
        "label": "sha",
        "paths": [
            "sata_ML/sha.wav", "sata_ML2/sha.wav", "sata_ML3/sha.wav", "sata_ML4/sha.wav", "sata_ML5/sha.wav", "sata_ML7/sha.wav"
        ],
        "tone_ranges": [
            (1.0, 2.0), (2.0, 3.0), (3.0, 4.0), (4.0, 5.0), (5.0, 6.0),
            (6.0, 7.0), (7.0, 8.0), (8.0, 9.0), (9.0, 10.0), (10.0, 11.0),
        ],
    },
    {
        "label": "tha",
        "paths": [
            "sata_ML/tha.wav", "sata_ML2/tha.wav", "sata_ML3/tha.wav", "sata_ML4/tha.wav", "sata_ML5/tha.wav", "sata_ML7/tha.wav"
        ],
        "tone_ranges": [
            (1.0, 2.0), (2.0, 3.0), (3.0, 4.0), (4.0, 5.0), (5.0, 6.0),
            (6.0, 7.0), (7.0, 8.0), (8.0, 9.0), (9.0, 10.0), (10.0, 11.0),
        ],
    },
    {
        "label": "tya",
        "paths": [
            "sata_ML/tya.wav", "sata_ML2/tya.wav", "sata_ML3/tya.wav", "sata_ML4/tya.wav", "sata_ML5/tya.wav", "sata_ML7/tya.wav"
        ],
        "tone_ranges": [
            (1.0, 2.0), (2.0, 3.0), (3.0, 4.0), (4.0, 5.0), (5.0, 6.0),
            (6.0, 7.0), (7.0, 8.0), (8.0, 9.0), (9.0, 10.0), (10.0, 11.0),
        ],
    },
    {
        "label": "ta",
        "paths": [
            "sata_ML/ta.wav", "sata_ML2/ta.wav", "sata_ML3/ta.wav", "sata_ML4/ta.wav", "sata_ML5/ta.wav", "sata_ML7/ta.wav"
        ],
        "tone_ranges": [
            (1.0, 2.0), (2.0, 3.0), (3.0, 4.0), (4.0, 5.0), (5.0, 6.0),
            (6.0, 7.0), (7.0, 8.0), (8.0, 9.0), (9.0, 10.0), (10.0, 11.0),
        ],
    },
]
# =============================

# ======= 出力設定 =======
OUTPUT_DIR = "ML_fft_dataset"   # 出力先フォルダ
# =======================

# ======= 音声/前処理設定（必要なら調整） =======
SR = None          # librosa.load の sr。Noneなら元のサンプリング周波数
ZERO_MEAN = True   # 波形をゼロ平均化するか
APPLY_DENOISE = True
# =============================================

# ======= FFT特徴量設定（重要） =======
N_FFT = 4096       # FFT点数（特徴次元に影響）
FMAX = 5000.0       # このHzまでを特徴量に使う
USE_LOG1P = True    # log1p(|FFT|) にするか
WINDOW = "hann"     # 窓関数
# ===================================

def make_window(n: int, kind: str) -> np.ndarray:
    kind = kind.lower()
    if kind == "hann":
        return np.hanning(n).astype(np.float32)
    if kind == "hamming":
        return np.hamming(n).astype(np.float32)
    return np.ones(n, dtype=np.float32)

def crop_or_pad(x: np.ndarray, n: int) -> np.ndarray:
    """長さを n に揃える（超過は先頭から切る／不足は0埋め）"""
    if len(x) >= n:
        return x[:n]
    y = np.zeros(n, dtype=np.float32)
    y[:len(x)] = x
    return y

def segment_to_fft_feature(y_seg: np.ndarray, sr: int) -> tuple[np.ndarray, np.ndarray]:
    """
    1区間の波形 -> FFT振幅特徴量ベクトル
    戻り: (feat, freqs_used)
    """
    # FFT点数に合わせて長さを揃える（切る or 0埋め）
    y_seg = crop_or_pad(y_seg.astype(np.float32), N_FFT)

    # 窓
    w = make_window(len(y_seg), WINDOW)
    y_seg = y_seg * w

    # rfft（実信号の正周波数側）
    X = np.fft.rfft(y_seg, n=N_FFT)
    mag = np.abs(X).astype(np.float32)

    # 周波数軸（rfft用） :contentReference[oaicite:2]{index=2}
    freqs = np.fft.rfftfreq(N_FFT, d=1.0 / sr)

    # FMAXまでを使用
    mask = freqs <= float(FMAX)
    mag = mag[mask]
    freqs_used = freqs[mask]

    if USE_LOG1P:
        mag = np.log1p(mag)

    return mag, freqs_used

def main():
    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ラベル一覧（固定順）
    labels = [cfg["label"] for cfg in FILE_CONFIGS]
    label_to_idx = {lb: i for i, lb in enumerate(labels)}

    X_list = []
    y_list = []
    manifest_rows = []

    freqs_ref = None
    sr_ref = None
    feature_dim_ref = None

    for cfg in FILE_CONFIGS:
        label = cfg["label"]
        paths = cfg["paths"]
        tone_ranges = cfg["tone_ranges"]

        for p in paths:
            wav_path = Path(p)
            y, sr = librosa.load(str(wav_path), sr=SR, mono=True)
            y = y.astype(np.float32)

            if ZERO_MEAN:
                y = y - float(np.mean(y))

            if APPLY_DENOISE:
                y = nr.reduce_noise(y=y, sr=sr, stationary=False).astype(np.float32)

            if sr_ref is None:
                sr_ref = int(sr)
            elif int(sr) != int(sr_ref):
                raise ValueError(f"Sampling rate mismatch: {sr_ref} vs {sr} (file={wav_path})")

            # tone_ranges に従って区間を切り出し → FFT特徴量
            for seg_idx, (t_start, t_end) in enumerate(tone_ranges, start=1):
                s0 = int(np.floor(t_start * sr))
                s1 = int(np.ceil(t_end * sr))
                s0 = max(0, s0)
                s1 = min(len(y), s1)
                if s1 <= s0:
                    continue

                y_seg = y[s0:s1]
                feat, freqs_used = segment_to_fft_feature(y_seg, sr)

                # 次元の整合性チェック（FMAX/N_FFTが同じなら揃うはず）
                if freqs_ref is None:
                    freqs_ref = freqs_used
                    feature_dim_ref = int(feat.shape[0])
                else:
                    if feat.shape[0] != feature_dim_ref:
                        raise ValueError(
                            f"Feature dim mismatch: {feature_dim_ref} vs {feat.shape[0]} "
                            f"(label={label}, file={wav_path}, seg={seg_idx})"
                        )

                X_list.append(feat)
                y_list.append(label_to_idx[label])

                manifest_rows.append({
                    "label": label,
                    "label_idx": label_to_idx[label],
                    "wav": str(wav_path),
                    "seg_idx": seg_idx,
                    "t_start": float(t_start),
                    "t_end": float(t_end),
                    "n_samples_seg": int(s1 - s0),
                })

    if not X_list:
        raise RuntimeError("No samples were generated. Check FILE_CONFIGS / tone_ranges.")

    X = np.stack(X_list, axis=0).astype(np.float32)  # (N, D)
    y = np.array(y_list, dtype=np.int64)

    # 保存（SVM学習側でこのdataset.npzを読む想定）
    npz_path = out_dir / "dataset.npz"
    np.savez_compressed(
        npz_path,
        X=X,
        y=y,
        labels=np.array(labels, dtype=object),
        freqs=np.array(freqs_ref, dtype=np.float32),
    )

    # manifest.csv
    csv_path = out_dir / "manifest.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["label", "label_idx", "wav", "seg_idx", "t_start", "t_end", "n_samples_seg"]
        )
        writer.writeheader()
        writer.writerows(manifest_rows)

    # meta.json
    meta = {
        "sr": int(sr_ref) if sr_ref is not None else None,
        "n_fft": int(N_FFT),
        "fmax": float(FMAX),
        "use_log1p": bool(USE_LOG1P),
        "window": str(WINDOW),
        "zero_mean": bool(ZERO_MEAN),
        "apply_denoise": bool(APPLY_DENOISE),
        "n_samples": int(X.shape[0]),
        "feature_dim": int(X.shape[1]),
        "labels": labels,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    print("=== FFT DATASET DONE ===")
    print("out_dir:", out_dir.resolve())
    print("dataset:", npz_path.resolve())
    print("N samples:", X.shape[0], "feature_dim:", X.shape[1])

if __name__ == "__main__":
    main()
