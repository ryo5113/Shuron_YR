"""
複数WAVを同時に「60Hzバンド幅の振幅和（必要ならlog1p）」でグラフ化
- 複数系列の重ね描きは fft.py と同じ構成（AUDIO_FILESをループしてplt.plot）:contentReference[oaicite:3]{index=3}
- 振幅和の計算は soundML_train_SVM_band.py の実装に合わせる:contentReference[oaicite:4]{index=4}
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import wave

# ========= ユーザー設定 =========
AUDIO_FILES = [
    r"word/10times_01/sakana1/cleaned_audio_chunks/voiced3/sakana.wav",
    r"word/10times_01/shakana1/cleaned_audio_chunks/voiced3/shakana.wav",
    r"word/10times_01/takana1/cleaned_audio_chunks/voiced3/takana.wav",
    r"word/10times_01/thakana1/cleaned_audio_chunks/voiced3/thakana.wav",
    r"word/10times_01/tyakana1/cleaned_audio_chunks/voiced3/tyakana.wav",
]

PLOT_MAX_HZ = 2000  # 表示上限（train側のFMAXに合わせる想定）

# train側の設定に合わせる（必要なら変更）
TARGET_SR = 48000   
FIXED_NFFT = 65536  
FMIN = 0
FMAX = 2000       
WINDOW = "hann"     
ZERO_MEAN = True    
USE_LOG1P = True    

BAND_HZ = 60.0      # 60Hzのみ
SHOW_LEGEND = True
# ==============================


def make_window(n: int, name: str) -> np.ndarray:
    name = name.lower()
    if name == "hann":
        return np.hanning(n).astype(np.float32)
    if name == "hamming":
        return np.hamming(n).astype(np.float32)
    if name == "rect":
        return np.ones(n, dtype=np.float32)
    raise ValueError(f"Unknown window: {name}")


def read_wav_mono_float32(wav_path: Path):
    """
    soundML_train_SVM_band.py と同様：
    wave標準ライブラリで読み込み、float32化し、複数chは平均してモノラル化:contentReference[oaicite:12]{index=12}
    """
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
    soundML_train_SVM_band.py の wav_to_fft_mag に合わせる：
    - sr一致チェック
    - ZERO_MEAN
    - 長い場合はエラー
    - 短い場合はゼロ埋め
    - windowを掛けて rfft → abs:contentReference[oaicite:13]{index=13}
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


def mag_to_equal_band_features_sum(mag, freqs, fmin, fmax, band_hz) -> np.ndarray:
    """
    soundML_train_SVM_band.py の mag_to_equal_band_features_sum と同等：
    fmin〜fmax を band_hz 等間隔で区切って、各バンド内の「振幅和」を返す:contentReference[oaicite:14]{index=14}
    """
    edges = np.arange(float(fmin), float(fmax) + float(band_hz), float(band_hz), dtype=np.float32)
    n_bands = int(len(edges) - 1)
    feat = np.zeros(n_bands, dtype=np.float32)

    for i in range(n_bands):
        lo = float(edges[i])
        hi = float(edges[i + 1])

        # 最終バンドだけ上端(hi)を含める:contentReference[oaicite:15]{index=15}
        if i == n_bands - 1:
            sel = (freqs >= lo) & (freqs <= hi)
        else:
            sel = (freqs >= lo) & (freqs < hi)

        feat[i] = float(np.sum(mag[sel])) if np.any(sel) else 0.0

    return feat


def main():
    plt.figure(figsize=(20, 6))
    plt.rcParams["font.size"] = 18

    nfft = int(FIXED_NFFT)
    sr = int(TARGET_SR)

    freqs = np.fft.rfftfreq(nfft, d=1.0 / sr).astype(np.float32)

    # x軸：バンド中心周波数
    edges = np.arange(float(FMIN), float(FMAX) + float(BAND_HZ), float(BAND_HZ), dtype=np.float32)
    centers = (edges[:-1] + edges[1:]) * 0.5

    for f in AUDIO_FILES:
        path = Path(f)
        if not path.exists():
            raise FileNotFoundError(f"ファイルが見つかりません: {path}")

        mag = wav_to_fft_mag(path, nfft=nfft, sr=sr)
        feat = mag_to_equal_band_features_sum(mag, freqs, fmin=FMIN, fmax=float(FMAX), band_hz=float(BAND_HZ))

        if USE_LOG1P:
            feat = np.log1p(feat)  # train側の USE_LOG1P と同じ処理:contentReference[oaicite:16]{index=16}

        band_labels = [f"{int(edges[i])}-{int(edges[i+1])}" for i in range(len(edges) - 1)]
        x = np.arange(len(band_labels))
        path = Path(f)

        # ループ内（各wavごと）で：
        plt.plot(x, feat, label=path.stem)

    plt.xticks(x, band_labels, rotation=45)   # 斜め表示（回転）は既存コードでも使用例あり:contentReference[oaicite:2]{index=2}
    plt.xlabel("Frequency Band (Hz)", fontsize=25)
    plt.ylabel("Amplitude Sum" if USE_LOG1P else "Amplitude sum", fontsize=25)
    plt.title(f"Amplitude sum per {int(BAND_HZ)}Hz band", fontsize=25)
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)

    if SHOW_LEGEND:
        plt.legend(bbox_to_anchor=(0.68, 1.1), loc='upper left', ncol = 2, fontsize=20)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
