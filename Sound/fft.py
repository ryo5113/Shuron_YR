"""
複数音声（WAVなど）を指定して、FFTで振動（周波数）スペクトルを描画するスクリプト
描画範囲: 0〜10000 Hz

必要ライブラリ:
  pip install numpy matplotlib soundfile
（WAVのみなら scipy でも可ですが、ここでは soundfile を使用）
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

try:
    import soundfile as sf
except ImportError as e:
    raise SystemExit("soundfile が必要です: pip install soundfile") from e


# ========= ユーザー設定 =========
AUDIO_FILES = [
    r"word/10times_02/sa/cleaned_audio_chunks/voiced2/sakana.wav",
    r"word/10times_02/sha/cleaned_audio_chunks/voiced2/shakana.wav",
    r"word/10times_02/tha/cleaned_audio_chunks/voiced2/thakana.wav",
    r"word/10times_02/tya/cleaned_audio_chunks/voiced2/tyakana.wav",
    r"word/10times_02/ta/cleaned_audio_chunks/voiced2/takana.wav",
]

PLOT_MAX_HZ = 5000  # 描画上限周波数 [Hz]
START_SEC = 0.0       # 解析開始位置 [s]
DURATION_SEC = None   # 解析時間 [s]（Noneなら全区間）
N_FFT = None          # Noneなら信号長に合わせる（必要なら 2**15 などを指定）
USE_DB = False         # True: dB表示 / False: 線形振幅
# ==============================


def to_mono(x: np.ndarray) -> np.ndarray:
    """(N,) or (N, C) を (N,) にする（ステレオ等は平均）"""
    if x.ndim == 1:
        return x
    if x.ndim == 2:
        return x.mean(axis=1)
    raise ValueError(f"Unsupported audio shape: {x.shape}")


def slice_by_time(x: np.ndarray, fs: int, start_sec: float, duration_sec):
    start = int(round(start_sec * fs))
    start = max(0, min(start, len(x)))
    if duration_sec is None:
        return x[start:]
    length = int(round(duration_sec * fs))
    end = min(len(x), start + max(0, length))
    return x[start:end]


def compute_spectrum(x: np.ndarray, fs: int, n_fft=None):
    """
    片側スペクトル（rFFT）を計算
    戻り値: freq[Hz], mag（線形 or dB変換前の線形）
    """
    x = x.astype(np.float64)

    if len(x) == 0:
        raise ValueError("解析区間が空です。START_SEC / DURATION_SEC を確認してください。")

    # DCオフセット除去
    x = x - np.mean(x)

    # 窓関数（リーケージ軽減）
    w = np.hanning(len(x))
    xw = x * w

    if n_fft is None:
        n_fft = len(xw)
    else:
        n_fft = int(n_fft)
        if n_fft <= 0:
            raise ValueError("N_FFT は正の整数にしてください。")

    # rFFT
    X = np.fft.rfft(xw, n=n_fft)
    mag = np.abs(X)

    # 振幅スケーリング（大雑把に「振幅」っぽくする）
    # ※厳密な校正が必要なら、目的に応じて正規化を見直してください。
    mag = mag * 2.0 / np.sum(w)

    freq = np.fft.rfftfreq(n_fft, d=1.0 / fs)
    return freq, mag


def main():
    plt.figure(figsize=(18,4))
    plt.rcParams["font.size"] = 18

    for f in AUDIO_FILES:
        path = Path(f)
        if not path.exists():
            raise FileNotFoundError(f"ファイルが見つかりません: {path}")

        x, fs = sf.read(str(path), always_2d=False)
        x = to_mono(np.asarray(x))
        x = slice_by_time(x, fs, START_SEC, DURATION_SEC)

        freq, mag = compute_spectrum(x, fs, n_fft=N_FFT)

        # 0〜10000Hz（ただしナイキストまで）
        max_hz = min(PLOT_MAX_HZ, fs / 2)
        mask = (freq >= 0) & (freq <= max_hz)

        if USE_DB:
            # 0割対策
            mag_db = 20.0 * np.log10(np.maximum(mag, 1e-12))
            plt.plot(freq[mask], mag_db[mask], label=f"{path.name} (fs={fs}Hz)")
        else:
            plt.plot(freq[mask], mag[mask], label=f"{path.name} (fs={fs}Hz)")

    plt.xlim(0, PLOT_MAX_HZ)
    plt.xlabel("Frequency [Hz]", fontsize=18)
    plt.ylabel("Magnitude [dB]" if USE_DB else "Magnitude [linear]", fontsize=18)
    plt.title("FFT Spectrum", fontsize=18)
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
