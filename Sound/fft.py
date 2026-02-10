"""
複数音声（WAVなど）を指定して、
(1) 生波形（時間波形）
(2) FFTで振動（周波数）スペクトル
を描画するスクリプト

必要ライブラリ:
  pip install numpy matplotlib soundfile
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
    r"word/10times_01/sakana1/cleaned_audio_chunks/voiced3/sakana.wav",
    r"word/10times_01/shakana1/cleaned_audio_chunks/voiced3/shakana.wav",
    r"word/10times_01/takana1/cleaned_audio_chunks/voiced3/takana.wav",
    r"word/10times_01/thakana1/cleaned_audio_chunks/voiced3/thakana.wav",
    r"word/10times_01/tyakana1/cleaned_audio_chunks/voiced3/tyakana.wav",
]

# --- 解析区間（波形・FFTともに同じ区間を使用） ---
START_SEC = 0.0        # 解析開始位置 [s]
DURATION_SEC = None    # 解析時間 [s]（Noneなら全区間）

# --- FFT表示設定 ---
PLOT_MAX_HZ = 8000     # FFT描画上限周波数 [Hz]
N_FFT = 65536          # Noneなら信号長に合わせる
USE_DB = False         # True: dB表示 / False: 線形振幅
DRAW_GRID_LINES = True # 60Hz刻みの縦線表示

# --- 波形表示設定 ---
PLOT_WAVEFORM = True   # Trueで生波形も描画
WAVEFORM_DOWNSAMPLE_MAX_POINTS = 20000  # 波形が長い場合に点数を間引く上限（見やすさ用）
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


def make_time_axis(n: int, fs: int, start_sec: float = 0.0) -> np.ndarray:
    """時間軸 [s] を作成"""
    return (np.arange(n) / fs) + start_sec


def maybe_downsample_for_plot(t: np.ndarray, x: np.ndarray, max_points: int):
    """描画用に点数を間引く（表示を軽く＆見やすく）"""
    if max_points is None or len(x) <= max_points:
        return t, x
    step = int(np.ceil(len(x) / max_points))
    return t[::step], x[::step]


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
    mag = mag * 2.0 / np.sum(w)

    freq = np.fft.rfftfreq(n_fft, d=1.0 / fs)
    return freq, mag


def main():
    plt.rcParams["font.size"] = 20

    # ===== 生波形用（別Figure）=====
    if PLOT_WAVEFORM:
        fig_wav, ax_wav = plt.subplots(1, 1, figsize=(18, 4))
        ax_wav.set_title("Waveform (Raw Data)")
        ax_wav.set_xlabel("Time [s]")
        ax_wav.set_ylabel("Amplitude")
        ax_wav.grid(True, which="both", linestyle="--", linewidth=0.5)

    # ===== FFT（赤線あり）=====
    fig_fft_grid, ax_fft_grid = plt.subplots(1, 1, figsize=(18, 4))
    ax_fft_grid.set_title("FFT Spectrum (with 60 Hz grid lines)")
    ax_fft_grid.set_xlabel("Frequency [Hz]")
    ax_fft_grid.set_ylabel("Magnitude [dB]" if USE_DB else "Amplitude")
    ax_fft_grid.grid(True, which="both", linestyle="--", linewidth=0.5)

    # ===== FFT（赤線なし）=====
    fig_fft_plain, ax_fft_plain = plt.subplots(1, 1, figsize=(18, 4))
    ax_fft_plain.set_title("FFT Spectrum")
    ax_fft_plain.set_xlabel("Frequency [Hz]")
    ax_fft_plain.set_ylabel("Magnitude [dB]" if USE_DB else "Amplitude")
    ax_fft_plain.grid(True, which="both", linestyle="--", linewidth=0.5)

    for f in AUDIO_FILES:
        path = Path(f)
        if not path.exists():
            raise FileNotFoundError(f"ファイルが見つかりません: {path}")

        x, fs = sf.read(str(path), always_2d=False)
        x = to_mono(np.asarray(x))
        x = slice_by_time(x, fs, START_SEC, DURATION_SEC)

        # --- 生波形 ---
        if PLOT_WAVEFORM:
            t = make_time_axis(len(x), fs, start_sec=START_SEC)
            t_plot, x_plot = maybe_downsample_for_plot(t, x, WAVEFORM_DOWNSAMPLE_MAX_POINTS)
            ax_wav.plot(t_plot, x_plot, label=path.stem)

        # --- FFT ---
        freq, mag = compute_spectrum(x, fs, n_fft=N_FFT)
        max_hz = min(PLOT_MAX_HZ, fs / 2)
        mask = (freq >= 0) & (freq <= max_hz)

        if USE_DB:
            y = 20.0 * np.log10(np.maximum(mag, 1e-12))
        else:
            y = mag

        # 同じFFT曲線を「赤線あり」「赤線なし」両方へ描画
        ax_fft_grid.plot(freq[mask], y[mask], label=path.stem)
        ax_fft_plain.plot(freq[mask], y[mask], label=path.stem)

    # 赤の縦線は「赤線ありFFT」だけに追加
    if DRAW_GRID_LINES:
        step_hz = 60
        for hz in range(0, int(PLOT_MAX_HZ) + 1, step_hz):
            ax_fft_grid.axvline(hz, color="red", linewidth=0.8, alpha=0.25)

    # X範囲
    ax_fft_grid.set_xlim(0, PLOT_MAX_HZ)
    ax_fft_plain.set_xlim(0, PLOT_MAX_HZ)

    # 凡例
    if PLOT_WAVEFORM:
        ax_wav.legend()
        fig_wav.tight_layout()

    ax_fft_grid.legend()
    ax_fft_plain.legend()
    fig_fft_grid.tight_layout()
    fig_fft_plain.tight_layout()

    # 別々に表示
    if PLOT_WAVEFORM:
        fig_wav.show()
    fig_fft_grid.show()
    fig_fft_plain.show()

    plt.show()

if __name__ == "__main__":
    main()
