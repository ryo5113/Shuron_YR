import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy import signal

# ===== 入力（ここを変更） =====
INPUT_FILES = [
    r"./Raw.wav",
    r"./Noise_Reduce.wav",
]
OUT_DIR = r"./spectrogram_out"

# ===== スペクトログラム設定 =====
NFFT = 2048          # FFTサイズ（周波数分解能）
NOVERLAP = 1536      # オーバーラップ（例: 75%）
WINDOW = "hann"
FMAX_HZ = 3000       # 例: 8000 にすると 8kHz まで表示
EPS = 1e-12         # log(0)回避
VMIN_DB = -100
VMAX_DB = -20
# ===== 波形プロット設定 =====
WAVEFORM_YLIM = 1.0  # 表示範囲（-1〜1の正規化前提）

def read_wav_mono_float(path: str) -> tuple[int, np.ndarray]:
    """wavを読み込み、モノラルfloat32（-1..1付近）に変換して返す"""
    sr, x = wavfile.read(path)

    # (samples, channels) -> mono
    if x.ndim == 2:
        x = x.mean(axis=1)

    # int16/int32 -> float
    if np.issubdtype(x.dtype, np.integer):
        maxv = np.iinfo(x.dtype).max
        x = x.astype(np.float32) / float(maxv)
    else:
        x = x.astype(np.float32)

    return sr, x

def save_spectrogram(path: str) -> str:
    sr, x = read_wav_mono_float(path)

    f, t, Sxx = signal.spectrogram(
        x,
        fs=sr,
        window=WINDOW,
        nperseg=NFFT,
        noverlap=NOVERLAP,
        nfft=NFFT,
        scaling="spectrum",  # 振幅ではなくパワーに近いスケーリング
        mode="magnitude"     # magnitude（振幅スペクトル）
    )

    # dB化（dBFS相当: 正規化済み信号の振幅をdB表示）
    S_db = 20.0 * np.log10(np.maximum(Sxx, EPS))

    # 表示周波数上限を適用
    if FMAX_HZ is not None:
        idx = f <= FMAX_HZ
        f = f[idx]
        S_db = S_db[idx, :]

    # 出力先
    os.makedirs(OUT_DIR, exist_ok=True)
    base = os.path.splitext(os.path.basename(path))[0]
    out_path = os.path.join(OUT_DIR, f"spectrogram_{base}.png")
    plt.rcParams["font.size"] = 24

    # 描画（別画像）
    plt.figure(figsize=(10, 4))
    m = plt.pcolormesh(t, f, S_db, shading="gouraud", vmin=VMIN_DB, vmax=VMAX_DB)
    plt.ylabel("Frequency [Hz]")
    plt.xlabel("Time [s]")
    plt.title(f"Spectrogram")
    cbar = plt.colorbar(m)
    cbar.set_ticks([-100, -80, -60, -40, -20])
    cbar.set_label("Amplitude [dB]")

    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()

    return out_path

def save_waveform(path: str) -> str:
    sr, x = read_wav_mono_float(path)
    t = np.arange(len(x)) / sr

    os.makedirs(OUT_DIR, exist_ok=True)
    base = os.path.splitext(os.path.basename(path))[0]
    out_path = os.path.join(OUT_DIR, f"waveform_{base}.png")
    plt.rcParams["font.size"] = 24

    plt.figure(figsize=(10, 3))
    plt.plot(t, x)
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")
    plt.title(f"Waveform : {base}")
    plt.ylim(-WAVEFORM_YLIM, WAVEFORM_YLIM)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()

    return out_path

def save_waveform_overlay(path_a: str, path_b: str) -> str:
    sr_a, xa = read_wav_mono_float(path_a)
    sr_b, xb = read_wav_mono_float(path_b)

    if sr_a != sr_b:
        raise ValueError(f"Sampling rate mismatch: {sr_a} vs {sr_b}")

    # 長さが違う場合は短い方に合わせる
    n = min(len(xa), len(xb))
    xa = xa[:n]
    xb = xb[:n]
    t = np.arange(n) / sr_a

    os.makedirs(OUT_DIR, exist_ok=True)
    base_a = os.path.splitext(os.path.basename(path_a))[0]
    base_b = os.path.splitext(os.path.basename(path_b))[0]
    out_path = os.path.join(OUT_DIR, f"waveform_overlay_{base_a}_vs_{base_b}.png")
    plt.rcParams["font.size"] = 24

    plt.figure(figsize=(18, 5))
    plt.plot(t, xa, label=base_a, alpha=0.8)
    plt.plot(t, xb, label=base_b, alpha=0.8)
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")
    plt.title("Waveform Overlay")
    plt.ylim(-WAVEFORM_YLIM, WAVEFORM_YLIM)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()

    return out_path

def save_fft_overlay_linear(path_a: str, path_b: str) -> str:
    sr_a, xa = read_wav_mono_float(path_a)
    sr_b, xb = read_wav_mono_float(path_b)
    if sr_a != sr_b:
        raise ValueError(f"Sampling rate mismatch: {sr_a} vs {sr_b}")

    # 長さを揃える（短い方に合わせる）
    n = min(len(xa), len(xb))
    xa = xa[:n]
    xb = xb[:n]

    # 窓関数（既存と同じHann）
    w = signal.windows.hann(n, sym=False)
    xa_w = xa * w
    xb_w = xb * w

    # 片側FFT
    Xa = np.fft.rfft(xa_w)
    Xb = np.fft.rfft(xb_w)
    f = np.fft.rfftfreq(n, d=1.0 / sr_a)

    # 線形振幅（dB化しない）
    Aa = np.abs(Xa) / (np.sum(w) + 1e-12)
    Ab = np.abs(Xb) / (np.sum(w) + 1e-12)

    # 表示周波数上限（あなたのスクリプトでは FMAX_HZ を使っているのでそれに合わせる）
    if FMAX_HZ is not None:
        idx = f <= FMAX_HZ
        f = f[idx]
        Aa = Aa[idx]
        Ab = Ab[idx]

    os.makedirs(OUT_DIR, exist_ok=True)
    base_a = os.path.splitext(os.path.basename(path_a))[0]
    base_b = os.path.splitext(os.path.basename(path_b))[0]
    out_path = os.path.join(OUT_DIR, f"fft_linear_overlay_{base_a}_vs_{base_b}.png")

    plt.rcParams["font.size"] = 24
    plt.figure(figsize=(18, 5))  # 添付例のように横長
    plt.plot(f, Aa, label=base_a, linewidth=1.0)
    plt.plot(f, Ab, label=base_b, linewidth=1.0)
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Amplitude")
    plt.title("Amplitude Spectrum")
    plt.xlim(0, FMAX_HZ if FMAX_HZ is not None else f[-1])
    plt.grid(True, alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()

    return out_path

def main():
    existing = [p for p in INPUT_FILES if os.path.exists(p)]
    for p in existing:
        out_s = save_spectrogram(p)
        out_w = save_waveform(p)
        print(f"[OK] saved: {out_s}")
        print(f"[OK] saved: {out_w}")

    if len(existing) >= 2:
        out_ov = save_waveform_overlay(existing[0], existing[1])
        out_fft = save_fft_overlay_linear(existing[0], existing[1])
        print(f"[OK] saved: {out_ov}")
        print(f"[OK] saved: {out_fft}")

if __name__ == "__main__":
    main()
