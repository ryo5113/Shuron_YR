# Denoise.py
# 依存ライブラリ:
#   pip install numpy matplotlib librosa soundfile noisereduce

import numpy as np
import matplotlib.pyplot as plt
import librosa
import soundfile as sf
import noisereduce as nr
from pathlib import Path

plt.close('all')

# ========= 設定 =========
INPUT_PATH  = "recordedSound_20251212_183527.wav"   # 入力音声ファイル
OUTPUT_DIR  = "out_denoise_ae_v2" # 出力フォルダ

# 有音部の区間（秒）: 振動スペクトル用に切り出し
TONE_START  = 0.5
TONE_END    = 10

# FFT表示帯域（Hz）
BAND_HIGH   = 3000

# STFTパラメータ
SR          = None      # None=元サンプリングのまま / 例:16000 など
N_FFT_STFT  = 2048
HOP         = 256
WINDOW      = "hann"

# 振動スペクトル用 FFT 点数（Noneならトーン長から自動）
N_FFT_SPEC  = None

# 波形の平均値を0にするかどうか
ZERO_MEAN   = True
# ========================


def extract_interval(y, fs, t_start, t_end):
    """y から [t_start, t_end] 秒区間を切り出し"""
    if t_end <= t_start:
        raise ValueError("終了時刻は開始時刻より大きくしてください。")
    n0 = int(round(t_start * fs))
    n1 = int(round(t_end   * fs))
    n0 = max(n0, 0)
    n1 = min(n1, len(y))
    if n1 <= n0:
        raise ValueError("指定区間が音声長と矛盾しています。")
    return y[n0:n1]


def compute_fft_fixed_N(y, fs, N_fft):
    """
    振動スペクトル用:
    - 指定 N_fft 点でFFTを計算
    - 短ければゼロ埋め，長ければ先頭 N_fft を使用
    - Hann窓＋片側振幅スペクトルに正規化
    """
    N_sig = len(y)
    if N_sig == 0:
        raise ValueError("FFT入力長が0です。")

    if N_sig >= N_fft:
        y_seg = y[:N_fft]
    else:
        y_seg = np.zeros(N_fft, dtype=np.float32)
        y_seg[:N_sig] = y

    window = np.hanning(N_fft)
    y_win = y_seg * window
    Y = np.fft.rfft(y_win)
    amp = np.abs(Y) / (N_fft / 2.0)
    amp[0] /= 2.0
    freq = np.fft.rfftfreq(N_fft, d=1.0 / fs)
    return freq, amp


def plot_and_save_spectrum(freq, amp, title, out_path):
    """振幅スペクトルを png で保存"""
    plt.figure()
    plt.plot(freq, amp)
    plt.xlabel("Frequency [Hz]")
    plt.xlim(0, BAND_HIGH)
    plt.ylabel("Amplitude")
    plt.ylim(0, 0.05)
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def stft_mag(y, n_fft=N_FFT_STFT, hop=HOP, window=WINDOW):
    """STFT を計算し，複素スペクトルと振幅スペクトルを返す"""
    Z = librosa.stft(y, n_fft=n_fft, hop_length=hop, window=window, center=True)
    M = np.abs(Z)
    return Z, M


def plot_spectrogram_rel(M, fs, hop, n_fft, title, out_path, fmax=None):
    """
    振幅 M から相対dB（録音内での最大値基準）スペクトログラムを作成して保存
    """
    eps = np.finfo(float).eps
    M_ref = M.max()
    S_db = 20.0 * np.log10(np.maximum(M / (M_ref + eps), eps))
    DB_FLOOR = -60.0
    S_db = np.maximum(S_db, DB_FLOOR)

    plt.figure()
    import librosa.display
    img = librosa.display.specshow(
        S_db,
        sr=fs,
        hop_length=hop,
        x_axis="time",
        y_axis="hz",#logは対数スケール,hzは線形スケール
        cmap="coolwarm"
    )
    if fmax is not None:
        plt.ylim((0, fmax))

    cbar = plt.colorbar(img, format="%.0f dB")
    cbar.set_label("Level (dB, rel. max)")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def main():
    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- 音声読み込み ----
    y, fs = librosa.load(INPUT_PATH, sr=SR, mono=True)
    y = y.astype(np.float32)
    if ZERO_MEAN:
        y = y - np.mean(y)

    print(f"Input: {Path(INPUT_PATH).resolve()}")
    print(f"Fs = {fs}, length = {len(y)} samples ({len(y)/fs:.2f} s)")

    # 休み0.5秒をノイズ参照にする
    y_noise = extract_interval(y, fs, 0.0, 0.5)

    # ---- ノイズ除去（旧 Denoise.py と同じ方式）----
    y_deno = nr.reduce_noise(y=y, y_noise=y_noise, sr=fs, stationary=True)

    # ---- 波形（元＋ノイズ除去後）----
    t = np.arange(len(y)) / fs
    plt.figure()
    plt.plot(t, y, label="Original")
    plt.plot(t, y_deno, label="Denoised", alpha=0.75)
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")
    plt.title("Waveform (Original / Denoised)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "waveform_orig_denoised.png", dpi=200)
    plt.close()

    # ---- 有音部の振動スペクトル（元・ノイズ除去後）----
    y_tone_orig = extract_interval(y,     fs, TONE_START, TONE_END)
    y_tone_deno = extract_interval(y_deno, fs, TONE_START, TONE_END)

    if N_FFT_SPEC is None:
        N_fft_spec = max(len(y_tone_orig), len(y_tone_deno))
    else:
        N_fft_spec = int(N_FFT_SPEC)

    freq, amp_orig = compute_fft_fixed_N(y_tone_orig, fs, N_fft_spec)
    _,    amp_deno = compute_fft_fixed_N(y_tone_deno, fs, N_fft_spec)

    plot_and_save_spectrum(
        freq,
        amp_orig,
        "Amplitude Spectrum (Tone - Original)",
        out_dir / "fft_tone_original2.png"
    )
    plot_and_save_spectrum(
        freq,
        amp_deno,
        "Amplitude Spectrum (Tone - Denoised)",
        out_dir / "fft_tone_denoised2.png"
    )

    # ---- スペクトログラム（元・ノイズ除去後）----
    _, M_orig = stft_mag(y,     n_fft=N_FFT_STFT, hop=HOP, window=WINDOW)
    _, M_deno = stft_mag(y_deno, n_fft=N_FFT_STFT, hop=HOP, window=WINDOW)

    plot_spectrogram_rel(
        M_orig,
        fs=fs,
        hop=HOP,
        n_fft=N_FFT_STFT,
        title="Spectrogram (Original)",
        out_path=out_dir / "spectrogram_original.png",
        fmax=BAND_HIGH
    )
    plot_spectrogram_rel(
        M_deno,
        fs=fs,
        hop=HOP,
        n_fft=N_FFT_STFT,
        title="Spectrogram (Denoised)",
        out_path=out_dir / "spectrogram_denoised.png",
        fmax=BAND_HIGH
    )

    # ---- ノイズ除去後の音声を保存 ----
    sf.write(out_dir / "cleaned_audio.wav", y_deno, int(fs))

    print("出力フォルダ:", out_dir.resolve())
    print(f"N_FFT_SPEC={N_fft_spec}, N_FFT_STFT={N_FFT_STFT}, HOP={HOP}, WINDOW='{WINDOW}'")


if __name__ == "__main__":
    main()
