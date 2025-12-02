# soundDenoise_envavg.py
# 依存ライブラリ:
#   pip install numpy matplotlib librosa soundfile moviepy

import numpy as np
import matplotlib.pyplot as plt
import librosa
from pathlib import Path
from moviepy.editor import VideoFileClip

plt.close('all')

# ========= 設定（ここを編集） =========
# 有音部を含むメイン音声 or 動画
INPUT_PATH  = "input.wav"

# 環境ノイズだけを録音した音声ファイルを複数指定
NOISE_FILES = [
    "noise/noise1.m4a",
    "noise/noise2.m4a",
    "noise/noise6.m4a"
]

OUTPUT_DIR  = "out_envavg"      # 出力フォルダ

# ---- メイン音声内の「有音区間」（秒） ----
TONE_START  = 3.0
TONE_END    = 4.0

# ---- FFT表示帯域（Hz） ----
BAND_LOW    = 10
BAND_HIGH   = 2000

# ---- STFT/差分パラメータ ----
SR          = None      # None=元サンプリングのまま / 例:16000 など
N_FFT_STFT  = 2048      # STFT用FFT点数
HOP         = 256
WINDOW      = "hann"
ALPHA       = 1.0       # ノイズ差分の強さ（>1で強め）
ZERO_MEAN   = True      # 平均0化（DC成分除去）

# ---- 振動スペクトル用FFT点数（Noneなら自動で最大長に合わせる） ----
N_FFT_SPEC  = None
# ===================================


def load_media(path, sr=SR, zero_mean=ZERO_MEAN):
    """
    音声ファイル or 動画ファイルからモノラル音声 y とサンプリング周波数 fs を取得する。
    """
    path = str(path)
    ext = Path(path).suffix.lower()

    audio_exts = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac"}
    video_exts = {".mp4", ".mov", ".avi", ".mkv", ".wmv"}

    if ext in audio_exts:
        y, fs = librosa.load(path, sr=sr, mono=True)
    elif ext in video_exts:
        with VideoFileClip(path) as clip:
            if clip.audio is None:
                raise ValueError("動画ファイルに音声トラックが存在しません。")
            fps = sr if sr is not None else clip.audio.fps
            arr = clip.audio.to_soundarray(fps=fps)
            if arr.ndim == 2:
                y = arr.mean(axis=1)
            else:
                y = arr.astype(np.float32)
            fs = fps
        if sr is not None and fs != sr:
            y = librosa.resample(y, orig_sr=fs, target_sr=sr)
            fs = sr
    else:
        raise ValueError(f"未対応の拡張子です: {ext}")

    y = y.astype(np.float32)
    if zero_mean:
        y = y - np.mean(y)
    return y, float(fs)


def extract_interval(y, fs, t_start, t_end):
    """
    y から [t_start, t_end] 秒の区間を切り出す。
    """
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
    - 指定された N_fft 点でFFTを計算する。
    - 信号が短ければゼロ埋め、長ければ先頭 N_fft 分を使用。
    - こうすることで、トーンとノイズで「同じ周波数軸」を保証する。
    """
    N_sig = len(y)
    if N_sig == 0:
        raise ValueError("FFT入力長が0です。")

    if N_sig >= N_fft:
        y_seg = y[:N_fft]
    else:
        # ゼロ埋めして長さ N_fft にそろえる
        y_seg = np.zeros(N_fft, dtype=np.float32)
        y_seg[:N_sig] = y

    window = np.hanning(N_fft)
    y_win = y_seg * window
    Y = np.fft.rfft(y_win)
    amp = np.abs(Y) / (N_fft/2.0)   # 片側振幅スペクトル（正規化）
    amp[0] /= 2.0                   # DC成分の補正
    freq = np.fft.rfftfreq(N_fft, d=1.0/fs)
    return freq, amp


def plot_and_save_spectrum(freq, amp, title, out_path):
    """
    振幅スペクトルを保存（線形スケール）。
    """
    plt.figure()
    plt.plot(freq, amp)
    plt.xlabel("Frequency [Hz]")
    plt.xlim(0, BAND_HIGH)
    plt.ylabel("Amplitude")
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def stft_mag(y, n_fft=N_FFT_STFT, hop=HOP, window=WINDOW):
    """
    STFT を計算し、複素スペクトル Z と振幅 M を返す。
    """
    Z = librosa.stft(y, n_fft=n_fft, hop_length=hop, window=window, center=True)
    M = np.abs(Z)
    return Z, M


def plot_spectrogram_rel(M, fs, hop, n_fft, title, out_path, fmax=None):
    """
    振幅 M から「相対強度(0〜1)」のスペクトログラムを作成して保存する。
    実dBではなく、録音内での相対的な強さだけを色で表現する。
    """
    eps = np.finfo(float).eps
    M_ref = M.max()
    S_db = 20.0 * np.log10(np.maximum(M / (M_ref + eps), eps))
    DB_FLOOR = -60.0  # 例えば -60 dB 以下は「無視」
    S_db = np.maximum(S_db, DB_FLOOR)

    plt.figure()
    import librosa.display
    img = librosa.display.specshow(
        S_db,
        sr=fs,
        hop_length=hop,
        x_axis="time",
        y_axis="hz",
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

    # ---- メイン音声 ----
    y_main, fs = load_media(INPUT_PATH, sr=SR, zero_mean=ZERO_MEAN)

    # 有音部（トーン）だけを切り出し（FFT用）
    y_tone = extract_interval(y_main, fs, TONE_START, TONE_END)

    # ---- 環境ノイズファイルをすべて読み込み ----
    noise_signals = []
    for nf in NOISE_FILES:
        y_n, fs_n = load_media(nf, sr=SR, zero_mean=ZERO_MEAN)
        if fs_n != fs:
            raise RuntimeError(f"サンプリング周波数が一致しません: {nf} (fs={fs_n}) vs main (fs={fs})")
        noise_signals.append(y_n)

    if len(noise_signals) == 0:
        raise RuntimeError("NOISE_FILES が空です。環境ノイズ音声を指定してください。")

    # ========= 1) FFT 用ノイズスペクトル（平均） =========
    # N_fft を決める：
    #  - 指定があればそれを使う
    #  - None の場合は「トーンの長さ」と「ノイズ群の中で最長の長さ」のうち最大にする
    if N_FFT_SPEC is None:
        max_noise_len = max(len(x) for x in noise_signals)
        N_fft_spec = max(len(y_tone), max_noise_len)
    else:
        N_fft_spec = int(N_FFT_SPEC)

    # ノイズ各ファイルのFFTを「同じ N_fft_spec」で計算し、振幅を平均
    noise_amps = []
    for y_n in noise_signals:
        freq_n, amp_n = compute_fft_fixed_N(y_n, fs, N_fft_spec)
        noise_amps.append(amp_n)
    noise_amps = np.stack(noise_amps, axis=0)      # (num_noise, F)
    amp_noise_mean = noise_amps.mean(axis=0)       # (F,)

    # トーンのFFTも「同じ N_fft_spec」で計算
    freq_t, amp_t = compute_fft_fixed_N(y_tone, fs, N_fft_spec)

    # 差分（トーン - 平均ノイズ）。負になったところは 0 にクリップ
    amp_diff = amp_t - amp_noise_mean
    amp_diff = np.maximum(amp_diff, 0.0)

    # FFT グラフ保存
    plot_and_save_spectrum(freq_t, amp_t,
                           "Amplitude Spectrum (Tone)",
                           out_dir / "fft_tone.png")
    plot_and_save_spectrum(freq_t, amp_noise_mean,
                           "Amplitude Spectrum (Noise mean)",
                           out_dir / "fft_noise_mean.png")
    plot_and_save_spectrum(freq_t, amp_diff,
                           "Amplitude Spectrum (Tone - Noise mean)",
                           out_dir / "fft_diff_tone_minus_noise_mean.png")

    # ========= 2) スペクトログラム用ノイズプロファイル（平均） =========
    # 各ノイズ信号の STFT パワーを時間平均 → さらにファイル間で平均
    noise_power_profiles = []
    for y_n in noise_signals:
        _, M_n = stft_mag(y_n, n_fft=N_FFT_STFT, hop=HOP, window=WINDOW)
        P_n = M_n**2                       # (F, T)
        P_n_mean_time = P_n.mean(axis=1)   # 時間平均 → (F,)
        noise_power_profiles.append(P_n_mean_time)

    noise_power_profiles = np.stack(noise_power_profiles, axis=0)  # (num_noise, F)
    Nbar = noise_power_profiles.mean(axis=0)                       # ファイル間平均 → (F,)

    # メイン音声全体の STFT
    Z_all, M_all = stft_mag(y_main, n_fft=N_FFT_STFT, hop=HOP, window=WINDOW)
    P_all = M_all**2   # (F, T)

    # 「全体音声 − 平均ノイズパワー」(負は 0 でクリップ)
    Nbar2d = Nbar.reshape(-1, 1)       # (F,1) にしてブロードキャスト
    P_clean = P_all - ALPHA * Nbar2d
    P_clean = np.maximum(P_clean, 0.0)
    M_clean = np.sqrt(P_clean)

    # スペクトログラム保存
    plot_spectrogram_rel(
        M_clean,
        fs=fs,
        hop=HOP,
        n_fft=N_FFT_STFT,
        title="Spectrogram (All - mean Noise)",
        out_path=out_dir / "spectrogram_all_minus_noise_mean.png",
        fmax=BAND_HIGH
    )

    print("入力ファイル :", Path(INPUT_PATH).resolve())
    print("ノイズファイル数 :", len(NOISE_FILES))
    print("出力フォルダ :", out_dir.resolve())
    print(f"Fs={fs}, N_FFT_SPEC={N_fft_spec}, N_FFT_STFT={N_FFT_STFT}, "
          f"hop={HOP}, window='{WINDOW}', ALPHA={ALPHA}, ZERO_MEAN={ZERO_MEAN}")


if __name__ == "__main__":
    main()
