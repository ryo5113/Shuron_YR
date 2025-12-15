# denoise_and_analyze_all.py
# 必要: moviepy, numpy, soundfile, matplotlib, scipy
# pip install moviepy numpy soundfile matplotlib scipy

import tempfile
from pathlib import Path
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
from moviepy.editor import VideoFileClip
from scipy import signal

# ========= 設定（ここだけ編集） =========
NOISE_VIDEOS = [
    "noise/noise.mp4",
    "noise/noise2.mp4",
    "noise/noise3.mp4",
    "noise/noise4.mp4",
    "noise/noise5.mp4",
]                        # 環境音のみ（同一環境・同一設定）
TARGET_VIDEO = "testv2.mp4"      # 解析対象（発声あり）
OUTPUT_DIR   = "out_denoised_testv2"    # 出力フォルダ

SAMPLE_RATE  = 16000             # 抽出サンプリング周波数[Hz]
START_SEC    = 0.0               # 切り出し開始秒（0で先頭）
DURATION_SEC = 0.0               # 切り出し長さ（0で末尾まで）
ZERO_MEAN    = True              # 平均0化（DC成分除去）

# STFT/ISTFT のパラメータ（周波数減算とスペクトログラムで共通使用）
NFFT         = 2048              # 窓長 = FFT長（周波数分解能）
HOP_SAMPLES  = 256               # ホップ長（時間分解能）
WINDOW       = "hann"            # 窓関数

# スペクトル減算パラメータ（P_clean = max(P - ALPHA*Nbar, BETA_FLOOR*Nbar)）
ALPHA        = 1.0               # オーバーサブトラクション係数
BETA_FLOOR   = 0.02              # フロア率（0〜1）
# ========================================

def extract_audio(video_path: str, sr: int, start: float, duration: float):
    """動画から音声（float32 mono）を抽出して返す。"""
    clip = VideoFileClip(video_path)
    if start > 0 or duration > 0:
        end = None if duration <= 0 else start + duration
        clip = clip.subclip(start, end)
    if clip.audio is None:
        raise RuntimeError(f"音声トラックがありません: {video_path}")
    with tempfile.TemporaryDirectory() as td:
        wav_tmp = Path(td) / "tmp.wav"
        clip.audio.write_audiofile(
            str(wav_tmp),
            fps=sr, nbytes=2, codec="pcm_s16le",
            ffmpeg_params=["-ac", "1"], logger=None
        )
        y, sr2 = sf.read(str(wav_tmp), dtype="float32")
        if y.ndim == 2:
            y = y.mean(axis=1)
        return y, sr2

def to_zero_mean(y: np.ndarray) -> np.ndarray:
    return y - np.mean(y) if ZERO_MEAN else y

def stft_mag_phase(y: np.ndarray, sr: int):
    """STFT（片側）を計算して (freqs, times, Z, |Z|, angle(Z)) を返す。"""
    win = signal.get_window(WINDOW, NFFT, fftbins=True)
    f, t, Z = signal.stft(to_zero_mean(y), fs=sr, window=win,
                          nperseg=NFFT, noverlap=NFFT - HOP_SAMPLES,
                          nfft=NFFT, return_onesided=True, boundary=None, padded=False)
    Mag = np.abs(Z)
    Phs = np.angle(Z)
    return f, t, Z, Mag, Phs

def istft_from_mag_phase(Mag: np.ndarray, Phs: np.ndarray, sr: int):
    """振幅と位相から ISTFT で時間波形を復元。"""
    Z = Mag * np.exp(1j * Phs)
    win = signal.get_window(WINDOW, NFFT, fftbins=True)
    _, y = signal.istft(Z, fs=sr, window=win,
                        nperseg=NFFT, noverlap=NFFT - HOP_SAMPLES)
    return y.astype(np.float32)

def average_noise_power_profile(noise_videos: list, sr: int):
    """
    環境音動画群の STFT を取り、各周波数ビンのパワー(|Z|^2)を
    時間平均→ファイル平均して、平均ノイズパワープロファイル Nbar[f] を返す。
    """
    N_list = []
    f_ref = None
    for nv in noise_videos:
        y, _ = extract_audio(nv, sr, START_SEC, DURATION_SEC)
        f, t, Z, Mag, _ = stft_mag_phase(y, sr)
        P = (Mag ** 2)  # 各フレームのパワー
        P_mean_t = np.mean(P, axis=1)  # 時間平均（freq次元が残る）
        if f_ref is None:
            f_ref = f
        else:
            if len(f_ref) != len(f) or not np.allclose(f_ref, f):
                raise RuntimeError("STFT周波数軸が一致しません（パラメータを同一に）。")
        N_list.append(P_mean_t)
    Nbar = np.mean(np.vstack(N_list), axis=0)  # ファイル平均
    return f_ref, Nbar  # 形状: (F,)

def rfft_one_sided_amp_phase_pow(y: np.ndarray, sr: int): #rfftの方が高速
    """
    全区間 rFFT の片側: 振幅/位相/パワーを返す（片側2倍整合、|Y|/N, |Y|^2/N^2）。
    """
    x = to_zero_mean(y).astype(np.float64)
    N = len(x)
    Y = np.fft.rfft(x, n=NFFT if N < NFFT else None)  # 必要ならゼロパディング
    f = np.fft.rfftfreq(Y.size*2-2, d=1.0/sr) if N < NFFT else np.fft.rfftfreq(N, d=1.0/sr)
    A = np.abs(Y) / (N if N >= NFFT else NFFT)
    P = (np.abs(Y) ** 2) / ((N if N >= NFFT else NFFT) ** 2)
    if A.size > 1:
        A[1:-1] *= 2.0
        P[1:-1] *= 2.0
    phase = np.angle(Y)
    return f, A, phase, P

def save_csv(freq, arr, path, header):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(header + "\n")
        for fi, vi in zip(freq, arr):
            f.write(f"{fi:.6f},{vi:.10e}\n")

def main():
    out = Path(OUTPUT_DIR); out.mkdir(parents=True, exist_ok=True)

    # 0) ノイズ平均パワープロファイル（STFTベース）
    fN, Nbar = average_noise_power_profile(NOISE_VIDEOS, SAMPLE_RATE)
    # 保存（可視化）
    plt.figure(figsize=(10,3))
    plt.plot(fN, Nbar)
    plt.title("Noise Power Profile (STFT average)")
    plt.xlabel("Frequency (Hz)"); plt.ylabel("Power")
    plt.xlim(0,1000)
    plt.tight_layout(); plt.savefig(out/"noise_power_profile.png", dpi=200); plt.close()
    save_csv(fN, Nbar, out/"noise_power_profile.csv", "freq_hz,noise_power_avg")

    # 1) 対象音声の抽出
    y_raw, sr = extract_audio(TARGET_VIDEO, SAMPLE_RATE, START_SEC, DURATION_SEC)
    sf.write(str(out/"audio_extracted_raw.wav"), y_raw, sr)

    # 2) 対象音声の STFT とノイズ減算（パワー）
    fT, tT, Zt, MagT, PhsT = stft_mag_phase(y_raw, sr)
    if len(fT) != len(fN) or not np.allclose(fT, fN):
        raise RuntimeError("ノイズと対象の STFT 周波数軸が一致しません。NFFT/HOP/WINDOWなどを同一にしてください。")
    P_T = MagT**2
    # ブロードキャスト用に (F,1) へ整形
    Nbar_2d = Nbar.reshape(-1, 1)
    floor = BETA_FLOOR * Nbar_2d
    P_clean = np.maximum(P_T - ALPHA * Nbar_2d, floor)
    Mag_clean = np.sqrt(P_clean)

    # 3) ISTFTでクリーン波形を復元
    y_clean = istft_from_mag_phase(Mag_clean, PhsT, sr)
    # 長さを元信号に合わせて切り揃え
    if len(y_clean) > len(y_raw):
        y_clean = y_clean[:len(y_raw)]
    sf.write(str(out/"audio_denoised.wav"), y_clean, sr)

    # ===== ここから 6種グラフは “ノイズ除去後の信号 y_clean” で作成 =====

    # ① 時間波形
    t = np.arange(len(y_clean))/sr
    plt.figure(figsize=(10,3))
    plt.plot(t, y_clean)
    plt.title("Waveform (Denoised)")
    plt.xlabel("Time (s)"); plt.ylabel("Amplitude")
    plt.tight_layout(); plt.savefig(out/"waveform_denoised.png", dpi=200); plt.close()

    # ②③④ 全区間 rFFT：振幅/位相/パワー
    f, A, phase, P = rfft_one_sided_amp_phase_pow(y_clean, sr)

    plt.figure(figsize=(10,3))
    plt.plot(f, A); plt.title("Single-Sided Amplitude Spectrum (Denoised)")
    plt.xlabel("Frequency (Hz)"); plt.ylabel("Amplitude")
    plt.xlim(0, 3000)
    plt.ylim(0, 0.006)
    plt.tight_layout(); plt.savefig(out/"amplitude_spectrum_denoised.png", dpi=200); plt.close()

    plt.figure(figsize=(10,3))
    plt.plot(f, phase); plt.title("Phase Spectrum (Denoised)")
    plt.xlabel("Frequency (Hz)"); plt.ylabel("Phase (rad)")
    plt.xlim(0, 1500)
    plt.tight_layout(); plt.savefig(out/"phase_spectrum_denoised.png", dpi=200); plt.close()

    plt.figure(figsize=(10,3))
    plt.plot(f, P); plt.title("Power Spectrum (Denoised)")
    plt.xlabel("Frequency (Hz)"); plt.ylabel("Power (normalized)")
    plt.xlim(0, 1500)
    plt.tight_layout(); plt.savefig(out/"power_spectrum_denoised.png", dpi=200); plt.close()

    # ⑤ PSD（Welch） on denoised
    nperseg = NFFT
    noverlap = NFFT - HOP_SAMPLES
    win = signal.get_window(WINDOW, nperseg, fftbins=True)
    fw, psd = signal.welch(to_zero_mean(y_clean), fs=sr, window=win,
                           nperseg=nperseg, noverlap=noverlap, detrend=False, return_onesided=True)
    plt.figure(figsize=(10,3))
    plt.semilogy(fw, psd)
    plt.title("Power Spectral Density (Welch, Denoised)")
    plt.xlabel("Frequency (Hz)"); plt.ylabel("PSD")
    plt.tight_layout(); plt.savefig(out/"psd_welch_denoised.png", dpi=200); plt.close()

    # ⑥ スペクトログラム（denoised の STFT をそのまま使用）
    # すでに Mag_clean(=|Z_clean|) と fT, tT があるので、それを dB化して描画
    S_db = 20 * np.log10(np.maximum(Mag_clean, np.finfo(float).eps))
    plt.figure(figsize=(10,4))
    plt.pcolormesh(tT, fT, S_db, shading="auto")
    plt.title("Spectrogram (STFT magnitude, dB, Denoised)")
    plt.xlabel("Time (s)"); plt.ylabel("Frequency (Hz)")
    plt.ylim(0, 1500)
    cbar = plt.colorbar(); cbar.set_label("Magnitude (dB)")
    plt.tight_layout(); plt.savefig(out/"spectrogram_denoised.png", dpi=200); plt.close()

    # 参考：低域の比較図（元 vs ノイズプロファイル vs クリーン平均）
    # フレーム平均したターゲット/クリーンのパワー
    P_T_avg = np.mean(P_T, axis=1)
    P_C_avg = np.mean(P_clean, axis=1)
    plt.figure(figsize=(10,3))
    plt.plot(fT, P_T_avg, label="Target (avg power)")
    plt.plot(fT, Nbar,    label="Noise (avg power)")
    plt.plot(fT, P_C_avg, label="Clean (avg power)")
    plt.xlim(0, 1500)
    plt.title("Averaged Power vs Noise vs Clean (0–500 Hz)")
    plt.xlabel("Frequency (Hz)"); plt.ylabel("Power")
    plt.legend(); plt.tight_layout()
    plt.savefig(out/"avg_power_compare_0_500Hz.png", dpi=200); plt.close()

    # サマリ
    print("Saved to:", out.resolve())
    print(f"Fs={sr}, NFFT={NFFT}, hop={HOP_SAMPLES}, window='{WINDOW}', "
          f"ALPHA={ALPHA}, BETA_FLOOR={BETA_FLOOR}, ZERO_MEAN={ZERO_MEAN}")

if __name__ == "__main__":
    main()
