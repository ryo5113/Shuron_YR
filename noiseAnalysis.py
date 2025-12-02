# testAnalysis_extended.py
# 必要ライブラリ: moviepy, numpy, soundfile, matplotlib, scipy
# pip install moviepy numpy soundfile matplotlib scipy

import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
from moviepy.editor import VideoFileClip
from scipy import signal

# ======= 設定（ここを書き換える） =======
INPUT_VIDEO = "noise/noise.mp4"   # 解析したい動画ファイル（絶対/相対パス）
OUTPUT_DIR  = "out_noise_all"      # 出力フォルダ
SAMPLE_RATE = 16000                # 抽出サンプリング周波数[Hz]
START_SEC   = 0.0                  # 切り出し開始秒（0で先頭）
DURATION_SEC= 0.0                  # 切り出し長さ（0で末尾まで）
ZERO_MEAN   = True                 # FFT前に平均を0化するか

# 解析パラメータ（STFT/PSD）
NFFT        = 2048                 # FFT長（スペクトログラム/PSD用）
HOP_SAMPLES = 256                  # STFTホップ長（サンプル）
WINDOW      = "hann"               # 窓関数
# =======================================

def extract_audio_to_wav(video_path: str, wav_path: str, sr: int, start: float, duration: float):
    clip = VideoFileClip(video_path)
    if start > 0 or duration > 0:
        end = None if duration <= 0 else start + duration
        clip = clip.subclip(start, end)
    if clip.audio is None:
        raise RuntimeError("入力動画に音声トラックが存在しません。")
    # モノラル/PCM16、指定サンプリング周波数で出力（verboseは使わない）
    clip.audio.write_audiofile(
        wav_path,
        fps=sr,
        nbytes=2,
        codec="pcm_s16le",
        ffmpeg_params=["-ac", "1"],
        logger=None
    )

def compute_one_sided_amplitude_phase_power(y: np.ndarray, sr: int): 
    """
    全区間のrFFTから 片側: 振幅, 位相, パワー を返す。
    振幅 A = |Y|/N（0/Nyquist以外は2倍）
    パワー P = |Y|^2 / N^2（0/Nyquist以外は2倍）
    位相は angle(Y)（-pi..pi）
    """
    y = np.asarray(y, dtype=np.float64)
    N = y.size
    if ZERO_MEAN:
        y = y - np.mean(y)
    Y = np.fft.rfft(y)
    f = np.fft.rfftfreq(N, d=1.0/sr) #N：信号のデータ数　d：サンプリング間隔 

    A = np.abs(Y) / N
    P = (np.abs(Y) ** 2) / (N ** 2)
    if N > 1:
        A[1:-1] *= 2.0
        P[1:-1] *= 2.0
    phase = np.angle(Y)  # [-pi, pi]
    return f, A, phase, P

def main():
    in_path = Path(INPUT_VIDEO)
    if not in_path.exists():
        print(f"入力ファイルが見つかりません: {in_path}", file=sys.stderr)
        sys.exit(1)

    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 一時WAVへ抽出
    with tempfile.TemporaryDirectory() as td:
        tmp_wav = Path(td) / "extracted.wav"
        try:
            extract_audio_to_wav(
                video_path=str(in_path),
                wav_path=str(tmp_wav),
                sr=SAMPLE_RATE,
                start=START_SEC,
                duration=DURATION_SEC
            )
        except Exception as e:
            print(f"音声抽出エラー: {e}", file=sys.stderr)
            sys.exit(2)

        # 音声読み込み（float32, mono）
        y, sr = sf.read(str(tmp_wav), dtype="float32")
        if y.ndim == 2:
            y = y.mean(axis=1)

    # 出力: 抽出WAV
    wav_out = out_dir / "audio_extracted.wav"
    sf.write(str(wav_out), y, sr)

    # 1) 時間波形
    t = np.arange(len(y)) / sr
    plt.figure(figsize=(10, 3))
    plt.plot(t, y)
    plt.title("Waveform")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.tight_layout()
    plt.savefig(out_dir / "waveform.png", dpi=200)
    plt.close()

    # 2) 3) 4) 全区間FFTによる 振幅/位相/パワー スペクトル
    f, A, phase, P = compute_one_sided_amplitude_phase_power(y, sr)

    # 振幅スペクトル
    plt.figure(figsize=(10, 3))
    plt.plot(f, A)
    plt.title("Single-Sided Amplitude Spectrum")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Amplitude")
    plt.xlim(0, 1500)
    plt.tight_layout()
    plt.savefig(out_dir / "amplitude_spectrum.png", dpi=200)
    plt.close()

    # 位相スペクトル（-pi..pi）
    plt.figure(figsize=(10, 3))
    plt.plot(f, phase)
    plt.title("Phase Spectrum")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Phase (rad)")
    plt.xlim(0, 1500)
    plt.tight_layout()
    plt.savefig(out_dir / "phase_spectrum.png", dpi=200)
    plt.close()

    # パワースペクトル
    plt.figure(figsize=(10, 3))
    plt.plot(f, P)
    plt.title("Power Spectrum")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Power (normalized)")
    plt.xlim(0, 1500)
    plt.tight_layout()
    plt.savefig(out_dir / "power_spectrum.png", dpi=200)
    plt.close()

    # 5) PSD（Welch法）
    # npersegはNFFT、noverlapはNFFT - HOP_SAMPLES相当
    nperseg = min(NFFT, len(y)) if len(y) > 0 else NFFT
    noverlap = max(0, nperseg - HOP_SAMPLES)
    win = signal.get_window(WINDOW, nperseg, fftbins=True)
    fw, psd = signal.welch(y - np.mean(y) if ZERO_MEAN else y,
                           fs=sr, window=win, nperseg=nperseg, noverlap=noverlap, detrend=False, return_onesided=True)
    plt.figure(figsize=(10, 3))
    plt.semilogy(fw, psd)  # PSDは対数表示が見やすい
    plt.title("Power Spectral Density (Welch)")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("PSD")
    plt.tight_layout()
    plt.savefig(out_dir / "psd_welch.png", dpi=200)
    plt.close()

    # 6) スペクトログラム（STFT magnitude, dB）
    f_stft, t_stft, Zxx = signal.stft(y - np.mean(y) if ZERO_MEAN else y,
                                      fs=sr, window=WINDOW, nperseg=NFFT,
                                      noverlap=max(0, NFFT - HOP_SAMPLES), nfft=NFFT, return_onesided=True)
    S_mag = np.abs(Zxx)
    S_db = 20 * np.log10(np.maximum(S_mag, np.finfo(float).eps))
    plt.figure(figsize=(10, 4))
    plt.pcolormesh(t_stft, f_stft, S_db, shading="auto")
    plt.title("Spectrogram (STFT magnitude, dB)")
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    plt.ylim(350, 550)
    cbar = plt.colorbar()
    cbar.set_label("Magnitude (dB)")
    plt.tight_layout()
    plt.savefig(out_dir / "spectrogram.png", dpi=200)
    plt.close()

    # サマリ出力
    print(f"input_video: {in_path}")
    print(f"output_dir:  {out_dir.resolve()}")
    print(f"sample_rate_hz: {sr}")
    print(f"duration_sec: {len(y)/sr:.6f}")
    print(f"saved:")
    print(" - waveform.png")
    print(" - amplitude_spectrum.png")
    print(" - phase_spectrum.png")
    print(" - power_spectrum.png")
    print(" - psd_welch.png")
    print(" - spectrogram.png")
    print(f" - audio_extracted.wav")
    print(f"FFT/STFT params: NFFT={NFFT}, hop={HOP_SAMPLES}, window='{WINDOW}'")

if __name__ == "__main__":
    main()
