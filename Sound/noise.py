#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path

import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
from scipy.signal import correlate, correlation_lags
from typing import Optional


# =========================
# ここを編集してパス指定
# =========================
NOISE_WAV_PATH = Path("ae/recordedSound_20251215_180213.wav")   # ノイズのみ
INPUT_WAV_PATH = Path("ae/recordedSound_20251215_180229.wav")   # 声入り（ノイズ込み）
OUTPUT_WAV_PATH = Path("ae/out_subtracted.wav")                 # 出力
PLOT_SAVE_PATH = Path("ae/waveform.png")                        # None にすると保存しない

SHOW_PLOT = True  # True: 画面表示 / False: 表示しない
# =========================


def ensure_mono(x: np.ndarray, name: str) -> np.ndarray:
    if x.ndim == 1:
        return x
    raise ValueError(f"{name} はモノラル(1ch)前提です（ステレオは未対応）。")


def estimate_lag_samples(signal: np.ndarray, noise: np.ndarray) -> int:
    """
    相互相関の最大からラグ（サンプル）を推定
    """
    corr = correlate(signal, noise, mode="full", method="fft")
    lags = correlation_lags(len(signal), len(noise), mode="full")
    return int(lags[np.argmax(corr)])


def build_aligned_noise(signal: np.ndarray, noise: np.ndarray, lag: int) -> np.ndarray:
    """
    可視化用：signalと同じ長さに整列した noise（はみ出しは0埋め）
    """
    aligned = np.zeros_like(signal, dtype=np.float32)

    if lag >= 0:
        n = min(len(noise), len(signal) - lag)
        if n > 0:
            aligned[lag:lag + n] = noise[:n]
    else:
        lag2 = -lag
        n = min(len(noise) - lag2, len(signal))
        if n > 0:
            aligned[:n] = noise[lag2:lag2 + n]

    return aligned


def subtract(signal: np.ndarray, aligned_noise: np.ndarray) -> np.ndarray:
    """
    time-domainでそのまま差し引き
    """
    return (signal.astype(np.float32) - aligned_noise.astype(np.float32))


def plot_waveforms(sr: int,
                   noise: np.ndarray,
                   signal: np.ndarray,
                   aligned_noise: np.ndarray,
                   output: np.ndarray,
                   lag: int,
                   save_path: Optional[Path]):
    fig = plt.figure(figsize=(12, 9))

    def _plot(ax_idx, y, title):
        ax = fig.add_subplot(4, 1, ax_idx)
        t = np.arange(len(y)) / sr
        ax.plot(t, y)  # matplotlib.pyplot.plot :contentReference[oaicite:2]{index=2}
        ax.set_title(title)
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Amplitude")

    _plot(1, noise, "Noise-only waveform")
    _plot(2, signal, "Signal waveform (voice + noise)")
    _plot(3, aligned_noise, f"Aligned noise (lag = {lag} samples = {lag/sr:.6f} s)")
    _plot(4, output, "Output waveform (signal - aligned_noise)")

    fig.tight_layout()

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(save_path), dpi=150)

    if SHOW_PLOT:
        plt.show()
    else:
        plt.close(fig)


def main():
    # WAV読み込み（soundfile.read） :contentReference[oaicite:3]{index=3}
    noise, sr_n = sf.read(str(NOISE_WAV_PATH))
    signal, sr = sf.read(str(INPUT_WAV_PATH))

    if sr != sr_n:
        raise ValueError(f"サンプルレート不一致: noise={sr_n}, input={sr}")

    noise = ensure_mono(noise, "noise")
    signal = ensure_mono(signal, "input")

    lag = estimate_lag_samples(signal, noise)  # correlation_lagsの仕様 :contentReference[oaicite:4]{index=4}
    aligned_noise = build_aligned_noise(signal, noise, lag)
    output = subtract(signal, aligned_noise)

    OUTPUT_WAV_PATH.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(OUTPUT_WAV_PATH), output, sr)  # soundfile.write :contentReference[oaicite:5]{index=5}

    print(f"done. lag={lag} samples ({lag/sr:.6f} sec)")
    print(f"output wav: {OUTPUT_WAV_PATH}")
    if PLOT_SAVE_PATH is not None:
        print(f"plot png:  {PLOT_SAVE_PATH}")

    plot_waveforms(sr, noise, signal, aligned_noise, output, lag, PLOT_SAVE_PATH)


if __name__ == "__main__":
    main()
