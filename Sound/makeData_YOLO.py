import os
import glob
import numpy as np
import tkinter as tk
from tkinter import filedialog

import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import stft, resample


# ====== 出力画像サイズ（ピクセル） ======
OUT_W = 1000
OUT_H = 1000

# ====== STFT設定 ======
# nperseg は scipy.signal.stft のデフォルトが 256（=窓長256）です。:contentReference[oaicite:0]{index=0}
NPERSEG = 256
# hop=256 にするため noverlap=0（hop = nperseg - noverlap）:contentReference[oaicite:1]{index=1}
NOVERLAP = 0
# FFT長
NFFT = 4096

# ====== 画像保存設定 ======
DPI = 200
FIGSIZE = (OUT_W / DPI, OUT_H / DPI)  # inch


def read_wav_mono_float(path: str):
    fs, x = wavfile.read(path)  # :contentReference[oaicite:2]{index=2}
    # int PCM -> float32 [-1, 1]
    if x.dtype.kind in "iu":
        x = x.astype(np.float32) / np.iinfo(x.dtype).max
    else:
        x = x.astype(np.float32)
    # stereo -> mono
    if x.ndim == 2:
        x = x.mean(axis=1)
    return fs, x


def pick_root_dir() -> str:
    root = tk.Tk()
    root.withdraw()
    folder = filedialog.askdirectory(title="ラベル分け済み音声データのルートフォルダを選択")  # :contentReference[oaicite:3]{index=3}
    root.destroy()
    return folder


def main():
    root_dir = pick_root_dir()
    if not root_dir:
        print("キャンセルされました。")
        return

    # ラベル = ルート直下のサブフォルダ名（その中のwavを対象にする）
    label_dirs = [d for d in glob.glob(os.path.join(root_dir, "*")) if os.path.isdir(d)]
    if not label_dirs:
        print("ルート直下にラベルフォルダが見つかりません。")
        return

    # wav収集（ラベルごと）
    label_to_wavs = {}
    for ld in label_dirs:
        label = os.path.basename(ld)
        wavs = sorted(glob.glob(os.path.join(ld, "**", "*.wav"), recursive=True))
        if wavs:
            label_to_wavs[label] = wavs

    if not label_to_wavs:
        print("ラベルフォルダ配下に .wav が見つかりません。")
        return

    # 全wavの最長durationを計算
    max_duration = 0.0
    total_files = 0
    for wavs in label_to_wavs.values():
        for p in wavs:
            fs, x = read_wav_mono_float(p)
            dur = len(x) / fs
            max_duration = max(max_duration, dur)
            total_files += 1

    out_root = os.path.join(root_dir, "spectrogram_normmax_by_label")
    os.makedirs(out_root, exist_ok=True)

    print(f"対象ファイル数: {total_files}")
    print(f"最長duration(max_duration): {max_duration:.6f} sec")
    print(f"出力先: {out_root}")

    # 作成
    for label, wavs in label_to_wavs.items():
        out_label_dir = os.path.join(out_root, label)
        os.makedirs(out_label_dir, exist_ok=True)

        for p in wavs:
            fs, x = read_wav_mono_float(p)

            # STFT（tは秒単位）:contentReference[oaicite:4]{index=4}
            f, t, Zxx = stft(
                x,
                fs=fs,
                window="hann",
                nperseg=NPERSEG,
                noverlap=NOVERLAP,
                nfft=NFFT,
                boundary=None,
                padded=False,
                return_onesided=True,
                detrend=False,
                scaling="spectrum",
            )

            S = np.abs(Zxx)
            S_db = 20.0 * np.log10(S + 1e-10)  # (freq_bins, time_frames)

            # 時間を「最長音声基準」で0〜1へ正規化（最長音声の終端=1）
            t_norm = t / max_duration

            # 4000×1000 に“等間隔サンプル”としてリサンプル
            # - 時間方向: OUT_W 列（0..1 を等間隔）
            # - 周波数方向: OUT_H 行（0..fs/2 を等間隔）
            S_time = resample(S_db, OUT_W, axis=1)   # time_frames -> OUT_W
            S_img  = resample(S_time, OUT_H, axis=0) # freq_bins   -> OUT_H

            # 画像化（軸や文字は学習用途を想定して省略）
            plt.figure(figsize=FIGSIZE, dpi=DPI)
            plt.imshow(
                S_img,
                origin="lower",
                aspect="auto",
                extent=[0.0, 1.0, 0.0, 2000.0],  # freq軸は0〜3000Hzに固定
            )
            plt.axis("off")
            plt.tight_layout(pad=0)

            base = os.path.splitext(os.path.basename(p))[0]
            out_path = os.path.join(out_label_dir, f"{base}.png")
            plt.savefig(out_path, dpi=DPI, bbox_inches="tight", pad_inches=0)
            plt.close()

    print("完了しました。")


if __name__ == "__main__":
    main()
