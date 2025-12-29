import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import stft, resample
from ultralytics import YOLO

# ====== 学習時(makeData_YOLO.py)と同じ設定 ======
OUT_W = 1024
OUT_H = 1024

NPERSEG = 256
NOVERLAP = 0          # hop = NPERSEG - NOVERLAP = 256
NFFT = 4096

DPI = 200
FIGSIZE = (OUT_W / DPI, OUT_H / DPI)

# 学習側は「実データの周波数」は0..fs/2を使いつつ、
# 表示だけ extent を 0..2000Hz に固定していました（makeData_YOLO.pyに合わせる）
DISPLAY_FREQ_MAX_HZ = 2000.0

# ====== 入力（推論したい音声） ======
# 長い音声(cleaned_audio.wav)から切り出して作る場合は start/end を指定
# WAV_PATH = r"sakana/sa/cleaned_audio.wav"
# START_SEC = 1.104
# END_SEC   = 1.344

# すでに切り出し済み wav（例: sa_denoised.wav）を使うなら：
WAV_PATH = r"takana/ta/ta_denoised.wav"
START_SEC = None
END_SEC   = None

# ====== YOLO学習済み重み ======
WEIGHTS_PATH = r"runs_02/classify/train/weights/best.pt"

# ====== 一時出力画像 ======
TMP_PNG = r"tmp_spec.png"


def read_wav_mono_float(path: str):
    fs, x = wavfile.read(path)
    if x.dtype.kind in "iu":
        x = x.astype(np.float32) / np.iinfo(x.dtype).max
    else:
        x = x.astype(np.float32)
    if x.ndim == 2:
        x = x.mean(axis=1)
    return fs, x


def slice_by_time(fs: int, x: np.ndarray, start_sec, end_sec):
    if start_sec is None or end_sec is None:
        return x
    s = max(0, int(round(start_sec * fs)))
    e = min(len(x), int(round(end_sec * fs)))
    if e <= s:
        raise ValueError(f"Invalid slice range: start={start_sec}, end={end_sec}, samples=({s},{e})")
    return x[s:e]


def wav_to_spec_png_like_training(wav_path: str, out_png_path: str, start_sec=None, end_sec=None):
    fs, x = read_wav_mono_float(wav_path)
    x = slice_by_time(fs, x, start_sec, end_sec)

    # makeData_YOLO.py と同じ stft 引数
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

    S_db = 20.0 * np.log10(np.abs(Zxx) + 1e-10)  # (freq_bins, time_frames)

    # 学習時と同じリサンプル（周波数は切らない）
    S_time = resample(S_db, OUT_W, axis=1)   # time_frames -> OUT_W
    S_img  = resample(S_time, OUT_H, axis=0) # freq_bins   -> OUT_H

    # 学習時と同じ描画（extent固定）
    plt.figure(figsize=FIGSIZE, dpi=DPI)
    plt.imshow(
        S_img,
        origin="lower",
        aspect="auto",
        extent=[0.0, 1.0, 0.0, DISPLAY_FREQ_MAX_HZ],
    )
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig(out_png_path, dpi=DPI, bbox_inches="tight", pad_inches=0)
    plt.close()

    return out_png_path


def main():
    if not os.path.exists(WAV_PATH):
        raise FileNotFoundError(WAV_PATH)
    if not os.path.exists(WEIGHTS_PATH):
        raise FileNotFoundError(WEIGHTS_PATH)

    # 1) wav -> 学習時と同じ作り方のスペクトログラムpng
    wav_to_spec_png_like_training(WAV_PATH, TMP_PNG, START_SEC, END_SEC)

    # 2) YOLO分類推論
    # imgsz=1000 だと stride の倍数チェックで 1024 に更新される警告が出るため、
    # 最初から 1024 を指定（挙動はログの警告内容と一致します）
    model = YOLO(WEIGHTS_PATH)
    results = model.predict(source=TMP_PNG, imgsz=1024)

    r = results[0]
    top1_id = int(r.probs.top1)
    top1_conf = float(r.probs.top1conf)
    top1_name = r.names[top1_id]
    print(f"top1: {top1_name}  conf={top1_conf:.4f}")


if __name__ == "__main__":
    main()
