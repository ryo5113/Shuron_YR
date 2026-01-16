# batch_resample_wavs.py
from pathlib import Path
import math
import numpy as np

import soundfile as sf
from scipy.signal import resample_poly


# ======================
# 設定（ここだけ編集）
# ======================
INPUT_DIR = r"takana_ct_chunks/voiced"         # 入力フォルダ（例：ラベル別フォルダを含むルート）
OUTPUT_DIR = r"takana_ct_chunks/voiced"   # 出力フォルダ
TARGET_SR = 48000                      # 変換先サンプリング周波数 [Hz]
SUBTYPE = "PCM_16"                     # 出力wavの量子化（例: PCM_16, PCM_24, FLOAT）
DRY_RUN = False                        # Trueで「変換せず一覧だけ表示」


def compute_up_down(sr_in: int, sr_out: int) -> tuple[int, int]:
    g = math.gcd(sr_in, sr_out)
    up = sr_out // g
    down = sr_in // g
    return up, down


def main():
    in_dir = Path(INPUT_DIR)
    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    wav_paths = sorted(in_dir.rglob("*.wav"))
    if not wav_paths:
        raise RuntimeError(f"No wav files found in: {in_dir}")

    print(f"Found {len(wav_paths)} wav files.")
    print(f"Target SR = {TARGET_SR} Hz")
    print(f"Output dir = {out_dir}")

    for i, wav_path in enumerate(wav_paths, start=1):
        rel = wav_path.relative_to(in_dir)
        out_path = out_dir / rel
        out_path.parent.mkdir(parents=True, exist_ok=True)

        if DRY_RUN:
            print(f"[{i:04d}] {wav_path} -> {out_path}")
            continue

        x, sr_in = sf.read(wav_path, always_2d=True)  # shape: (n_samples, n_channels)
        # x is float64/float32 depending on file; we keep float for processing

        if sr_in != TARGET_SR:
            up, down = compute_up_down(sr_in, TARGET_SR)

            # resample along time axis (axis=0), keep channels
            y = resample_poly(x, up=up, down=down, axis=0)

        else:
            y = x

        # soundfile expects shape (n_samples, n_channels) for multi-ch
        sf.write(out_path, y, TARGET_SR, subtype=SUBTYPE)

        if (i % 50) == 0 or i == len(wav_paths):
            print(f"Processed {i}/{len(wav_paths)}...")

    print("Done.")


if __name__ == "__main__":
    main()
