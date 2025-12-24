from pathlib import Path

import numpy as np
import librosa
import noisereduce as nr
import soundfile as sf  # pip install soundfile

# 追加（画像生成）
import matplotlib.pyplot as plt
from PIL import Image, ImageChops

# ======================
# 設定（ここだけ変えればOK）
# ======================
input_path = r"sakana/sa/cleaned_audio.wav"
output_wav_path = r"sakana/sa/sa_denoised2.wav"

# ★追加：推論用スペクトログラム画像の出力先
output_img_dir = r"./_predict_images"

SR = None       # 元のサンプリング周波数を保持するなら None。固定したいなら例: 16000
ZERO_MEAN = True

# ★追加：教師データ生成と同じSTFT/画像パラメータ（allDenoise_ML.py準拠）
BAND_HIGH   = 5000
N_FFT_STFT  = 2048
HOP         = 256
WINDOW      = "hann"

ML_IMG_DPI  = 200
ML_FIGSIZE  = (20, 3)
ML_DB_FLOOR = -80.0
ML_CMAP     = "coolwarm"

# ★追加：1秒まるごと画像化するならこのまま
# （必要なら 0.55～0.85 のように変更してください）
tone_ranges = [(0.844, 1.344)]  # (開始秒, 終了秒) のリスト
# ======================


def stft_mag(y, n_fft=N_FFT_STFT, hop=HOP, window=WINDOW):
    """denoised波形から STFT振幅 |Z| を作る（教師データ側と同じ）"""
    Z = librosa.stft(y, n_fft=n_fft, hop_length=hop, window=window, center=True)
    M = np.abs(Z)
    return M


def save_ml_spectrogram_image(M, fs, hop, t_start, t_end, out_path, fmax=None):
    """
    allDenoise_ML.py の save_ml_spectrogram_image と同等：
    - 文字（タイトル/軸/目盛/カラーバー）なし
    - ヒートマップのみ
    - 余白が残った場合は白背景を自動クロップ
    """
    n_frames = M.shape[1]
    frame_start = int(np.floor(t_start * fs / hop))
    frame_end   = int(np.ceil(t_end   * fs / hop))
    frame_start = max(0, frame_start)
    frame_end   = min(n_frames, frame_end)
    if frame_end <= frame_start:
        M_seg = M
    else:
        M_seg = M[:, frame_start:frame_end]

    eps = np.finfo(float).eps
    M_ref = M_seg.max()
    S_db = 20.0 * np.log10(np.maximum(M_seg / (M_ref + eps), eps))
    S_db = np.maximum(S_db, ML_DB_FLOOR)

    # 周波数上限で「行」をカット（教師データ側の挙動に合わせる）
    if fmax is not None:
        n_bins = S_db.shape[0]
        max_bin = int(np.floor((fmax / (fs / 2.0)) * (n_bins - 1))) + 1
        max_bin = max(1, min(max_bin, n_bins))
        S_db = S_db[:max_bin, :]

    # 余白が出にくい全面描画
    fig = plt.figure(figsize=ML_FIGSIZE, frameon=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)

    ax.imshow(
        S_db,
        origin="lower",
        aspect="auto",
        cmap=ML_CMAP,
        interpolation="nearest"
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(".tmp.png")
    fig.savefig(tmp_path, dpi=ML_IMG_DPI, bbox_inches="tight", pad_inches=0)
    plt.close(fig)

    # 白背景の余白を自動クロップ
    im = Image.open(tmp_path).convert("RGB")
    bg = Image.new("RGB", im.size, (255, 255, 255))
    diff = ImageChops.difference(im, bg)
    bbox = diff.getbbox()
    if bbox is not None:
        im = im.crop(bbox)
    im.save(out_path)
    tmp_path.unlink(missing_ok=True)


def main():
    # ---- 音声読み込み ----
    y, fs = librosa.load(input_path, sr=SR, mono=True)  # sr=Noneで元のfs保持
    y = y.astype(np.float32)

    if ZERO_MEAN:
        y = y - np.mean(y)

    print(f"Input: {Path(input_path).resolve()}")
    print(f"Fs = {fs}, length = {len(y)} samples ({len(y) / fs:.2f} s)")

    # ---- ノイズ除去 ----
    y_deno = nr.reduce_noise(y=y, sr=fs, stationary=False)

    # ---- wav保存（既存機能）----
    Path(output_wav_path).parent.mkdir(parents=True, exist_ok=True)
    sf.write(output_wav_path, y_deno, fs, subtype="PCM_16")
    print(f"Saved wav: {Path(output_wav_path).resolve()}")

    # ---- ★追加：推論用スペクトログラム画像を保存 ----
    M_deno = stft_mag(y_deno, n_fft=N_FFT_STFT, hop=HOP, window=WINDOW)

    out_img_dir = Path(output_img_dir)
    out_img_dir.mkdir(parents=True, exist_ok=True)

    wav_stem = Path(input_path).stem
    for i, (t_start, t_end) in enumerate(tone_ranges, start=1):
        out_png = out_img_dir / f"{wav_stem}_seg{i:02d}.png"
        save_ml_spectrogram_image(
            M_deno, fs=fs, hop=HOP,
            t_start=t_start, t_end=t_end,
            out_path=out_png,
            fmax=BAND_HIGH
        )
        print(f"Saved png: {out_png.resolve()}")


if __name__ == "__main__":
    main()
