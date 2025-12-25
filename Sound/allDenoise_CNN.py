import numpy as np
import librosa
import noisereduce as nr
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image, ImageChops

# ======= 入力設定（あなたが埋める） =======
# 1つの発音ラベルにつき wav を7本（各wavに10発音＝tone_ranges 10個）入れる想定
# paths は同一フォルダ内に置くならファイル名だけでOK
FILE_CONFIGS = [
    {
        "label": "sa",
        "paths": [
            "sata_ML/sa.wav", "sata_ML2/sa.wav", "sata_ML3/sa.wav", "sata_ML4/sa.wav", "sata_ML5/sa.wav", "sata_ML7/sa.wav"
        ],
        "tone_ranges": [
            (1.0, 2.0), (2.0, 3.0), (3.0, 4.0), (4.0, 5.0), (5.0, 6.0),
            (6.0, 7.0), (7.0, 8.0), (8.0, 9.0), (9.0, 10.0), (10.0, 11.0),
        ], 
    },
    {
        "label": "sha",
        "paths": [
            "sata_ML/sha.wav", "sata_ML2/sha.wav", "sata_ML3/sha.wav", "sata_ML4/sha.wav", "sata_ML5/sha.wav", "sata_ML7/sha.wav"
        ],
        "tone_ranges": [
            (1.0, 2.0), (2.0, 3.0), (3.0, 4.0), (4.0, 5.0), (5.0, 6.0),
            (6.0, 7.0), (7.0, 8.0), (8.0, 9.0), (9.0, 10.0), (10.0, 11.0),
        ],
    },
    {
        "label": "tha",
        "paths": [
            "sata_ML/tha.wav", "sata_ML2/tha.wav", "sata_ML3/tha.wav", "sata_ML4/tha.wav", "sata_ML5/tha.wav", "sata_ML7/tha.wav"
        ],
        "tone_ranges": [
            (1.0, 2.0), (2.0, 3.0), (3.0, 4.0), (4.0, 5.0), (5.0, 6.0),
            (6.0, 7.0), (7.0, 8.0), (8.0, 9.0), (9.0, 10.0), (10.0, 11.0), 
        ],
    },
    {
        "label": "tya",
        "paths": [
            "sata_ML/tya.wav", "sata_ML2/tya.wav", "sata_ML3/tya.wav", "sata_ML4/tya.wav", "sata_ML5/tya.wav", "sata_ML7/tya.wav"
        ],
        "tone_ranges": [
            (1.0, 2.0), (2.0, 3.0), (3.0, 4.0), (4.0, 5.0), (5.0, 6.0),
            (6.0, 7.0), (7.0, 8.0), (8.0, 9.0), (9.0, 10.0), (10.0, 11.0),
        ],
    },
    {
        "label": "ta",
        "paths": [
            "sata_ML/ta.wav", "sata_ML2/ta.wav", "sata_ML3/ta.wav", "sata_ML4/ta.wav", "sata_ML5/ta.wav", "sata_ML7/ta.wav"
        ],
        "tone_ranges": [
            (1.0, 2.0), (2.0, 3.0), (3.0, 4.0), (4.0, 5.0), (5.0, 6.0),
            (6.0, 7.0), (7.0, 8.0), (8.0, 9.0), (9.0, 10.0), (10.0, 11.0),
        ],
    },
]
# =========================================

# ======= 出力設定 =======
OUTPUT_DIR = "ML_dataset_add2"   # ここに label/ 以下で画像保存
BAND_HIGH = 5000                     # fmax
# =======================

# ======= STFT設定（allDenoise_ML.py と同等の想定） =======
SR = None
N_FFT_STFT = 2048
HOP = 256
WINDOW = "hann"
ZERO_MEAN = True
# ========================================================

# ======= ML画像設定（allDenoise_ML.py の思想を流用） =======
ML_IMG_DPI = 200
ML_FIGSIZE = (20, 3)
ML_DB_FLOOR = -80.0   # 要件：床をもっと下まで（必要なら -100 などに）
ML_CMAP = "coolwarm"
# ===========================================================

def stft_mag(y, n_fft=N_FFT_STFT, hop=HOP, window=WINDOW):
    Z = librosa.stft(y, n_fft=n_fft, hop_length=hop, window=window, center=True)
    return np.abs(Z)

def _frames_from_time(M, fs, hop, t_start, t_end):
    n_frames = M.shape[1]
    frame_start = int(np.floor(t_start * fs / hop))
    frame_end   = int(np.ceil(t_end   * fs / hop))
    frame_start = max(0, frame_start)
    frame_end   = min(n_frames, frame_end)
    if frame_end <= frame_start:
        return 0, n_frames
    return frame_start, frame_end

def _cut_fmax(S_db, fs, fmax):
    if fmax is None:
        return S_db
    n_bins = S_db.shape[0]
    max_bin = int(np.floor((fmax / (fs / 2.0)) * (n_bins - 1))) + 1
    max_bin = max(1, min(max_bin, n_bins))
    return S_db[:max_bin, :]

def save_ml_spectrogram_image_global_ref(M, fs, hop, t_start, t_end, out_path, fmax, M_ref_global):
    """
    allDenoise_ML.py の save_ml_spectrogram_image をベースにしつつ、
    M_ref を「セグメント最大」ではなく「ラベル内(=7録音)の最大」に固定する版。
    """
    frame_start, frame_end = _frames_from_time(M, fs, hop, t_start, t_end)
    M_seg = M[:, frame_start:frame_end]

    eps = np.finfo(float).eps
    # ★ここが差分：M_seg.max() ではなく M_ref_global を使う
    S_db = 20.0 * np.log10(np.maximum(M_seg / (M_ref_global + eps), eps))
    S_db = np.maximum(S_db, ML_DB_FLOOR)
    S_db = _cut_fmax(S_db, fs, fmax)

    # 余白なし保存（allDenoise_ML.pyの方式を踏襲）
    fig = plt.figure(figsize=ML_FIGSIZE, frameon=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(S_db, origin="lower", aspect="auto", cmap=ML_CMAP, interpolation="nearest")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(".tmp.png")
    fig.savefig(tmp_path, dpi=ML_IMG_DPI, bbox_inches="tight", pad_inches=0)
    plt.close(fig)

    # 念のため白背景クロップ
    im = Image.open(tmp_path).convert("RGB")
    bg = Image.new("RGB", im.size, (255, 255, 255))
    diff = ImageChops.difference(im, bg)
    bbox = diff.getbbox()
    if bbox is not None:
        im = im.crop(bbox)
    im.save(out_path)
    tmp_path.unlink(missing_ok=True)

def main():
    out_root = Path(OUTPUT_DIR)
    out_root.mkdir(parents=True, exist_ok=True)

    for cfg in FILE_CONFIGS:
        label = cfg["label"]
        paths = cfg["paths"]
        tone_ranges = cfg["tone_ranges"]

        label_dir = out_root / label
        label_dir.mkdir(parents=True, exist_ok=True)

        # ---- 1) 同一ラベルの7録音を読み込み→denoise→STFTを作り、最大値を集計 ----
        Ms = []           # 各wavのM_denoを保持（あとで画像保存に使う）
        fs_common = None
        max_list = []

        for p in paths:
            y, fs = librosa.load(p, sr=SR, mono=True)
            y = y.astype(np.float32)
            if ZERO_MEAN:
                y = y - np.mean(y)

            y_deno = nr.reduce_noise(y=y, sr=fs, stationary=False)

            M = stft_mag(y_deno, n_fft=N_FFT_STFT, hop=HOP, window=WINDOW)
            Ms.append((Path(p).stem, M, fs))

            if fs_common is None:
                fs_common = fs
            elif fs_common != fs:
                raise ValueError(f"Sampling rate mismatch in label={label}: {fs_common} vs {fs} (file={p})")

            # tone_ranges の範囲だけ見て最大を取る（無音区間の影響を減らす）
            for (t_start, t_end) in tone_ranges:
                frame_start, frame_end = _frames_from_time(M, fs, HOP, t_start, t_end)
                M_seg = M[:, frame_start:frame_end]
                if BAND_HIGH is not None:
                    # fmax相当のbinだけを対象にした最大値
                    n_bins = M_seg.shape[0]
                    max_bin = int(np.floor((BAND_HIGH / (fs / 2.0)) * (n_bins - 1))) + 1
                    max_bin = max(1, min(max_bin, n_bins))
                    M_seg = M_seg[:max_bin, :]
                max_list.append(float(np.max(M_seg)))

        if not max_list:
            raise RuntimeError(f"No segments found for label={label} (check tone_ranges).")

        M_ref_global = float(np.max(max_list))
        if M_ref_global <= 0:
            raise RuntimeError(f"M_ref_global <= 0 for label={label}. (maybe silent?)")

        print(f"[{label}] global_ref(max over 6 wavs * 10 segs, fmax={BAND_HIGH}) = {M_ref_global:.6e}")

                # ---- 2) 共通M_ref_globalで、(len(paths)*10)枚を保存 ----
        n_seg = len(tone_ranges)

        for wav_i, (stem, M, fs) in enumerate(Ms):
            for seg_idx, (t_start, t_end) in enumerate(tone_ranges, start=1):
                global_seg = wav_i * n_seg + seg_idx  # 1..60（6wavの場合）
                out_path = label_dir / f"{stem}_seg{global_seg:02d}_add.png"

                save_ml_spectrogram_image_global_ref(
                    M, fs=fs, hop=HOP,
                    t_start=t_start, t_end=t_end,
                    out_path=out_path,
                    fmax=BAND_HIGH,
                    M_ref_global=M_ref_global
                )
        
    print("DONE:", Path(OUTPUT_DIR).resolve())

if __name__ == "__main__":
    main()
