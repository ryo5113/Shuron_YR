import numpy as np
import matplotlib.pyplot as plt
import librosa
import soundfile as sf
import noisereduce as nr
from pathlib import Path
from PIL import Image, ImageChops

plt.close('all')

# ========= 設定 =========
FILE_CONFIGS = [
    {
        "path": "sa.wav",
        "tone_ranges": [
            (1.1, 1.6),   # 1回目
            (2.1, 2.6),   # 2回目
            (3.0, 3.5),   # 3回目
            (4.0, 4.5),   # 4回目
            (5.0, 5.5),   # 5回目
            (6.0, 6.5),   # 6回目
            (7.0, 7.5),   # 7回目
            (8.0, 8.5),   # 8回目
            (9.0, 9.5),   # 9回目
            (10.0, 10.5),   # 10回目
        ],
        "label": "sa",
    },
    {
        "path": "sha.wav",
        "tone_ranges": [
            (1.1, 1.6),   # 1回目
            (2.05, 2.55),   # 2回目
            (3.0, 3.5),   # 3回目
            (4.0, 4.5),   # 4回目
            (5.0, 5.5),   # 5回目
            (6.0, 6.5),   # 6回目
            (7.0, 7.5),   # 7回目
            (8.0, 8.5),   # 8回目
            (9.0, 9.5),   # 9回目
            (10.0, 10.5),   # 10回目
        ],
        "label": "sha",
    },
    {
        "path": "tha.wav",
        "tone_ranges": [
            (1.15, 1.65),   # 1回目
            (2.15, 2.65),   # 2回目
            (3.1, 3.6),   # 3回目
            (4.1, 4.6),   # 4回目
            (5.1, 5.6),   # 5回目
            (6.1, 6.6),   # 6回目
            (7.1, 7.6),   # 7回目
            (8.1, 8.6),   # 8回目
            (9.1, 9.6),   # 9回目
            (10.1, 10.6),   # 10回目
        ],
        "label": "tha",
    },
    {
        "path": "tya.wav",
        "tone_ranges": [
            (1.1, 1.6),   # 1回目
            (2.1, 2.6),   # 2回目
            (3.0, 3.5),   # 3回目
            (4.0, 4.5),   # 4回目
            (5.0, 5.5),   # 5回目
            (6.0, 6.5),   # 6回目
            (7.0, 7.5),   # 7回目
            (8.0, 8.5),   # 8回目
            (9.0, 9.5),   # 9回目
            (10.0, 10.5),   # 10回目
        ],
        "label": "tya",
    },
    {
        "path": "ta.wav",
        "tone_ranges": [
            (1.1, 1.6),   # 1回目
            (2.05, 2.55),   # 2回目
            (3.0, 3.5),   # 3回目
            (4.0, 4.5),   # 4回目
            (5.0, 5.5),   # 5回目
            (6.0, 6.5),   # 6回目
            (7.0, 7.5),   # 7回目
            (8.0, 8.5),   # 8回目
            (9.0, 9.5),   # 9回目
            (10.0, 10.5),   # 10回目
        ],
        "label": "ta",
    }
]

OUTPUT_DIR  = "sata_ML3"

BAND_HIGH   = 3000

SR          = None
N_FFT_STFT  = 2048
HOP         = 256
WINDOW      = "hann"

N_FFT_SPEC  = None
ZERO_MEAN   = True

THRESH_AMP = 0.1
# ========================

# === ML DATASET 出力設定（追加）=========================================
ML_DATASET_DIRNAME = "_ml_dataset"   # OUTPUT_DIR配下に作る
ML_IMG_DPI = 200                    # 既存出力と合わせるなら200
ML_FIGSIZE = (20, 3)                # 既存seg画像と合わせるなら(20,3)
ML_DB_FLOOR = -50.0
ML_CMAP = "coolwarm"
# ======================================================================

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
    plt.figure(figsize=(20,3))
    plt.plot(freq, amp)
    plt.xlabel("Frequency [Hz]", fontsize=20)
    plt.xlim(0, BAND_HIGH)
    plt.ylabel("Amplitude", fontsize=20)
    plt.ylim(0, 0.01)
    plt.title(title, fontsize=20)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

def stft_mag(y, n_fft=N_FFT_STFT, hop=HOP, window=WINDOW):
    Z = librosa.stft(y, n_fft=n_fft, hop_length=hop, window=window, center=True)
    M = np.abs(Z)
    return Z, M

def estimate_segment_durations_from_spectrogram(M, fs, hop, energy_threshold_ratio=0.01, min_segment_frames=2):
    if M.size == 0:
        return []

    frame_energy = np.sum(M, axis=0)
    max_energy = frame_energy.max()
    if max_energy <= 0:
        return []

    thresh = max_energy * float(energy_threshold_ratio)
    active = frame_energy >= thresh

    segments = []
    dt = hop / float(fs)

    in_segment = False
    start_idx = 0

    for i, flag in enumerate(active):
        if flag and not in_segment:
            in_segment = True
            start_idx = i
        elif not flag and in_segment:
            end_idx = i - 1
            length = end_idx - start_idx + 1
            if length >= min_segment_frames:
                start_time = start_idx * dt
                end_time = (end_idx + 1) * dt
                segments.append(
                    {
                        "start_frame": start_idx,
                        "end_frame": end_idx,
                        "start_time": start_time,
                        "end_time": end_time,
                        "duration": end_time - start_time,
                    }
                )
            in_segment = False

    if in_segment:
        end_idx = len(active) - 1
        length = end_idx - start_idx + 1
        if length >= min_segment_frames:
            start_time = start_idx * dt
            end_time = (end_idx + 1) * dt
            segments.append(
                {
                    "start_frame": start_idx,
                    "end_frame": end_idx,
                    "start_time": start_time,
                    "end_time": end_time,
                    "duration": end_time - start_time,
                }
            )

    return segments

def plot_spectrogram_rel(M, fs, hop, n_fft, title, out_path, fmax=None):
    eps = np.finfo(float).eps
    M_ref = M.max()
    S_db = 20.0 * np.log10(np.maximum(M / (M_ref + eps), eps))
    DB_FLOOR = -50.0
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
    cbar.set_label("Relative Intensity(dB)")
    plt.xlabel("Time [s]")
    plt.ylabel("Frequency[Hz]")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

def plot_spectrogram_rel_segment(M, fs, hop, t_start, t_end, title, out_path, fmax=None):
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
    DB_FLOOR = -50.0
    S_db = np.maximum(S_db, DB_FLOOR)

    plt.figure(figsize=(20,3))
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
    cbar.set_label("Relative Intensity(dB)", fontsize=17)
    plt.xlabel("Time [s]", fontsize=22)
    plt.ylabel("Frequency[Hz]", fontsize=22)
    plt.tick_params(axis="x", labelsize=20)
    plt.tick_params(axis="y", labelsize=17)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

# === 教師データ用（ML用）スペクトログラム画像保存（追加）==================
def save_ml_spectrogram_image(M, fs, hop, t_start, t_end, out_path, fmax=None):
    """
    教師データ用のPNGを作る。
    - 画像内に文字（タイトル/軸ラベル/目盛/カラーバー）を一切入れない
    - ヒートマップ領域だけを保存（bbox_inches='tight', pad_inches=0）
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

    # ---- 周波数上限(fmax)を行方向で「実データとして」カットしてから描画 ----
    if fmax is not None:
        n_bins = S_db.shape[0]
        # Mはrfft相当で0..fs/2がn_binsに対応（線形）
        max_bin = int(np.floor((fmax / (fs / 2.0)) * (n_bins - 1))) + 1
        max_bin = max(1, min(max_bin, n_bins))
        S_db = S_db[:max_bin, :]

    # ---- 余白が出にくい imshow で全面描画（軸なし）----
    fig = plt.figure(figsize=ML_FIGSIZE, frameon=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])  # 画面全面
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

    # ---- 念のため：白背景だけを自動クロップ（残余白対策）----
    im = Image.open(tmp_path).convert("RGB")
    bg = Image.new("RGB", im.size, (255, 255, 255))
    diff = ImageChops.difference(im, bg)
    bbox = diff.getbbox()
    if bbox is not None:
        im = im.crop(bbox)
    im.save(out_path)
    tmp_path.unlink(missing_ok=True)
# ======================================================================

def main():
    base_out_dir = Path(OUTPUT_DIR)
    base_out_dir.mkdir(parents=True, exist_ok=True)

    # 追加：教師データ用ベースフォルダ
    ml_base_dir = base_out_dir / ML_DATASET_DIRNAME
    ml_base_dir.mkdir(parents=True, exist_ok=True)

    overlay_data = []
    per_segment_overlay = {}
    mean_overlay_data = []

    for cfg in FILE_CONFIGS:
        input_path = cfg["path"]
        tone_ranges = cfg["tone_ranges"]
        label   = cfg.get("label", Path(input_path).stem)

        print("====", input_path, "====")

        out_dir = base_out_dir / label
        out_dir.mkdir(parents=True, exist_ok=True)

        # 追加：教師データ用ラベルフォルダ
        ml_label_dir = ml_base_dir / label
        ml_label_dir.mkdir(parents=True, exist_ok=True)

        y, fs = librosa.load(input_path, sr=SR, mono=True)
        y = y.astype(np.float32)
        if ZERO_MEAN:
            y = y - np.mean(y)

        print(f"Input: {Path(input_path).resolve()}")
        print(f"Fs = {fs}, length = {len(y)} samples ({len(y)/fs:.2f} s)")

        y_deno = nr.reduce_noise(y=y, sr=fs, stationary=False)

        if N_FFT_SPEC is None:
            seg_lengths = []
            for (t_start, t_end) in tone_ranges:
                y_tmp = extract_interval(y_deno, fs, t_start, t_end)
                seg_lengths.append(len(y_tmp))
            N_fft_spec_file = int(max(seg_lengths)) if len(seg_lengths) > 0 else 0
        else:
            N_fft_spec_file = int(N_FFT_SPEC)
        if N_fft_spec_file <= 0:
            raise ValueError(f"N_fft_spec_file must be > 0, got {N_fft_spec_file} (label={label})")

        amp_deno_list = []
        freq_common = None

        plt.rcParams["font.size"] = 18

        t = np.arange(len(y)) / fs
        plt.figure()
        plt.plot(t, y, label="Original")
        plt.plot(t, y_deno, label="Denoised", alpha=0.75)
        plt.xlabel("Time [s]")
        plt.ylabel("Amplitude")
        plt.title(f"Waveform (Original / Denoised) - {label}", fontsize=17)
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / "waveform_orig_denoised.png", dpi=200)
        plt.close()

        file_segment_fft = []

        for seg_idx, (t_start, t_end) in enumerate(tone_ranges):
            seg_num = seg_idx + 1

            y_tone_orig = extract_interval(y,     fs, t_start, t_end)
            y_tone_deno = extract_interval(y_deno, fs, t_start, t_end)

            N_fft_spec = N_fft_spec_file

            freq, amp_orig = compute_fft_fixed_N(y_tone_orig, fs, N_fft_spec)
            _,    amp_deno = compute_fft_fixed_N(y_tone_deno, fs, N_fft_spec)

            if freq_common is None:
                freq_common = freq
            amp_deno_list.append(amp_deno)

            if seg_num == 1:
                orig_name = "fft_tone_original.png"
                deno_name = "fft_tone_denoised.png"
                orig_title = f"Amplitude Spectrum ({label} - seg1 Original)"
                deno_title = f"Amplitude Spectrum ({label} - analysis{seg_num})"
            else:
                orig_name = f"fft_tone{seg_num}_original.png"
                deno_name = f"fft_tone{seg_num}_denoised.png"
                orig_title = f"Amplitude Spectrum ({label} - seg{seg_num} Original)"
                deno_title = f"Amplitude Spectrum ({label} - analysis{seg_num})"

            plot_and_save_spectrum(freq, amp_orig, orig_title, out_dir / orig_name)
            plot_and_save_spectrum(freq, amp_deno, deno_title, out_dir / deno_name)

            file_segment_fft.append((freq, amp_deno, seg_num))

            if seg_num == 1:
                overlay_data.append((freq, amp_deno, label))
            per_segment_overlay.setdefault(seg_idx, []).append((freq, amp_deno, label))

        if len(amp_deno_list) > 0 and freq_common is not None:
            mean_amp = np.mean(np.stack(amp_deno_list, axis=0), axis=0)
            mean_title = f"Amplitude Spectrum (Mean) - {label}"
            plot_and_save_spectrum(freq_common, mean_amp, mean_title, out_dir / "fft_tone_mean_denoised.png")
            mean_overlay_data.append((freq_common, mean_amp, label))

        if file_segment_fft:
            plt.figure(figsize=(20,3))
            for freq, amp, seg_num in file_segment_fft:
                plt.plot(freq, amp, label=f"seg{seg_num}")
            plt.xlabel("Frequency [Hz]", fontsize=30)
            plt.xlim(0, BAND_HIGH)
            plt.ylabel("Amplitude", fontsize=30)
            plt.ylim(0, 0.04)
            plt.title(f"Amplitude Spectrum(All) - {label}")
            plt.grid(True)
            plt.tick_params(labelsize=24)
            plt.legend()
            plt.tight_layout()
            plt.savefig(out_dir / "fft_tone_denoised_allSegments.png", dpi=200)
            plt.close()

        _, M_orig = stft_mag(y,     n_fft=N_FFT_STFT, hop=HOP, window=WINDOW)
        _, M_deno = stft_mag(y_deno, n_fft=N_FFT_STFT, hop=HOP, window=WINDOW)

        plot_spectrogram_rel(
            M_orig, fs=fs, hop=HOP, n_fft=N_FFT_STFT,
            title=f"Spectrogram - {label}",
            out_path=out_dir / "spectrogram_original.png",
            fmax=BAND_HIGH
        )
        plot_spectrogram_rel(
            M_deno, fs=fs, hop=HOP, n_fft=N_FFT_STFT,
            title=f"Spectrogram ({label})",
            out_path=out_dir / "spectrogram_denoised.png",
            fmax=BAND_HIGH
        )

        # ② 既存：各 tone_range ごとにスペクトログラム画像を保存（タイトル等あり）
        for seg_idx, (t_start, t_end) in enumerate(tone_ranges):
            seg_num = seg_idx + 1
            seg_title = f"Spectrogram ({label} - seg{seg_num})"
            seg_path  = out_dir / f"spectrogram_denoised_seg{seg_num}.png"
            plot_spectrogram_rel_segment(
                M_deno, fs=fs, hop=HOP,
                t_start=t_start, t_end=t_end,
                title=seg_title, out_path=seg_path,
                fmax=BAND_HIGH
            )

        # ★追加：教師データ用（ML用）画像を保存（文字要素なし）
        #   ファイル名には label は入れてOK（画像中に入らないため）
        wav_stem = Path(input_path).stem
        START_OFFSET = 20
        for seg_idx, (t_start, t_end) in enumerate(tone_ranges):
            seg_num = (seg_idx + 1) + START_OFFSET
            ml_path = ml_label_dir / f"{wav_stem}_seg{seg_num:02d}.png"
            save_ml_spectrogram_image(
                M_deno, fs=fs, hop=HOP,
                t_start=t_start, t_end=t_end,
                out_path=ml_path,
                fmax=BAND_HIGH
            )

        segments = estimate_segment_durations_from_spectrogram(M_deno, fs=fs, hop=HOP)
        if segments:
            durations = [seg["duration"] for seg in segments]
            mean_duration = float(np.mean(durations))
        else:
            durations = []
            mean_duration = 0.0

        txt_path = out_dir / "segment_durations_from_spectrogram.txt"
        with open(txt_path, "w", encoding="utf-8") as f_txt:
            f_txt.write("# 有音区間(1回の発音)の推定時間 [スペクトログラム(denoised)ベース]\n")
            f_txt.write(f"# file: {input_path}\n")
            f_txt.write(f"fs = {fs}\n")
            f_txt.write(f"hop_length = {HOP}\n")
            f_txt.write(f"frame_duration_sec = {HOP/float(fs):.6f}\n")
            f_txt.write(f"num_segments = {len(segments)}\n")
            for i, seg in enumerate(segments, 1):
                f_txt.write(
                    f"segment_{i}: start_time={seg['start_time']:.6f}s, "
                    f"end_time={seg['end_time']:.6f}s, duration={seg['duration']:.6f}s\n"
                )
            f_txt.write(f"mean_duration = {mean_duration:.6f}s\n")

        sf.write(out_dir / "cleaned_audio.wav", y_deno, int(fs))

        print("出力フォルダ:", out_dir.resolve())
        print(f"N_FFT_STFT={N_FFT_STFT}, HOP={HOP}, WINDOW='{WINDOW}'")

    # ---- 以下、既存の重ね描き処理などは元のまま（省略なしで残してOK）----
    # （ここはあなたの元コードのままにしてあります）
    if overlay_data:
        plt.figure()
        for freq, amp, label in overlay_data:
            plt.plot(freq, amp, label=label)
        plt.xlabel("Frequency [Hz]")
        plt.xlim(0, BAND_HIGH)
        plt.ylabel("Amplitude")
        plt.ylim(0, 0.04)
        plt.title("Amplitude Spectrum (all) - seg1")
        plt.grid(True)
        plt.legend(fontsize=20)
        plt.tight_layout()
        plt.savefig(base_out_dir / "fft_tone_denoised_overlay.png", dpi=200)
        plt.close()

    for seg_idx, series in per_segment_overlay.items():
        if not series:
            continue
        plt.figure()
        for freq, amp, label in series:
            plt.plot(freq, amp, label=label)
        plt.xlabel("Frequency [Hz]")
        plt.xlim(0, BAND_HIGH)
        plt.ylabel("Amplitude")
        plt.ylim(0, 0.04)
        seg_num = seg_idx + 1
        plt.title(f"Amplitude Spectrum (all) - seg{seg_num}")
        plt.grid(True)
        plt.legend(fontsize=20)
        plt.tight_layout()
        out_name = f"fft_tone_denoised_overlay_seg{seg_num}_allFiles.png"
        plt.savefig(base_out_dir / out_name, dpi=200)
        plt.close()

    if mean_overlay_data:
        plt.figure(figsize=(20,3))
        for freq, amp, label in mean_overlay_data:
            plt.plot(freq, amp, label=label)
        plt.xlabel("Frequency [Hz]")
        plt.xlim(0, BAND_HIGH)
        plt.ylabel("Amplitude")
        plt.ylim(0, 0.03)
        plt.title("Amplitude Spectrum (mean) - all")
        plt.grid(True)
        plt.legend(fontsize=20)
        plt.tight_layout()
        plt.savefig(base_out_dir / "fft_tone_mean_denoised_overlay_allFiles.png", dpi=200)
        plt.close()

if __name__ == "__main__":
    main()
