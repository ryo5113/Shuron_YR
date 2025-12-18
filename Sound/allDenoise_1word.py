import numpy as np
import matplotlib.pyplot as plt
import librosa
import soundfile as sf
import noisereduce as nr
from pathlib import Path

plt.close('all')

# ========= 設定 =========
# 解析したいファイルと有音区間（秒）をここに列挙
# tone_ranges に 3 区間分 [ (start1, end1), (start2, end2), (start3, end3) ]
FILE_CONFIGS = [
    # {
    #     "path": "sakana.wav",
    #     "tone_ranges": [
    #         (1.104, 1.344),   # 1回目
    #     ],
    #     "label": "sa",
    # },
    {
        "path": "sa_ta/sa/cleaned_audio.wav",
        "tone_ranges": [
            (0.0, 1.0),   # 1回目
        ],
        "label": "sa",
    },
    {
    
        "path": "sa_ta/ta/cleaned_audio.wav",
        "tone_ranges": [
            (0.0, 1.0),   # 1回目
        ],
        "label": "ta",
    }
    # {
    #     "path": "takana.wav",
    #     "tone_ranges": [
    #         (0.887, 1.108),   # 2回目
    #     ],
    #     "label": "ta",
    # }
]

OUTPUT_DIR  = "saVSta(1sound)"  # すべての結果をまとめるフォルダ

#NOISE_PATH = "recordedSound_20251212_202635.wav"  # ノイズ参照音声ファイル

# FFT表示帯域（Hz）
BAND_HIGH   = 3000

# STFTパラメータ
SR          = None      # None=元サンプリングのまま
N_FFT_STFT  = 2048
HOP         = 256
WINDOW      = "hann"

# 振動スペクトル用 FFT 点数（Noneならトーン長から自動）
N_FFT_SPEC  = None

# 波形の平均値を0にするかどうか
ZERO_MEAN   = True

# 振幅の閾値(>= この値を 1 とする)
THRESH_AMP = 0.01
# ========================


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
    """
    振動スペクトル用:
    - 指定 N_fft 点でFFTを計算
    - 短ければゼロ埋め，長ければ先頭 N_fft を使用
    - Hann窓＋片側振幅スペクトルに正規化
    """
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
    """振幅スペクトルを png で保存"""
    plt.figure(figsize=(20,3))
    plt.plot(freq, amp)
    plt.xlabel("Frequency [Hz]")
    plt.xlim(0, BAND_HIGH)
    plt.ylabel("Amplitude")
    plt.ylim(0, 0.025)  # 必要に応じて調整
    plt.title(title, fontsize=17)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def stft_mag(y, n_fft=N_FFT_STFT, hop=HOP, window=WINDOW):
    """STFT を計算し，複素スペクトルと振幅スペクトルを返す"""
    Z = librosa.stft(y, n_fft=n_fft, hop_length=hop, window=window, center=True)
    M = np.abs(Z)
    return Z, M


def estimate_segment_durations_from_spectrogram(M, fs, hop, energy_threshold_ratio=0.01, min_segment_frames=2):
    """スペクトログラムの振幅 M から、有音区間(連続した高エネルギーフレーム)の
    開始・終了時刻と継続時間を推定して返す。
    """
    if M.size == 0:
        return []

    # フレームごとのエネルギー（全周波数の絶対値の和）を計算
    frame_energy = np.sum(M, axis=0)
    max_energy = frame_energy.max()
    if max_energy <= 0:
        return []

    thresh = max_energy * float(energy_threshold_ratio)
    active = frame_energy >= thresh  # True: 有音フレーム

    segments = []
    dt = hop / float(fs)

    in_segment = False
    start_idx = 0

    for i, flag in enumerate(active):
        if flag and not in_segment:
            # 有音区間開始
            in_segment = True
            start_idx = i
        elif not flag and in_segment:
            # 有音区間終了
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

    # 末尾が有音で終わる場合の処理
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


def plot_spectrogram_rel(M, fs, hop, n_fft, title, out_path, fmax=None, M_ref=None):
    """
    振幅 M から相対dB（録音内での最大値基準）スペクトログラムを作成して保存
    """
    eps = np.finfo(float).eps
    if M_ref is None:
        M_ref = M.max()  # フォールバック
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
    plt.ylabel("Frequency [Hz]")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_spectrogram_rel_segment(M, fs, hop, t_start, t_end, title, out_path, fmax=None, M_ref=None):
    """
    ② M から [t_start, t_end] 秒に対応するフレームのみ切り出してスペクトログラム表示
    （時間軸は 0 から切り出し長さまでになる）
    """
    n_frames = M.shape[1]
    frame_start = int(np.floor(t_start * fs / hop))
    frame_end   = int(np.ceil(t_end   * fs / hop))
    frame_start = max(0, frame_start)
    frame_end   = min(n_frames, frame_end)
    if frame_end <= frame_start:
        # 区間がおかしい場合は全体を出しておく
        M_seg = M
    else:
        M_seg = M[:, frame_start:frame_end]

    eps = np.finfo(float).eps
    if M_ref is None:
        M_ref = M.max()  # フォールバック
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
    plt.ylabel("Frequency [Hz]", fontsize=22)
    plt.tick_params(axis="x", labelsize=20)
    plt.tick_params(axis="y", labelsize=17)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_and_save_amp_spectrum(freq, amp, title, out_path):
    """
    ③ FFT結果を 1/0 に二値化して保存
    - 閾値: amp.max() * BINARY_THRESH_RATIO
    """
    if amp.size == 0:
        return
    thresh = THRESH_AMP
    amp = (amp >= thresh).astype(int)

    plt.figure(figsize=(8,15))
    plt.step(freq, amp, where="mid")
    plt.xlabel("Frequency [Hz]")
    plt.xlim(0, BAND_HIGH)
    plt.ylabel("Binary")
    plt.yticks([0, 1])
    plt.ylim(-0.2, 1.2)
    plt.title(title + f"\n(thresh = {thresh:.4f})", fontsize=14)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

    return amp  # ④ 用に返しておく


def main():
    base_out_dir = Path(OUTPUT_DIR)
    base_out_dir.mkdir(parents=True, exist_ok=True)

    # 重ね描き用に、(freq, amp_deno, label) を貯める（1区間目のみ）
    overlay_data = []
    per_segment_overlay = {}   # 既存: 各segごとに全ファイルのFFT
    per_segment_binary  = {}   # ④用: 各segごとに全ファイルのbinary FFT

    for cfg in FILE_CONFIGS:
        input_path = cfg["path"]
        tone_ranges = cfg["tone_ranges"]
        label   = cfg.get("label", Path(input_path).stem)

        print("====", input_path, "====")

        out_dir = base_out_dir / label
        out_dir.mkdir(parents=True, exist_ok=True)

        # ---- 音声読み込み ----
        y, fs = librosa.load(input_path, sr=SR, mono=True)
        y = y.astype(np.float32)
        if ZERO_MEAN:
            y = y - np.mean(y)

        print(f"Input: {Path(input_path).resolve()}")
        print(f"Fs = {fs}, length = {len(y)} samples ({len(y)/fs:.2f} s)")

        #y_noise = extract_interval(y, fs, 0.0, 0.5)

        # ---- ノイズ除去 ----
        y_deno = nr.reduce_noise(y=y, sr=fs, stationary=False)

        plt.rcParams["font.size"] = 18

        # ---- 波形（元＋ノイズ除去後）----
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

        # ① このファイル内 3 区間の FFT を重ねるためのバッファ
        file_segment_fft = []      # [(freq, amp_deno, seg_num), ...]

        # ---- 有音部の振動スペクトル（3区間分）----
        for seg_idx, (t_start, t_end) in enumerate(tone_ranges):
            seg_num = seg_idx + 1  # 1, 2, 3 …

            y_tone_orig = extract_interval(y,     fs, t_start, t_end)
            y_tone_deno = extract_interval(y_deno, fs, t_start, t_end)

            out_wav = out_dir / f"{label}_denoised.wav"
            sf.write(out_wav, y_tone_deno, int(fs))

            if N_FFT_SPEC is None:
                N_fft_spec = max(len(y_tone_orig), len(y_tone_deno))
            else:
                N_fft_spec = int(N_FFT_SPEC)

            freq, amp_orig = compute_fft_fixed_N(y_tone_orig, fs, N_fft_spec)
            _,    amp_deno = compute_fft_fixed_N(y_tone_deno, fs, N_fft_spec)

            # ファイル名／タイトル：1区間目は従来名、それ以外は番号付き
            if seg_num == 1:
                orig_name = "fft_tone_original.png"
                deno_name = "fft_tone_denoised.png"
                orig_title = f"Amplitude Spectrum ({label} - seg1 Original)"
                deno_title = f"Amplitude Spectrum ({label})"
            else:
                orig_name = f"fft_tone{seg_num}_original.png"
                deno_name = f"fft_tone{seg_num}_denoised.png"
                orig_title = f"Amplitude Spectrum ({label} - seg{seg_num} Original)"
                deno_title = f"Amplitude Spectrum ({label} - analysis{seg_num})"

            plot_and_save_spectrum(
                freq,
                amp_orig,
                orig_title,
                out_dir / orig_name
            )
            plot_and_save_spectrum(
                freq,
                amp_deno,
                deno_title,
                out_dir / deno_name
            )

            # ① ファイル内 3 区間の重ね描き用
            file_segment_fft.append((freq, amp_deno, seg_num))

            # 既存の重ね描き用
            if seg_num == 1:
                overlay_data.append((freq, amp_deno, label))
            per_segment_overlay.setdefault(seg_idx, []).append((freq, amp_deno, label))

            # ③ この区間の二値化スペクトルを出力
            bin_title = f"Binary Spectrum ({label} - seg{seg_num})"
            bin_path = out_dir / f"fft_tone{seg_num}_denoised_binary.png"
            binary = plot_and_save_amp_spectrum(freq, amp_deno, bin_title, bin_path)

            # ④ 縦並べ用に保存
            per_segment_binary.setdefault(seg_idx, []).append((freq, binary, label))

        # ① 各ファイルごとに「3区間FFT重ね描き」
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

        # ---- スペクトログラム（元・ノイズ除去後）----
        _, M_orig = stft_mag(y,     n_fft=N_FFT_STFT, hop=HOP, window=WINDOW)
        _, M_deno = stft_mag(y_deno, n_fft=N_FFT_STFT, hop=HOP, window=WINDOW)
        M_ref_global = M_orig.max()

        plot_spectrogram_rel(
            M_orig,
            fs=fs,
            hop=HOP,
            n_fft=N_FFT_STFT,
            title=f"Spectrogram - {label}",
            out_path=out_dir / "spectrogram_original.png",
            fmax=BAND_HIGH,
            M_ref=M_ref_global
        )
        plot_spectrogram_rel(
            M_deno,
            fs=fs,
            hop=HOP,
            n_fft=N_FFT_STFT,
            title=f"Spectrogram ({label})",
            out_path=out_dir / "spectrogram_denoised.png",
            fmax=BAND_HIGH,
            M_ref=M_ref_global
        )

        # ② 各 tone_range ごとにスペクトログラムを切り出して保存
        for seg_idx, (t_start, t_end) in enumerate(tone_ranges):
            seg_num = seg_idx + 1
            seg_title = f"Spectrogram ({label} - seg{seg_num})"
            seg_path  = out_dir / f"spectrogram_denoised_seg{seg_num}.png"
            plot_spectrogram_rel_segment(
                M_deno, fs=fs, hop=HOP,
                t_start=t_start, t_end=t_end,
                title=seg_title, out_path=seg_path,
                fmax=BAND_HIGH,
                M_ref=M_ref_global
            )

        # ---- スペクトログラムから有音区間(1回の発音)の時間長を推定 ----
        segments = estimate_segment_durations_from_spectrogram(M_deno, fs=fs, hop=HOP)
        if segments:
            durations = [seg["duration"] for seg in segments]
            mean_duration = float(np.mean(durations))
        else:
            durations = []
            mean_duration = 0.0

        # 結果をテキストファイルに保存
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

        # ---- ノイズ除去後の音声を保存 ----
        sf.write(out_dir / "cleaned_audio.wav", y_deno, int(fs))

        print("出力フォルダ:", out_dir.resolve())
        print(f"N_FFT_SPEC={N_fft_spec}, N_FFT_STFT={N_FFT_STFT}, HOP={HOP}, WINDOW='{WINDOW}'")

    # ---- 振動スペクトル（ノイズ除去後）の重ね描き（1区間目のみ）----
    if overlay_data:
        plt.figure(figsize=(20,3))
        for freq, amp, label in overlay_data:
            plt.plot(freq, amp, label=label)
        plt.xlabel("Frequency [Hz]")
        plt.xlim(0, BAND_HIGH)
        plt.ylabel("Amplitude")
        plt.ylim(0, 0.03)  # 必要に応じて変更
        plt.title("Amplitude Spectrum (sound)")
        plt.grid(True)
        plt.legend(fontsize=20)
        plt.tight_layout()
        plt.savefig(base_out_dir / "fft_tone_denoised_overlay.png", dpi=200)
        plt.close()

    # 既存: 各segごとに「全ファイル」のFFTを重ねた図
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

    # ④ 各segごとに「バイナリFFTを縦に並べた図」
    for seg_idx, series in per_segment_binary.items():
        if not series:
            continue
        seg_num = seg_idx + 1
        n_files = len(series)
        fig, axes = plt.subplots(
            n_files, 1, sharex=True,
            figsize=(15, 3 * n_files)
        )
        if n_files == 1:
            axes = [axes]

        for ax, (freq, binary, label) in zip(axes, series):
            ax.step(freq, binary, where="mid")
            ax.set_xlim(0, BAND_HIGH)
            ax.tick_params(axis='x', labelsize=30)
            ax.set_ylim(-0.2, 1.2)
            ax.set_yticks([0, 1])
            ax.set_ylabel(label, fontsize=20)
            ax.tick_params(axis='y', labelsize=24)
            ax.grid(True)

        axes[-1].set_xlabel("Frequency [Hz]", fontsize=30)
        fig.suptitle(f"Binary Spectrum (all files) - seg{seg_num}", fontsize=25)
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        out_name = f"fft_tone_binary_overlay_seg{seg_num}_allFiles.png"
        fig.savefig(base_out_dir / out_name, dpi=200)
        plt.close(fig)


if __name__ == "__main__":
    main()
