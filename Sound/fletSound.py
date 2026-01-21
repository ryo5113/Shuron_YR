# fletSound_multi_label.py
# 元: fletSound.py（録音→raw保存→ノイズ処理→分割）をベースに
#  - ラベル別「録音Start」ボタンを追加
#  - Stopボタンは共通で、直前に選択されたラベルへ保存

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import traceback

import flet as ft

import numpy as np
import sounddevice as sd
import soundfile as sf
import librosa
import noisereduce as nr
from pydub import AudioSegment

import voiceCutting  # 添付 voiceCutting.py と同じフォルダに置く想定


# ========= 設定 =========
SAMPLE_RATE = 48000
CHANNELS = 1
DTYPE = "int16"

TARGET_COUNT = 10

# ラベル一覧（必要に応じて編集）
LABELS = ["sakana", "shakana", "takana", "thakana", "tyakana"]


@dataclass
class AppState:
    subject_dir: Path | None = None
    raw_dir: Path | None = None
    clean_dir: Path | None = None
    learn_dir: Path | None = None

    current_label: str | None = None  # 直前に選択されたラベル

    is_recording: bool = False
    stream: sd.InputStream | None = None
    frames: list[np.ndarray] | None = None
    last_raw_wav: Path | None = None

def safe_subject_name(name: str) -> str:
    bad = ["\\", "/", ":", "*", "?", '"', "<", ">", "|"]
    for b in bad:
        name = name.replace(b, "_")
    return name.strip()

def now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def get_next_index(out_dir: Path) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    max_n = 0
    for p in out_dir.glob("*.wav"):
        try:
            n = int(p.stem)  # "12.wav" -> 12
            if n > max_n:
                max_n = n
        except ValueError:
            # "abc.wav" 等は無視
            pass
    return max_n + 1

def denoise_wav_to_path(in_wav: Path, out_wav: Path) -> tuple[np.ndarray, int]:
    # allDenoise.py と同等の流れ（sr=None, mono=True, reduce_noise, sf.write）
    y, fs = librosa.load(str(in_wav), sr=None, mono=True)
    y = y.astype(np.float32)
    y = y - np.mean(y)  # ZERO_MEAN 相当
    y_deno = nr.reduce_noise(y=y, sr=fs, stationary=False)
    y_deno = np.clip(y_deno, -1.0, 1.0).astype(np.float32)
    y_i16 = (y_deno * 32767.0).astype(np.int16)
    sf.write(str(out_wav), y_i16, int(fs), subtype="PCM_16")

    return y_deno, int(fs)

def split_cleaned_wav_to_folder(cleaned_wav: Path, out_dir: Path, target_count: int = 10, start_index: int = 1) -> int:
    # voiceCutting.py の構成をそのまま利用
    voiceCutting.TARGET_COUNT = int(target_count)

    # 前回あなたが調整して動いた閾値（必要ならここはあなたの設定値に合わせてください）
    voiceCutting.MIN_AVG_DBFS = -60.0
    voiceCutting.MIN_PEAK_DBFS = -40.0

    audio = AudioSegment.from_file(str(cleaned_wav))

    ranges = voiceCutting.detect_nonsilent(
        audio,
        min_silence_len=voiceCutting.MIN_SILENCE_LEN_MS,
        silence_thresh=voiceCutting.SILENCE_THRESH_DBFS,
    )

    out_dir.mkdir(parents=True, exist_ok=True)

    if not ranges:
        return 0

    refined = voiceCutting.postprocess_ranges(audio, ranges)

    saved = 0
    for start_ms, end_ms in refined:
        s = max(0, start_ms - voiceCutting.KEEP_SILENCE_MS)
        e = min(len(audio), end_ms + voiceCutting.KEEP_SILENCE_MS)
        chunk = audio[s:e]

        if voiceCutting.is_valid_chunk(chunk):
            saved += 1
            file_index = start_index + saved - 1
            out_path = out_dir / f"{file_index}.wav"
            chunk.export(str(out_path), format="wav", parameters=["-ac", "1", "-ar", "48000", "-acodec", "pcm_s16le"])
            if saved >= target_count:
                break

    return saved

def main(page: ft.Page):
    page.title = "録音→ノイズ処理→自動分割（複数ラベル）"
    page.window_width = 820
    page.window_height = 560

    state = AppState()

    subject_name = ft.TextField(label="被験者名（親フォルダ名）", width=520)
    status = ft.Text(value="未作成", selectable=True)
    paths_view = ft.Text(value="", selectable=True)

    def set_status(msg: str):
        status.value = msg
        page.update()

    def set_paths():
        if state.subject_dir is None:
            paths_view.value = ""
        else:
            paths_view.value = (
                f"親フォルダ: {state.subject_dir}\n"
                f"raw: {state.raw_dir}\n"
                f"clean: {state.clean_dir}\n"
                f"learn: {state.learn_dir}\n"
                f"現在ラベル: {state.current_label if state.current_label else '-'}\n"
                f"最終録音: {state.last_raw_wav if state.last_raw_wav else '-'}"
            )
        page.update()

    def on_create_folder(_):
        name = safe_subject_name(subject_name.value or "")
        if not name:
            set_status("被験者名が空です。")
            return

        base = Path.cwd()
        subject_dir = base / name
        raw_dir = subject_dir / "raw"
        clean_dir = subject_dir / "clean"
        learn_dir = subject_dir / "learn"

        subject_dir.mkdir(parents=True, exist_ok=True)
        raw_dir.mkdir(parents=True, exist_ok=True)
        clean_dir.mkdir(parents=True, exist_ok=True)
        learn_dir.mkdir(parents=True, exist_ok=True)

        state.subject_dir = subject_dir
        state.raw_dir = raw_dir
        state.clean_dir = clean_dir
        state.learn_dir = learn_dir

        set_status(f"作成しました: {subject_dir}")
        set_paths()

    def start_recording_for_label(label: str):
        if state.subject_dir is None or state.raw_dir is None or state.learn_dir is None:
            set_status("先に被験者フォルダを作成してください。")
            return
        if state.is_recording:
            set_status("すでに録音中です。")
            return

        state.current_label = label
        state.frames = []
        state.is_recording = True

        def callback(indata, frames, time, status_):
            if status_:
                pass
            state.frames.append(indata.copy())

        try:
            stream = sd.InputStream(
                samplerate=SAMPLE_RATE,
                channels=CHANNELS,
                dtype=DTYPE,
                callback=callback,
            )
            stream.start()
            state.stream = stream
            set_status(f"録音中…（ラベル={label} / Stopで終了）")
            set_paths()
        except Exception as e:
            state.is_recording = False
            state.stream = None
            state.frames = None
            set_status(f"録音開始に失敗: {e}")

    def stop_recording(_):
        if not state.is_recording or state.stream is None or state.frames is None:
            set_status("録音中ではありません。")
            return
        if state.current_label is None:
            set_status("ラベルが未選択です（ラベルの録音Startボタンから開始してください）。")
            return

        # 録音停止
        try:
            state.stream.stop()
            state.stream.close()
        except Exception:
            pass

        state.stream = None
        state.is_recording = False

        # 保存（raw/タイムスタンプ.wav）
        stamp = now_stamp()
        raw_wav = state.raw_dir / f"{stamp}.wav"

        try:
            audio = np.concatenate(state.frames, axis=0)
            sf.write(str(raw_wav), audio, SAMPLE_RATE, subtype="PCM_16")
            state.last_raw_wav = raw_wav
            set_status(f"録音保存: {raw_wav.name}（ラベル={state.current_label}）")
            set_paths()
        except Exception as e:
            set_status(f"録音保存に失敗: {e}")
            return
        finally:
            state.frames = None

        # ③ ノイズ処理 + 分割（別スレッド）
        def run_postprocess():
            try:
                # learn/<label>/ を作る
                label_dir = state.learn_dir / state.current_label
                label_dir.mkdir(parents=True, exist_ok=True)

                clean_dir = state.clean_dir / state.current_label
                clean_dir.mkdir(parents=True, exist_ok=True)

                cleaned_wav = clean_dir / f"cleaned_audio_{stamp}.wav"
                denoise_wav_to_path(raw_wav, cleaned_wav)

                out_dir = label_dir
                start_index = get_next_index(out_dir)

                # learn/<label>/1/ に 1.wav..10.wav
                saved = split_cleaned_wav_to_folder(cleaned_wav, label_dir, target_count=TARGET_COUNT, start_index=start_index) 

                set_status(f"完了: ラベル={state.current_label} / 分割 {saved} 個を {label_dir} に保存")
                set_paths()
            except Exception as e:
                tb = traceback.format_exc()
                set_status(f"後処理に失敗: {e}\n{tb}")

        page.run_thread(run_postprocess)

    # ラベル別ボタン生成
    label_buttons = []
    for lbl in LABELS:
        label_buttons.append(
            ft.Button(
                content=ft.Text(f"{lbl} 録音開始"),
                on_click=lambda e, l=lbl: start_recording_for_label(l),
            )
        )

    page.add(
        ft.Column(
            [
                ft.Text("① 被験者フォルダ作成 → ② ラベル選択して録音Start → Stop（保存→③処理）", size=18, weight=ft.FontWeight.BOLD),
                subject_name,
                ft.Row([ft.Button(content=ft.Text("① フォルダ作成"), on_click=on_create_folder)]),
                ft.Divider(),
                ft.Text("② ラベル別 録音Start"),
                ft.Row(label_buttons, wrap=True, spacing=8),
                ft.Row([ft.Button(content=ft.Text("Stop（保存→③処理開始）"), on_click=stop_recording)]),
                ft.Divider(),
                ft.Text("状態:"),
                status,
                ft.Divider(),
                ft.Text("パス:"),
                paths_view,
            ],
            spacing=10,
        )
    )


if __name__ == "__main__":
    ft.app(target=main)
