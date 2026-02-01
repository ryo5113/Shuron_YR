# 学習View: 録音→ノイズ処理→分割→(60Hzのみ)学習→モデル保存
# 推論View: モデル選択→録音→ノイズ処理→分割→推論（確率%表示）

from __future__ import annotations

import threading
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import json
import traceback
import re

import flet as ft

import numpy as np
import sounddevice as sd
import soundfile as sf
import librosa
import noisereduce as nr
from pydub import AudioSegment
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

import voiceCutting  # 既存のものを同フォルダに置く前提

# ====== 録音条件（Python内で統一）======
SAMPLE_RATE = 48000
CHANNELS = 1
DTYPE = "int16"
SAVE_SUBTYPE = "PCM_16"

# ====== FFT/特徴量（学習スクリプトの既定に合わせる）======
DEFAULT_FMIN = 0.0
DEFAULT_FMAX = 8000.0
DEFAULT_WINDOW = "hann"
DEFAULT_ZERO_MEAN = True
DEFAULT_USE_LOG1P = True
DEFAULT_SR = 48000
DEFAULT_NFFT = 65536
BAND_HZ = 60.0  # 学習は60Hzのみ
C_GRID = [0.001, 0.005, 0.01, 0.05, 0.1, 1.0, 3.0, 5.0, 10.0]
GAMMA_GRID = ["scale"]

# ====== 保存先（学習View）======
MODEL_FILENAME = "model.joblib"
META_FILENAME = "meta.json"

# ====== 分割最大数 ======
TARGET_COUNT = 200


def now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def safe_name(name: str) -> str:
    bad = ["\\", "/", ":", "*", "?", '"', "<", ">", "|"]
    for b in bad:
        name = name.replace(b, "_")
    return name.strip()


def get_next_index(out_dir: Path, prefix: str) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    max_n = 0
    pat = re.compile(rf"^{re.escape(prefix)}_(\d+)$")

    for p in out_dir.glob("*.wav"):
        stem = p.stem

        # 新形式: prefix_123.wav
        m = pat.match(stem)
        if m:
            n = int(m.group(1))
            max_n = max(max_n, n)
            continue

        # 旧形式: 123.wav（後方互換）
        try:
            n = int(stem)
            max_n = max(max_n, n)
        except ValueError:
            pass

    return max_n + 1


def denoise_wav_to_path(in_wav: Path, out_wav: Path) -> tuple[np.ndarray, int]:
    # librosa load -> zero-mean -> noisereduce -> PCM_16で保存
    y, fs = librosa.load(str(in_wav), sr=None, mono=True)
    y = y.astype(np.float32)
    y = y - np.mean(y)

    try:
        y_deno = nr.reduce_noise(y=y, sr=int(fs), stationary=False)
    except Exception:
        y_deno = y  # 必ず代入（未代入エラー防止）

    y_deno = np.clip(y_deno, -1.0, 1.0).astype(np.float32)
    y_i16 = (y_deno * 32767.0).astype(np.int16)
    sf.write(str(out_wav), y_i16, int(fs), subtype=SAVE_SUBTYPE)
    return y_deno, int(fs)


def split_cleaned_wav_to_folder(cleaned_wav: Path, out_dir: Path, start_index: int, prefix: str, mirror_dir: Path | None = None,) -> list[Path]:
    voiceCutting.TARGET_COUNT = int(TARGET_COUNT)

    audio = AudioSegment.from_file(str(cleaned_wav))
    ranges = voiceCutting.detect_nonsilent(
        audio,
        min_silence_len=voiceCutting.MIN_SILENCE_LEN_MS,
        silence_thresh=voiceCutting.SILENCE_THRESH_DBFS,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    if mirror_dir is not None:
        mirror_dir.mkdir(parents=True, exist_ok=True)

    if not ranges:
        return []

    refined = voiceCutting.postprocess_ranges(audio, ranges)

    saved_paths: list[Path] = []
    saved = 0
    for start_ms, end_ms in refined:
        s = max(0, start_ms - voiceCutting.KEEP_SILENCE_MS)
        e = min(len(audio), end_ms + voiceCutting.KEEP_SILENCE_MS)
        chunk = audio[s:e]

        if voiceCutting.is_valid_chunk(chunk):
            saved += 1
            idx = start_index + saved - 1
            filename = f"{prefix}_{idx}.wav"

            out_path = out_dir / filename
            chunk.export(
                str(out_path),
                format="wav",
                parameters=["-ac", "1", "-ar", "48000", "-acodec", "pcm_s16le"],
            )
            saved_paths.append(out_path)

            # ALL側へも同名で保存（ミラー）
            if mirror_dir is not None:
                mirror_path = mirror_dir / filename
                chunk.export(
                    str(mirror_path),
                    format="wav",
                    parameters=["-ac", "1", "-ar", "48000", "-acodec", "pcm_s16le"],
                )

            if saved >= TARGET_COUNT:
                break

    return saved_paths

def make_window(n: int, name: str) -> np.ndarray:
    name = name.lower()
    if name == "hann":
        return np.hanning(n).astype(np.float32)
    if name == "hamming":
        return np.hamming(n).astype(np.float32)
    if name == "rect":
        return np.ones(n, dtype=np.float32)
    raise ValueError(f"Unknown window: {name}")


def read_wav_mono_float32(wav_path: Path) -> tuple[np.ndarray, int]:
    # 保存はPCM16でも、FFT入力はfloat32に統一（学習・推論の整合）
    x, sr = sf.read(str(wav_path), dtype="float32", always_2d=True)
    x = x.mean(axis=1)
    return x.astype(np.float32), int(sr)


def wav_to_fft_mag(wav_path: Path, nfft: int, sr: int, window: str, zero_mean: bool) -> np.ndarray:
    x, sr_read = read_wav_mono_float32(wav_path)
    if sr_read != sr:
        raise ValueError(f"sr mismatch: {sr_read} vs {sr} (file={wav_path})")

    if zero_mean:
        x = x - float(np.mean(x))

    if len(x) > nfft:
        raise ValueError(f"Input longer than NFFT. len={len(x)} > nfft={nfft} (file={wav_path})")

    x_pad = np.zeros(nfft, dtype=np.float32)
    x_pad[: len(x)] = x

    w = make_window(nfft, window)
    X = np.fft.rfft(x_pad * w, n=nfft)
    return np.abs(X).astype(np.float32)


def mag_to_equal_band_features_sum(mag: np.ndarray, freqs: np.ndarray, fmin: float, fmax: float, band_hz: float) -> np.ndarray:
    edges = np.arange(float(fmin), float(fmax) + float(band_hz), float(band_hz), dtype=np.float32)
    n_bands = int(len(edges) - 1)
    feat = np.zeros(n_bands, dtype=np.float32)

    for i in range(n_bands):
        lo = float(edges[i])
        hi = float(edges[i + 1])
        if i == n_bands - 1:
            sel = (freqs >= lo) & (freqs <= hi)
        else:
            sel = (freqs >= lo) & (freqs < hi)
        feat[i] = float(np.sum(mag[sel])) if np.any(sel) else 0.0

    return feat


def wav_to_feature_vector(wav_path: Path) -> np.ndarray:
    sr = DEFAULT_SR
    nfft = DEFAULT_NFFT
    fmin = DEFAULT_FMIN
    fmax = DEFAULT_FMAX
    window = DEFAULT_WINDOW
    zero_mean = DEFAULT_ZERO_MEAN
    use_log1p = DEFAULT_USE_LOG1P
    band_hz = BAND_HZ

    freqs = np.fft.rfftfreq(nfft, d=1.0 / sr).astype(np.float32)
    mag = wav_to_fft_mag(wav_path, nfft=nfft, sr=sr, window=window, zero_mean=zero_mean)
    feat = mag_to_equal_band_features_sum(mag, freqs=freqs, fmin=fmin, fmax=fmax, band_hz=band_hz)
    if use_log1p:
        feat = np.log1p(feat)
    return feat.astype(np.float32)


def collect_dataset(dataset_root: Path, label_order: list[str]) -> tuple[np.ndarray, np.ndarray]:
    X_list = []
    y_list = []
    for lab in label_order:
        d = dataset_root / lab
        if not d.exists():
            continue
        for wf in sorted(d.glob("*.wav")):
            X_list.append(wav_to_feature_vector(wf))
            y_list.append(lab)

    if len(X_list) == 0:
        raise ValueError(f"No wav files found under: {dataset_root}")

    X = np.vstack([x.reshape(1, -1) for x in X_list]).astype(np.float32)
    y = np.array(y_list, dtype=object)
    return X, y


def train_60hz_and_export(dataset_root: Path, label_order: list[str], export_dir: Path) -> dict:
    # 60Hzのみ学習して EXPORTED_models/band_060Hz/ に保存（joblib形式は trainスクリプト互換）
    X, y = collect_dataset(dataset_root, label_order)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("svc", SVC(kernel="rbf", probability=True, random_state=42)),
    ])
    pipe.fit(X_tr, y_tr)

    param_grid = {
        "svc__C": C_GRID,
        "svc__gamma": GAMMA_GRID,
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    gs = GridSearchCV(pipe, param_grid=param_grid, cv=cv, scoring="accuracy", n_jobs=-1, refit=True)
    gs.fit(X_tr, y_tr)

    best_model = gs.best_estimator_
    y_pred = best_model.predict(X_te)
    acc = float(accuracy_score(y_te, y_pred))

    export_dir.mkdir(parents=True, exist_ok=True)

    payload = {"model": best_model, "label_names": list(label_order)}
    joblib.dump(payload, export_dir / MODEL_FILENAME)

    meta = {
        "fft": {
            "sr": DEFAULT_SR,
            "nfft": DEFAULT_NFFT,
            "fmin": DEFAULT_FMIN,
            "fmax": DEFAULT_FMAX,
            "window": DEFAULT_WINDOW,
            "zero_mean": DEFAULT_ZERO_MEAN,
            "use_log1p": DEFAULT_USE_LOG1P,
        },
        "feature": {"band_hz": BAND_HZ},
        "best_params": gs.best_params_,
        "cv_best_score": float(gs.best_score_),
        "label_names": list(label_order),
        "test_accuracy": acc,
    }
    (export_dir / META_FILENAME).write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    cm = confusion_matrix(y_te, y_pred, labels=list(label_order))

    n = max(2, len(label_order))
    fig_w = max(8.0, 1.2 * n)   # ラベル数に応じて横幅を広げる
    fig_h = max(6.0, 1.0 * n)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    plt.rcParams["font.size"] = 18

    disp = ConfusionMatrixDisplay(cm, display_labels=list(label_order))
    disp.plot(ax=ax, values_format="d", colorbar=False)

    # x軸ラベルが重ならないように回転
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

    fig.tight_layout()
    fig.savefig(export_dir / "confusion_matrix.png", dpi=200)
    plt.close(fig)

    return {"model_dir": str(export_dir), "test_accuracy": acc, "labels": list(label_order)}

@dataclass
class AppState:
    # 共通
    subject_dir: Path | None = None
    raw_dir: Path | None = None
    clean_dir: Path | None = None

    # 学習用
    dataset_root: Path | None = None  # dataset_root/label/*.wav
    current_label: str | None = None
    is_recording: bool = False
    stream: sd.InputStream | None = None
    frames: list[np.ndarray] | None = None
    cue_stop_event: threading.Event | None = None
    last_train_label: str | None = None
    prepared_labels: list[str] | None = None

    # 推論用
    infer_dir: Path | None = None
    is_recording_infer: bool = False
    stream_infer: sd.InputStream | None = None
    frames_infer: list[np.ndarray] | None = None

    # 一時保存用
    last_train_raw: Path | None = None
    last_train_clean: Path | None = None
    last_train_chunks: list[Path] | None = None

def main(page: ft.Page):
    page.title = "発音識別アプリ"
    page.window_width = 1080
    page.window_height = 720
    page.horizontal_alignment = ft.CrossAxisAlignment.CENTER

    state = AppState()

    status = ft.Text(value="準備できました", size=24, weight=ft.FontWeight.BOLD)

    all_buttons = []

    def reg_button(btn: ft.ElevatedButton):
        all_buttons.append(btn)
        return btn

    def apply_responsive_layout(w: float, h: float):
        # コンテンツ幅（広すぎる場合は上限）
        content_w = min(int(w * 0.95), 1400)

        # TextField類（画面幅に追従）
        tfw = max(320, int(content_w * 0.40))
        subject_name.width = tfw
        for tf in word_fields:
            tf.width = tfw
        model_parent.width = content_w  # 推論のモデル親フォルダは長くなりがちなので広め

        # ボタン幅
        bw = max(100, int(w * 0.10))
        for b in all_buttons:
            b.width = bw

        # 推論結果の表示領域：画面高さに応じて確保（小さい画面で見切れ対策）
        results_panel.height = max(200, int(h * 0.35))

        page.update()

    def on_resize(e: ft.PageResizeEvent):
        apply_responsive_layout(e.width, e.height)

    page.on_resize = on_resize

    def set_status(msg: str):
        status.value = msg
        page.update()

    def centered_cell(text: str):
        return ft.DataCell(
            ft.Container(
                content=ft.Text(text),
                alignment=ft.alignment.center,
                expand=True,
            )
        )
    
    def start_cue_cycle():
    # 既に動いていたら止めてから開始
        stop_cue_cycle()
        ev = threading.Event()
        state.cue_stop_event = ev

        def worker():
            try:
                colors = ["red", "green"]
                idx = 0
                while not ev.is_set():
                    cue_box.bgcolor = colors[idx % 2]
                    page.update()
                    idx += 1
                    if ev.wait(1.0):
                        break
            finally:
                cue_box.bgcolor = "red"
                page.update()

        threading.Thread(target=worker, daemon=True).start()

    def stop_cue_cycle():
        if state.cue_stop_event is not None:
            state.cue_stop_event.set()
            state.cue_stop_event = None

    # =========================
    # 学習View UI & handlers
    # =========================
    subject_name = ft.TextField(label="学習：保存フォルダ名", width=520)
    word_fields = [
        ft.TextField(label="覚えさせたい単語①", value="sakana", width=520),
        ft.TextField(label="覚えさせたい単語②", value="shakana", width=520),
        ft.TextField(label="覚えさせたい単語③", value="takana", width=520),
        ft.TextField(label="覚えさせたい単語④", value="thakana", width=520),
        ft.TextField(label="覚えさせたい単語⑤", value="tyakana", width=520),
    ]
    train_paths = ft.Text(value="", selectable=True)
    cue_box = ft.Container(
        width=140,
        height=140,
        bgcolor="red",   # 初期は緑
        border_radius=8,
    )
    cue_text = ft.Text("色が切り替わるタイミングで発音してください", size=18)
    label_counts: dict[str, int] = {}
    label_count_view: dict[str, ft.Text] = {}

    def ensure_label_state():
        """現在の単語に対して、取得回数の表示オブジェクトを用意する"""
        labs = state.prepared_labels if state.prepared_labels is not None else get_words()
        for lab in get_words():
            if lab not in label_counts:
                label_counts[lab] = 0
            if lab not in label_count_view:
                label_count_view[lab] = ft.Text(value=f"取得回数: {label_counts[lab]}")

    def get_words():
        words = []
        for tf in word_fields:
            w = (tf.value or "").strip()
            if w:
                words.append(w)
        return words[:5]
    
    def on_prepare_words(_):
        labels = get_words()
        if not labels:
            set_status("覚えさせたい単語を1つ以上入力してください。")
            return

        # 準備完了で確定
        state.prepared_labels = labels

        # 取得回数表示の生成（確定単語に対して）
        ensure_label_state()

        # ページ更新（ボタン数を確定単語数に合わせて作り直す）
        show_train()


    def update_train_paths():
        if state.subject_dir is None:
            train_paths.value = ""
        else:
            train_paths.value = (
                f"subject_dir: {state.subject_dir}\n"
                f"raw: {state.raw_dir}\n"
                f"clean: {state.clean_dir}\n"
                f"dataset_root: {state.dataset_root}\n"
                f"recording: {state.is_recording}\n"
                f"録音条件: sr={SAMPLE_RATE}, ch={CHANNELS}, dtype={DTYPE}, wav={SAVE_SUBTYPE}\n"
            )
        page.update()

    def on_create_train_folder(_):
        name = safe_name(subject_name.value or "")
        if not name:
            set_status("学習：保存フォルダ名が空です。")
            return

        base = Path.cwd()
        state.subject_dir = base / name
        state.raw_dir = state.subject_dir / "raw"
        state.clean_dir = state.subject_dir / "clean"
        state.dataset_root = state.subject_dir / "dataset_wav"

        state.subject_dir.mkdir(parents=True, exist_ok=True)
        state.raw_dir.mkdir(parents=True, exist_ok=True)
        state.clean_dir.mkdir(parents=True, exist_ok=True)
        state.dataset_root.mkdir(parents=True, exist_ok=True)

        set_status("AI学習：フォルダを作成しました")
        update_train_paths()

    def start_record_for_label(label: str):
        if state.subject_dir is None or state.raw_dir is None or state.clean_dir is None or state.dataset_root is None:
            set_status("学習：先にフォルダ作成してください。")
            return
        if state.is_recording:
            set_status("学習：すでに録音中です。")
            return

        state.current_label = label
        state.frames = []
        state.is_recording = True

        def callback(indata, frames, time, status_):
            state.frames.append(indata.copy())

        try:
            state.stream = sd.InputStream(
                samplerate=SAMPLE_RATE,
                channels=CHANNELS,
                dtype=DTYPE,
                callback=callback,
            )
            state.stream.start()
            start_cue_cycle()
            set_status(f"学習：録音中... label={label}（Stopで保存→分割→datasetへ）")
            update_train_paths()
        except Exception as e:
            stop_cue_cycle()
            state.is_recording = False
            state.stream = None
            state.frames = None
            set_status(f"学習：録音開始失敗: {e}")

    def on_stop_train_record_for(word: str):
        if not state.is_recording or state.stream is None or state.frames is None:
            set_status("録音中ではありません。")
            return
        if state.current_label != word:
            set_status("この単語は録音中ではありません。")
            return

        # stop stream
        try:
            state.stream.stop()
            state.stream.close()
        except Exception:
            pass
        stop_cue_cycle()

        state.stream = None
        state.is_recording = False

        stamp = now_stamp()
        raw_wav = state.raw_dir / f"{state.current_label}_{stamp}.wav"
        cleaned_wav = state.clean_dir / f"cleaned_{state.current_label}_{stamp}.wav"

        # 保存
        try:
            audio = np.concatenate(state.frames, axis=0)  # int16
            sf.write(str(raw_wav), audio, SAMPLE_RATE, subtype=SAVE_SUBTYPE)
        except Exception as e:
            set_status("音声の保存に失敗しました。もう一度録音してください。")
            state.frames = None
            return
        finally:
            state.frames = None
        # 直近録音を一時保存
        state.last_train_raw = raw_wav
        state.last_train_clean = cleaned_wav
        state.last_train_chunks = []

        def worker():
            try:
                denoise_wav_to_path(raw_wav, cleaned_wav)

                out_label_dir = state.dataset_root / state.current_label
                out_label_dir.mkdir(parents=True, exist_ok=True)
                prefix = state.subject_dir.name

                all_dataset_root = Path.cwd() / "ALL" / "dataset_wav"
                out_all_label_dir = all_dataset_root / state.current_label
                out_all_label_dir.mkdir(parents=True, exist_ok=True)

                start_index = get_next_index(out_label_dir, prefix=prefix)
                chunks = split_cleaned_wav_to_folder(cleaned_wav, out_label_dir, start_index=start_index, prefix=prefix, mirror_dir=out_all_label_dir)
                state.last_train_chunks = list(chunks)
                # ★ここを追加：直近がどの単語かを「保存完了時」に確定させる
                state.last_train_label = state.current_label

                # ★ここを追加：取得回数の更新
                ensure_label_state()
                label_counts[state.current_label] = label_counts.get(state.current_label, 0) + len(chunks)
                label_count_view[state.current_label].value = f"取得回数: {label_counts[state.current_label]}"

                set_status(f"学習：{len(chunks)}個の保存を行いました。")
                update_train_paths()
                page.update()
            except Exception:
                set_status("録音時にエラーが発生しました。もう一度録音してください。")

        threading.Thread(target=worker, daemon=True).start()

    def on_delete_last_for(word: str):
        deleted = 0

        if state.last_train_label != word:
            set_status("この単語の直近データはありません。")
            return

        # chunks（dataset内）を先に消す
        if state.last_train_chunks:
            for p in state.last_train_chunks:
                try:
                    if p and p.exists():
                        p.unlink()
                        deleted += 1
                except Exception:
                    pass

        # cleaned / raw を消す
        for p in [state.last_train_clean, state.last_train_raw]:
            try:
                if p and p.exists():
                    p.unlink()
                    deleted += 1
            except Exception:
                pass

        # 状態クリア
        state.last_train_raw = None
        state.last_train_clean = None
        state.last_train_chunks = []
        state.last_train_label = None

        set_status(f"学習：直近データを削除しました（{deleted}件）")
        update_train_paths()

    def on_train_start(_):
        if state.dataset_root is None:
            set_status("先にフォルダ作成を行ってください。")
            return

        labels = get_words()
        if not labels:
            set_status("覚えさせたい単語が登録されていません。")
            return

        def worker():
            try:
                res = train_60hz_and_export(state.dataset_root, labels, export_dir=state.subject_dir)
                set_status("学習が完了しました。AI評価に移るため、「戻る」ボタンを押し、AI評価を行ってください。")
            except Exception:
                set_status("収録でエラーが発生しました。収録をやり直してください。")

        threading.Thread(target=worker, daemon=True).start()

    def build_train_page():
        ensure_label_state()
        words = state.prepared_labels if state.prepared_labels is not None else []

        # 単語ごとの Row（開始・停止・削除・取得回数）を縦に並べる
        label_rows = ft.Column(
            controls=[
                ft.Row(
                    [
                        reg_button(ft.ElevatedButton(
                            text=f"単語{i+1} 録音開始",
                            on_click=lambda e, w=word: start_record_for_label(w)
                        )),
                        reg_button(ft.ElevatedButton(
                            text="録音停止",
                            on_click=lambda e, w=word: on_stop_train_record_for(w)
                        )),
                        reg_button(ft.ElevatedButton(
                            text="削除",
                            on_click=lambda e, w=word: on_delete_last_for(w)
                        )),
                        label_count_view[word],
                    ],
                    alignment=ft.MainAxisAlignment.CENTER
                )
                for i, word in enumerate(words)
            ],
            spacing=8,
        )
        folder_btn = reg_button(ft.ElevatedButton(text="フォルダ作成", on_click=on_create_train_folder))

        return ft.Column(
            [
                ft.Row([reg_button(ft.ElevatedButton("戻る", on_click=lambda _: show_home()))]),
                ft.Text("AI学習", size=18),
                # 単語入力欄（最大5）
                ft.Row([subject_name, folder_btn], alignment=ft.MainAxisAlignment.CENTER),
                ft.Row(
                    word_fields,
                    alignment=ft.MainAxisAlignment.CENTER,
                    scroll=ft.ScrollMode.AUTO,
                ),
                ft.Row(
                    [reg_button(ft.ElevatedButton(text="準備完了", on_click=on_prepare_words))],
                    alignment=ft.MainAxisAlignment.CENTER,
                ),
                status,
                ft.Row([cue_box], alignment=ft.MainAxisAlignment.CENTER),
                cue_text,
                ft.Divider(),
                label_rows,
                reg_button(ft.ElevatedButton(text="AI学習開始", on_click=on_train_start)),
                ft.Divider(),
                train_paths,
            ],
            spacing=10,
            horizontal_alignment=ft.CrossAxisAlignment.CENTER,
        )

    # =========================
    # 推論View UI & handlers
    # =========================
    model_parent = ft.TextField(label="AI学習で登録したフォルダ名を入力", width=900)

    infer_paths = ft.Text(value="", selectable=True)
    results_table = ft.DataTable(
        columns=[
            ft.DataColumn(ft.Text("No.")),
            ft.DataColumn(ft.Text("推定ラベル")),
            ft.DataColumn(ft.Text("確率(%)")),
            ft.DataColumn(ft.Text("wav")),
        ],
        rows=[],
    )
    results_panel = ft.Container(
        content=ft.ListView(
            controls=[
                ft.Row([results_table], alignment=ft.MainAxisAlignment.CENTER, scroll=ft.ScrollMode.AUTO)
            ],
            expand=True,                      # 縦スクロール領域を確保
            spacing=0,
            padding=0,
        ),
        expand=True, 
        alignment=ft.alignment.center,                     # 親Column内で残り領域を使う
    )

    def update_infer_paths():
        if state.infer_dir is None:
            infer_paths.value = ""
        else:
            infer_paths.value = (
                f"infer_dir: {state.infer_dir}\n"
                f"recording: {state.is_recording_infer}\n"
                f"録音条件: sr={SAMPLE_RATE}, ch={CHANNELS}, dtype={DTYPE}, wav={SAVE_SUBTYPE}\n"
                f"モデル親フォルダ: {model_parent.value or ''}\n"
            )
        page.update()

    def on_create_infer_folder(_):
        name = safe_name(model_parent.value or "")
        if not name:
            set_status("推論：保存フォルダ名が空です。")
            return
        base = Path.cwd()
        state.infer_dir = base / name
        (state.infer_dir / "raw").mkdir(parents=True, exist_ok=True)
        (state.infer_dir / "clean").mkdir(parents=True, exist_ok=True)
        (state.infer_dir / "chunks").mkdir(parents=True, exist_ok=True)
        set_status(f"推論：フォルダ作成 {state.infer_dir}")
        update_infer_paths()

    def start_infer_record(_):
        if state.infer_dir is None:
            set_status("推論：先に保存フォルダを作成してください。")
            return
        if state.is_recording_infer:
            set_status("推論：すでに録音中です。")
            return
        if not (model_parent.value or "").strip():
            set_status("推論：モデル親フォルダを指定してください。")
            return

        state.frames_infer = []
        state.is_recording_infer = True

        def callback(indata, frames, time, status_):
            state.frames_infer.append(indata.copy())

        try:
            state.stream_infer = sd.InputStream(
                samplerate=SAMPLE_RATE,
                channels=CHANNELS,
                dtype=DTYPE,
                callback=callback,
            )
            state.stream_infer.start()
            set_status("推論：録音中...（Stopで保存→分割→推論）")
            update_infer_paths()
        except Exception as e:
            state.is_recording_infer = False
            state.stream_infer = None
            state.frames_infer = None
            set_status(f"推論：録音開始失敗: {e}")

    def stop_and_infer(_):
        if not state.is_recording_infer or state.stream_infer is None or state.frames_infer is None:
            set_status("推論：録音中ではありません。")
            return
        if state.infer_dir is None:
            set_status("推論：保存フォルダが未作成です。")
            return
        if not (model_parent.value or "").strip():
            set_status("推論：モデル親フォルダを指定してください。")
            return

        # stop stream
        try:
            state.stream_infer.stop()
            state.stream_infer.close()
        except Exception:
            pass

        state.stream_infer = None
        state.is_recording_infer = False

        stamp = now_stamp()
        raw_wav = state.infer_dir / "raw" / f"{stamp}.wav"
        cleaned_wav = state.infer_dir / "clean" / f"cleaned_{stamp}.wav"
        chunk_dir = state.infer_dir / "chunks"

        # raw保存
        try:
            audio = np.concatenate(state.frames_infer, axis=0)  # int16
            sf.write(str(raw_wav), audio, SAMPLE_RATE, subtype=SAVE_SUBTYPE)
        except Exception as e:
            set_status(f"推論：raw保存失敗: {e}")
            state.frames_infer = None
            return
        finally:
            state.frames_infer = None

        def worker():
            try:
                parent = Path(model_parent.value.strip())
                model_path = parent / "model.joblib"
                payload = joblib.load(str(model_path))
                model = payload["model"]
                label_names = payload["label_names"]

                denoise_wav_to_path(raw_wav, cleaned_wav)
                start_index = get_next_index(chunk_dir, prefix="infer")
                chunks = split_cleaned_wav_to_folder(cleaned_wav, chunk_dir, start_index=start_index, prefix="infer", mirror_dir=None,)

                rows = []
                for i, wp in enumerate(chunks, start=1):
                    try:
                        feat = wav_to_feature_vector(wp)
                        proba = model.predict_proba([feat])[0]
                        pred_id = int(np.argmax(proba))
                        pred_label = str(label_names[pred_id])
                        pred_pct = float(proba[pred_id]) * 100.0

                        rows.append(
                            ft.DataRow(
                                cells=[
                                    centered_cell(str(i)),
                                    centered_cell(pred_label),
                                    centered_cell(f"{pred_pct:.1f}"),
                                    centered_cell(f"{wp.name}"),
                                ]
                            )
                        )
                    except Exception as e:
                        rows.append(
                            ft.DataRow(
                                cells=[
                                    centered_cell(str(i)),
                                    centered_cell("ERROR"),
                                    centered_cell("-"),
                                    centered_cell(f"{wp.name}"),
                                ]
                            )
                        )

                results_table.rows = rows
                set_status(f"推論：完了（分割={len(chunks)} → 推論={len(chunks)}）")
                update_infer_paths()
                page.update()

            except Exception:
                set_status("推論処理でエラーが発生しました。再度録音してください。")

        threading.Thread(target=worker, daemon=True).start()

    def build_infer_page():
        return ft.Column(
            [
                ft.Row([reg_button(ft.ElevatedButton("戻る", on_click=lambda _: show_home()))]),
                ft.Text("AI推論", size=18),
                model_parent,
                ft.Row([
                    reg_button(ft.ElevatedButton(text="AI推論モデル読みこみ", on_click=on_create_infer_folder)),
                ], alignment=ft.MainAxisAlignment.CENTER),
                ft.Divider(),
                ft.Row([
                    reg_button(ft.ElevatedButton(text="録音開始", on_click=start_infer_record)),
                    reg_button(ft.ElevatedButton(text="録音終了", on_click=stop_and_infer)),
                ], alignment=ft.MainAxisAlignment.CENTER),
                status,
                ft.Divider(),
                #status,
                infer_paths,
                ft.Divider(),
                ft.Text("推論結果（推定ラベル / 確率%）"),
                results_panel,
            ],
            spacing=10,
            horizontal_alignment=ft.CrossAxisAlignment.CENTER,
        )

    # =========================
    # Home View（モード選択）: fletMouthPredect.py と同様に controls を入れ替える
    # =========================
    def build_home_page():
        return ft.Column(
            [
                ft.Text("モード選択", size=20),
                ft.Row(
                    [
                        reg_button(ft.ElevatedButton("学習へ", on_click=lambda _: show_train())),
                        reg_button(ft.ElevatedButton("推論へ", on_click=lambda _: show_infer())),
                    ],
                    alignment=ft.MainAxisAlignment.CENTER,
                ),
                ft.Divider(),
                status,
            ],
            spacing=10
        )

    def show_home():
        page.controls.clear()
        page.add(build_home_page())
        set_status("")
        page.update() 
        apply_responsive_layout(page.width, page.height)

    def show_train():
        page.controls.clear()
        page.add(build_train_page())
        page.update()
        apply_responsive_layout(page.width, page.height)

    def show_infer():
        page.controls.clear()
        page.add(build_infer_page())
        page.update()
        apply_responsive_layout(page.width, page.height)

    # 起動時
    show_home()


if __name__ == "__main__":
    ft.app(target=main)
