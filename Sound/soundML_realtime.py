# 録音→ノイズ処理→自動分割→(学習と同じ)FFT+バンド特徴→SVM推論（録音停止後に実行）
#
# 前提:
#  - voiceCutting.py が同フォルダにある（fletSound系と同じ）
#  - model.joblib は soundML_train_SVM_band.py の形式 {"model": Pipeline, "label_names": [...]}
#  - 可能なら同階層に meta.json（学習時パラメータ）があると、推論側の設定が自動一致します

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import json
import traceback

import flet as ft

import numpy as np
import sounddevice as sd
import soundfile as sf
import librosa
import noisereduce as nr
from pydub import AudioSegment

import joblib
import voiceCutting  # 既存のものを利用


# ========= ユーザー指定 =========
#MODEL_JOBLIB_PATH = Path("word_Ex1/trained_all_svm_model_band/EXPORTED_models/band_060Hz/model.joblib")  # 必要なら実パスに変更
#MODEL_JOBLIB_PATH = Path("word/trained_Y_svm_model_band/EXPORTED_models/band_060Hz/model.joblib")  # 必要なら実パスに変更
MODEL_JOBLIB_PATH = Path("YR/model.joblib")  # 必要なら実パスに変更

# ========= 録音設定（fletSound系を踏襲） =========
SAMPLE_RATE = 48000
CHANNELS = 1
DTYPE = "int16"

# 分割保存先の最大数（必要なら増やせますが、ここでは「分割された数だけ推論」を優先）
TARGET_COUNT = 100


# ========= 学習側と同じ特徴量化（soundML_train_SVM_band.py相当） =========
DEFAULT_FMIN = 0
DEFAULT_FMAX = 8000
DEFAULT_WINDOW = "hann"
DEFAULT_ZERO_MEAN = True
DEFAULT_USE_LOG1P = True
DEFAULT_SR = 48000
DEFAULT_NFFT = 65536
DEFAULT_BAND_HZ = 60.0


def now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def safe_subject_name(name: str) -> str:
    bad = ["\\", "/", ":", "*", "?", '"', "<", ">", "|"]
    for b in bad:
        name = name.replace(b, "_")
    return name.strip()


def get_next_index(out_dir: Path) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    max_n = 0
    for p in out_dir.glob("*.wav"):
        try:
            n = int(p.stem)
            if n > max_n:
                max_n = n
        except ValueError:
            pass
    return max_n + 1

def denoise_wav_to_path(in_wav: Path, out_wav: Path) -> tuple[np.ndarray, int]:
    # fletSound系の流れ（librosa load → zero-mean → noisereduce → write）
    y, fs = librosa.load(str(in_wav), sr=None, mono=True)
    y = y.astype(np.float32)
    y = y - np.mean(y)
    y_deno = nr.reduce_noise(y=y, sr=int(fs), stationary=False)
    y_deno = np.clip(y_deno, -1.0, 1.0).astype(np.float32)
    y_i16 = (y_deno * 32767.0).astype(np.int16)
    sf.write(str(out_wav), y_i16, int(fs), subtype="PCM_16")

    return y_deno, int(fs)

def split_cleaned_wav_to_folder(
    cleaned_wav: Path,
    out_dir: Path,
    target_count: int,
    start_index: int,
) -> list[Path]:
    # voiceCutting.py を利用（fletSound系を踏襲）
    voiceCutting.TARGET_COUNT = int(target_count)

    audio = AudioSegment.from_file(str(cleaned_wav))

    ranges = voiceCutting.detect_nonsilent(
        audio,
        min_silence_len=voiceCutting.MIN_SILENCE_LEN_MS,
        silence_thresh=voiceCutting.SILENCE_THRESH_DBFS,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
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
            file_index = start_index + saved - 1
            out_path = out_dir / f"{file_index}.wav"
            chunk.export(str(out_path), format="wav")
            saved_paths.append(out_path)
            if saved >= target_count:
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
    # soundfileで読み込み（PCM16/PCM32/float WAV等を吸収）
    x, sr = sf.read(str(wav_path), dtype="float32", always_2d=True)  # shape: (N, C)
    x = x.mean(axis=1)  # モノラル化
    return x.astype(np.float32), int(sr)

def wav_to_fft_mag(
    wav_path: Path,
    nfft: int,
    sr: int,
    window: str,
    zero_mean: bool,
) -> np.ndarray:
    x, sr_read = read_wav_mono_float32(wav_path)
    if sr_read != sr:
        raise ValueError(f"sr mismatch: {sr_read} vs {sr} (file={wav_path})")

    if zero_mean:
        x = x - float(np.mean(x))

    if len(x) > nfft:
        raise ValueError(f"Input longer than NFFT. len={len(x)} > nfft={nfft} (file={wav_path})")

    x_pad = np.zeros(nfft, dtype=np.float32)
    x_pad[:len(x)] = x

    w = make_window(nfft, window)
    X = np.fft.rfft(x_pad * w, n=nfft)
    mag = np.abs(X).astype(np.float32)
    return mag


def mag_to_equal_band_features_sum(
    mag: np.ndarray,
    freqs: np.ndarray,
    fmin: float,
    fmax: float,
    band_hz: float,
) -> np.ndarray:
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


def load_model_and_meta(model_joblib: Path) -> tuple[object, list[str], dict]:
    bundle = joblib.load(str(model_joblib))
    model = bundle["model"]
    label_names = bundle["label_names"]

    meta_path = model_joblib.parent / "meta.json"
    meta = {}
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    return model, label_names, meta


def get_fft_feature_params(meta: dict) -> dict:
    # meta.json があればそれを優先、なければ学習コード既定値で埋める
    fft = meta.get("fft", {}) if isinstance(meta, dict) else {}
    feat = meta.get("feature", {}) if isinstance(meta, dict) else {}

    return {
        "sr": int(fft.get("sr", DEFAULT_SR)),
        "nfft": int(fft.get("nfft", DEFAULT_NFFT)),
        "fmin": float(fft.get("fmin", DEFAULT_FMIN)),
        "fmax": float(fft.get("fmax", DEFAULT_FMAX)),
        "window": str(fft.get("window", DEFAULT_WINDOW)),
        "zero_mean": bool(fft.get("zero_mean", DEFAULT_ZERO_MEAN)),
        "use_log1p": bool(fft.get("use_log1p", DEFAULT_USE_LOG1P)),
        "band_hz": float(feat.get("band_hz", DEFAULT_BAND_HZ)),
    }


def wav_to_feature_vector(wav_path: Path, params: dict) -> np.ndarray:
    sr = int(params["sr"])
    nfft = int(params["nfft"])
    fmin = float(params["fmin"])
    fmax = float(params["fmax"])
    window = str(params["window"])
    zero_mean = bool(params["zero_mean"])
    use_log1p = bool(params["use_log1p"])
    band_hz = float(params["band_hz"])

    freqs = np.fft.rfftfreq(nfft, d=1.0 / sr).astype(np.float32)

    mag = wav_to_fft_mag(
        wav_path=wav_path,
        nfft=nfft,
        sr=sr,
        window=window,
        zero_mean=zero_mean,
    )

    feat = mag_to_equal_band_features_sum(
        mag=mag,
        freqs=freqs,
        fmin=fmin,
        fmax=fmax,
        band_hz=band_hz,
    )

    if use_log1p:
        feat = np.log1p(feat)

    return feat.astype(np.float32)


@dataclass
class AppState:
    subject_dir: Path | None = None
    raw_dir: Path | None = None
    clean_dir: Path | None = None
    chunk_dir: Path | None = None

    is_recording: bool = False
    stream: sd.InputStream | None = None
    frames: list[np.ndarray] | None = None
    last_raw_wav: Path | None = None


def main(page: ft.Page):
    page.title = "発音評価リアルタイム推論"
    page.window_width = 980
    page.window_height = 720

    state = AppState()

    subject_name = ft.TextField(label="被験者名（フォルダ名）", width=520)
    status = ft.Text(value="未作成", selectable=True)
    paths_view = ft.Text(value="", selectable=True)

    # 推論結果テーブル
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
                ft.Row([results_table], scroll=ft.ScrollMode.AUTO)  # 横スクロール
            ],
            expand=True,                      # 縦スクロール領域を確保
            spacing=0,
            padding=0,
        ),
        expand=True,                          # 親Column内で残り領域を使う
    )

    def set_status(msg: str):
        status.value = msg
        page.update()

    def set_paths():
        if state.subject_dir is None:
            paths_view.value = ""
        else:
            paths_view.value = (
                f"フォルダ: {state.subject_dir}\n"
                f"raw: {state.raw_dir}\n"
                f"clean: {state.clean_dir}\n"
                f"chunks: {state.chunk_dir}\n"
                f"最終録音: {state.last_raw_wav if state.last_raw_wav else '-'}\n"
                f"モデル: {MODEL_JOBLIB_PATH}"
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
        chunk_dir = subject_dir / "chunks"

        subject_dir.mkdir(parents=True, exist_ok=True)
        raw_dir.mkdir(parents=True, exist_ok=True)
        clean_dir.mkdir(parents=True, exist_ok=True)
        chunk_dir.mkdir(parents=True, exist_ok=True)

        state.subject_dir = subject_dir
        state.raw_dir = raw_dir
        state.clean_dir = clean_dir
        state.chunk_dir = chunk_dir

        set_status(f"作成しました: {subject_dir}")
        set_paths()

    def start_recording(_):
        if state.subject_dir is None or state.raw_dir is None:
            set_status("先に被験者フォルダを作成してください。")
            return
        if state.is_recording:
            set_status("すでに録音中です。")
            return

        state.frames = []
        state.is_recording = True

        def callback(indata, frames, time, status_):
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
            set_status("録音中…（Stopで終了→推論）")
            set_paths()
        except Exception as e:
            state.is_recording = False
            state.stream = None
            state.frames = None
            set_status(f"録音開始に失敗: {e}")

    def stop_and_infer(_):
        if not state.is_recording or state.stream is None or state.frames is None:
            set_status("録音中ではありません。")
            return
        if state.raw_dir is None or state.clean_dir is None or state.chunk_dir is None:
            set_status("フォルダが未作成です。")
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
            set_status(f"録音保存: {raw_wav.name} → 後処理＆推論中…")
            set_paths()
        except Exception as e:
            set_status(f"録音保存に失敗: {e}")
            return
        finally:
            state.frames = None

        # 後処理＋推論（別スレッド）
        def run_postprocess_and_infer():
            try:
                if not MODEL_JOBLIB_PATH.exists():
                    set_status(f"モデルが見つかりません: {MODEL_JOBLIB_PATH}")
                    return

                model, label_names, meta = load_model_and_meta(MODEL_JOBLIB_PATH)
                params = get_fft_feature_params(meta)

                cleaned_wav = state.clean_dir / f"cleaned_{stamp}.wav"
                denoise_wav_to_path(raw_wav, cleaned_wav)

                start_index = get_next_index(state.chunk_dir)
                voiceCutting.SILENCE_THRESH_DBFS = -40  # 録音用にしきい値を上げる
                chunk_paths = split_cleaned_wav_to_folder(
                    cleaned_wav=cleaned_wav,
                    out_dir=state.chunk_dir,
                    target_count=TARGET_COUNT,
                    start_index=start_index,
                )

                # テーブル更新
                rows = []
                for i, wav_path in enumerate(chunk_paths, start=1):
                    try:
                        feat = wav_to_feature_vector(wav_path, params)
                        proba = model.predict_proba([feat])[0]  # shape=(n_classes,)
                        pred_id = int(np.argmax(proba))
                        pred_label = str(label_names[pred_id])
                        pred_pct = float(proba[pred_id]) * 100.0

                        rows.append(
                            ft.DataRow(
                                cells=[
                                    ft.DataCell(ft.Text(str(i))),
                                    ft.DataCell(ft.Text(pred_label)),
                                    ft.DataCell(ft.Text(f"{pred_pct:.1f}")),
                                    ft.DataCell(ft.Text(wav_path.name)),
                                ]
                            )
                        )
                    except Exception as e:
                        rows.append(
                            ft.DataRow(
                                cells=[
                                    ft.DataCell(ft.Text(str(i))),
                                    ft.DataCell(ft.Text("ERROR")),
                                    ft.DataCell(ft.Text("-")),
                                    ft.DataCell(ft.Text(f"{wav_path.name} / {e}")),
                                ]
                            )
                        )

                results_table.rows = rows
                set_status(f"完了: 分割 {len(chunk_paths)} 個 → 推論 {len(chunk_paths)} 回")
                set_paths()
                page.update()

            except Exception as e:
                tb = traceback.format_exc()
                set_status(f"後処理/推論に失敗: {e}\n{tb}")

        page.run_thread(run_postprocess_and_infer)

    page.add(
        ft.Column(
            [
                ft.Text("リアルタイム推論", size=18, weight=ft.FontWeight.BOLD),
                subject_name,
                ft.Row([ft.Button("① フォルダ選択", on_click=on_create_folder)]),
                ft.Divider(),
                ft.Row(
                    [
                        ft.Button("録音開始", on_click=start_recording),
                        ft.Button("録音停止", on_click=stop_and_infer),
                    ]
                ),
                ft.Divider(),
                ft.Text("状態:"),
                status,
                ft.Divider(),
                ft.Text("パス:"),
                paths_view,
                ft.Divider(),
                ft.Text("推論結果（推定ラベル / 確率%）:"),
                results_panel,
            ],
            spacing=10,
        )
    )


if __name__ == "__main__":
    ft.app(target=main)
