# voiceCutting.py方式で区間切り出し → 各区間をSVM(model.joblib)で推論して出力

import os
import json
import unicodedata
import numpy as np
import joblib

from pathlib import Path
from pydub import AudioSegment
from pydub.silence import detect_nonsilent

# ====== 入力（要変更） ======
AUDIO_PATH  = "word_Ex1/10times_Ex1_B/takana6/cleaned_audio.wav"
OUTPUT_PATH = "word_Ex1/10times_Ex1_B/takana6/svm_segmented_result2.txt"

# 学習スクリプトが保存した場所（例）
MODEL_JOBLIB = "word_Ex1/trained_all_svm_model_band/sum73/BEST_band_020Hz/model.joblib"
META_JSON    = "word_Ex1/trained_all_svm_model_band/sum73/BEST_band_020Hz/meta.json"
# ============================

# ====== 切り出し（wordWhisper3.py/voiceCutting.py と同等） ======
MIN_SILENCE_LEN_MS = 250
SILENCE_THRESH_DBFS = -62.0
KEEP_SILENCE_MS = 100

TARGET_COUNT = 10
FRAME_MS = 5
TRIM_REL_DB = 30.0
MAX_CHUNK_MS = 1000
MIN_CHUNK_MS = 120
VALLEY_DROP_DB = 20.0
MIN_GAP_MS = 80
MIN_AVG_DBFS = -35.0
MIN_PEAK_DBFS = -25.0

def is_valid_chunk(chunk: AudioSegment) -> bool:
    if len(chunk) < MIN_CHUNK_MS:
        return False
    if chunk.dBFS < MIN_AVG_DBFS:
        return False
    if chunk.max_dBFS < MIN_PEAK_DBFS:
        return False
    return True

def _frame_rms_dbfs(seg: AudioSegment, frame_ms: int) -> np.ndarray:
    sr = seg.frame_rate
    samples = np.array(seg.get_array_of_samples()).astype(np.float32)
    if seg.channels > 1:
        samples = samples.reshape((-1, seg.channels)).mean(axis=1)

    max_amp = float(seg.max_possible_amplitude)
    if max_amp > 0:
        samples /= max_amp

    hop = int(sr * frame_ms / 1000.0)
    hop = max(hop, 1)

    rms = []
    for i in range(0, len(samples), hop):
        x = samples[i:i+hop]
        if len(x) == 0:
            continue
        v = np.sqrt(np.mean(x * x) + 1e-12)
        rms.append(20.0 * np.log10(v + 1e-12))
    return np.array(rms, dtype=np.float32)

def tighten_bounds(audio: AudioSegment, s_ms: int, e_ms: int) -> tuple[int, int]:
    seg = audio[s_ms:e_ms]
    rms_db = _frame_rms_dbfs(seg, FRAME_MS)
    if len(rms_db) == 0:
        return s_ms, e_ms

    peak = float(np.max(rms_db))
    thr = peak - TRIM_REL_DB
    idx = np.where(rms_db >= thr)[0]
    if len(idx) == 0:
        return s_ms, e_ms

    hop = FRAME_MS
    s2 = s_ms + int(idx[0] * hop)
    e2 = s_ms + int((idx[-1] + 1) * hop)
    s2 = max(0, s2)
    e2 = min(len(audio), e2)
    if e2 <= s2:
        return s_ms, e_ms
    return s2, e2

def split_if_too_long(audio: AudioSegment, s_ms: int, e_ms: int) -> list[tuple[int, int]]:
    dur = e_ms - s_ms
    if dur <= MAX_CHUNK_MS:
        return [(s_ms, e_ms)]

    seg = audio[s_ms:e_ms]
    rms_db = _frame_rms_dbfs(seg, FRAME_MS)
    if len(rms_db) == 0:
        return [(s_ms, e_ms)]

    peak = float(np.max(rms_db))
    valley_thr = peak - VALLEY_DROP_DB
    cand = np.where(rms_db <= valley_thr)[0]
    if len(cand) == 0:
        return [(s_ms, e_ms)]

    hop = FRAME_MS
    mid = len(rms_db) // 2
    min_gap_frames = max(int(MIN_GAP_MS / hop), 1)
    valid = [i for i in cand if (i > min_gap_frames) and (i < len(rms_db) - min_gap_frames)]
    if not valid:
        return [(s_ms, e_ms)]

    best = min(valid, key=lambda i: (abs(i - mid), rms_db[i]))
    cut_ms = s_ms + int(best * hop)

    left = split_if_too_long(audio, s_ms, cut_ms)
    right = split_if_too_long(audio, cut_ms, e_ms)
    return left + right

def postprocess_ranges(audio: AudioSegment, ranges: list[list[int]]) -> list[tuple[int, int]]:
    out = []
    for s, e in ranges:
        s2, e2 = tighten_bounds(audio, s, e)
        parts = split_if_too_long(audio, s2, e2)
        for ps, pe in parts:
            ps2, pe2 = tighten_bounds(audio, ps, pe)
            out.append((ps2, pe2))

    guard = 0
    while len(out) < TARGET_COUNT and guard < 50:
        guard += 1
        i = int(np.argmax([e - s for s, e in out]))
        s, e = out.pop(i)
        parts = split_if_too_long(audio, s, e)
        if len(parts) == 1:
            out.append((s, e))
            break
        out.extend(parts)

    out.sort(key=lambda x: x[0])
    return out

# ====== 特徴量（soundML_train_SVM_band.py と同じ計算） ======
def make_window(n: int, name: str) -> np.ndarray:
    name = name.lower()
    if name == "hann":
        return np.hanning(n).astype(np.float32)
    if name == "hamming":
        return np.hamming(n).astype(np.float32)
    if name == "rect":
        return np.ones(n, dtype=np.float32)
    raise ValueError(f"Unknown window: {name}")

def audiosegment_to_float32_mono(seg: AudioSegment, target_sr: int) -> np.ndarray:
    seg = seg.set_channels(1)
    seg = seg.set_frame_rate(target_sr)
    samples = np.array(seg.get_array_of_samples()).astype(np.float32)
    if seg.sample_width == 2:
        samples /= 32768.0
    elif seg.sample_width == 4:
        samples /= 2147483648.0
    else:
        maxv = float(2 ** (8 * seg.sample_width - 1))
        samples /= maxv
    return samples.astype(np.float32)

def wav_to_fft_mag_from_array(x: np.ndarray, nfft: int, window: str, zero_mean: bool) -> np.ndarray:
    if zero_mean:
        x = x - float(np.mean(x))
    if len(x) > nfft:
        raise ValueError(f"Input longer than NFFT. len={len(x)} > nfft={nfft}")
    x_pad = np.zeros(nfft, dtype=np.float32)
    x_pad[:len(x)] = x
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

def main():
    # 1) モデルとメタ読み込み（学習設定と一致させるため）
    pack = joblib.load(MODEL_JOBLIB)
    clf = pack["model"]
    label_names = pack["label_names"]

    meta = json.loads(Path(META_JSON).read_text(encoding="utf-8"))

    sr = int(meta["fft"]["sr"])
    nfft = int(meta["fft"]["nfft"])
    fmin = float(meta["fft"]["fmin"])
    fmax = float(meta["fft"]["fmax"])
    window = str(meta["fft"]["window"])
    zero_mean = bool(meta["fft"]["zero_mean"])
    use_log1p = bool(meta["fft"]["use_log1p"])
    band_hz = float(meta["feature"]["band_hz"])

    freqs = np.fft.rfftfreq(nfft, d=1.0 / sr).astype(np.float32)

    # 2) 音声を区間切り出し
    audio = AudioSegment.from_file(AUDIO_PATH)
    ranges = detect_nonsilent(
        audio,
        min_silence_len=MIN_SILENCE_LEN_MS,
        silence_thresh=SILENCE_THRESH_DBFS,
    )
    if not ranges:
        raise RuntimeError("no voiced segments detected (ranges is empty).")

    refined = postprocess_ranges(audio, ranges)

    # 3) 各区間をSVMで推論して出力
    lines = []
    lines.append("[model]")
    lines.append(f"model_joblib={MODEL_JOBLIB}")
    lines.append(f"meta_json={META_JSON}")
    lines.append(f"labels={label_names}")
    lines.append("")
    lines.append("[segments]")

    idx = 0
    for (start_ms, end_ms) in refined:
        s = max(0, start_ms - KEEP_SILENCE_MS)
        e = min(len(audio), end_ms + KEEP_SILENCE_MS)
        chunk = audio[s:e]
        if not is_valid_chunk(chunk):
            continue

        idx += 1
        x = audiosegment_to_float32_mono(chunk, target_sr=sr)

        mag = wav_to_fft_mag_from_array(x, nfft=nfft, window=window, zero_mean=zero_mean)
        feat = mag_to_equal_band_features_sum(mag, freqs, fmin=fmin, fmax=fmax, band_hz=band_hz)
        if use_log1p:
            feat = np.log1p(feat)

        X = feat.reshape(1, -1).astype(np.float32)

        pred_id = int(clf.predict(X)[0])
        pred_label = label_names[pred_id]

        # 「どのように認識したか」＝確率（PROBABILITY=True の前提）
        if hasattr(clf, "predict_proba"):
            proba = clf.predict_proba(X)[0]
            # ラベルと確率を結合して降順ソート
            pairs = list(zip(label_names, proba))
            pairs.sort(key=lambda x: x[1], reverse=True)

            # 上位3件だけ（％表示）
            top3 = pairs[:3]
            top3_str = " ".join([f"{lab}:{p*100:.1f}%" for lab, p in top3])

            # 予測ラベル（最大確率）
            pred_label = pairs[0][0]

            lines.append(f"{idx:02d}\t{s/1000:.2f}-{e/1000:.2f}\t{pred_label}\t{top3_str}")
        else:
            lines.append(f"{idx:02d}\t{s/1000:.2f}-{e/1000:.2f}\t{pred_label}")

    out_dir = os.path.dirname(OUTPUT_PATH) or "."
    os.makedirs(out_dir, exist_ok=True)
    Path(OUTPUT_PATH).write_text("\n".join(lines) + "\n", encoding="utf-8")

if __name__ == "__main__":
    main()
