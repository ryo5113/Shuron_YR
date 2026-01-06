# voiceCutting.py と同じ有音区間検出＋後処理を行い、
# whisperで文字起こし＋漢字抑止を行うスクリプト

import os
import sys
import unicodedata
import numpy as np
import whisper

from pydub import AudioSegment
from pydub.silence import detect_nonsilent

# ====== 入力（必要に応じて変更） ======
AUDIO_PATH  = "word/10times_02/sakana.wav"
OUTPUT_PATH = "word/10times_02/sa/sakana_segmented_pydub.txt"

LANGUAGE    = "ja"       
MODEL_NAME  = "large-v3"
TEMPERATURE = 0.0

# ====== voiceCutting.py と同じ分割パラメータ ======
MIN_SILENCE_LEN_MS = 250
SILENCE_THRESH_DBFS = -70.0
KEEP_SILENCE_MS = 50

# 後処理パラメータ（voiceCutting.pyと同様）
TARGET_COUNT = 10
FRAME_MS = 5
TRIM_REL_DB = 25.0
MAX_CHUNK_MS = 1200
MIN_CHUNK_MS = 120
VALLEY_DROP_DB = 20.0
MIN_GAP_MS = 80
MIN_AVG_DBFS = -35.0
MIN_PEAK_DBFS = -25.0
# =====================================

# ====== 漢字抑止（元 wordWhisper.py の方針と同じ） ======
def token_has_kanji(token_text: str) -> bool:
    for ch in token_text:
        name = unicodedata.name(ch, "")
        if "CJK UNIFIED IDEOGRAPH" in name:
            return True
        code = ord(ch)
        if (0x4E00 <= code <= 0x9FFF) or (0x3400 <= code <= 0x4DBF) or (0xF900 <= code <= 0xFAFF):
            return True
    return False

def build_suppress_tokens_for_kanji(language: str):
    tok = whisper.tokenizer.get_tokenizer(multilingual=True, language=language, task="transcribe")
    n_vocab = tok.encoding.n_vocab  # tok.n_vocab ではなく encoding 側
    suppress = []
    for tid in range(n_vocab):
        try:
            s = tok.decode([tid])
        except Exception:
            continue
        if s and token_has_kanji(s):
            suppress.append(tid)
    return sorted(set(suppress))

# ====== voiceCutting.py 相当の後処理 ======
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

    # 10個に近づけたい場合：最長を再分割して増やす（voiceCutting.pyと同様）
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

# ====== AudioSegment -> whisper用 float32 wave ======
def audiosegment_to_whisper_wave(seg: AudioSegment) -> np.ndarray:
    seg = seg.set_channels(1)
    seg = seg.set_frame_rate(16000)
    samples = np.array(seg.get_array_of_samples()).astype(np.float32)
    if seg.sample_width == 2:
        samples /= 32768.0
    elif seg.sample_width == 4:
        samples /= 2147483648.0
    else:
        maxv = float(2 ** (8 * seg.sample_width - 1))
        samples /= maxv
    return samples

def main():
    if LANGUAGE != "ja":
        raise ValueError('LANGUAGE は前提により "ja" にしてください。')

    model = whisper.load_model(MODEL_NAME)

    # 1) 固定閾値で有音区間検出（voiceCutting.pyと同じ）
    audio = AudioSegment.from_file(AUDIO_PATH)
    ranges = detect_nonsilent(
        audio,
        min_silence_len=MIN_SILENCE_LEN_MS,
        silence_thresh=SILENCE_THRESH_DBFS,
    )

    if not ranges:
        raise RuntimeError("no voiced segments detected (ranges is empty).")

    # 2) 後処理で範囲を整形（voiceCutting.pyと同じ）
    refined = postprocess_ranges(audio, ranges)

    # 3) 漢字抑止（元wordWhisper.pyと同様）
    suppress_tokens = build_suppress_tokens_for_kanji(LANGUAGE)

    options = whisper.DecodingOptions(
        language=LANGUAGE,
        task="transcribe",
        temperature=TEMPERATURE,
        fp16=False,
        suppress_tokens=suppress_tokens,
        without_timestamps=True,
    )

    lines = []
    lines.append(f"[language] {LANGUAGE}")
    lines.append(f"[detect_nonsilent] count={len(ranges)} min_silence_len_ms={MIN_SILENCE_LEN_MS} silence_thresh_dbfs={SILENCE_THRESH_DBFS}")
    lines.append(f"[postprocess] refined_count={len(refined)} target_count={TARGET_COUNT}")
    lines.append(f"[keep_silence_ms] {KEEP_SILENCE_MS}")
    lines.append(f"[suppress_tokens_kanji] count={len(suppress_tokens)}")
    lines.append("")
    lines.append("[segments]")

    idx = 0
    for (start_ms, end_ms) in refined:
        s = max(0, start_ms - KEEP_SILENCE_MS)
        e = min(len(audio), end_ms + KEEP_SILENCE_MS)
        chunk = audio[s:e]

        # voiceCutting.py と同じ「有効chunk判定」
        if not is_valid_chunk(chunk):
            continue

        idx += 1

        wav = audiosegment_to_whisper_wave(chunk)
        mel = whisper.log_mel_spectrogram(
            whisper.pad_or_trim(wav),
            n_mels=model.dims.n_mels
        ).to(model.device)

        result = whisper.decode(model, mel, options)
        text = (result.text or "").strip()

        # ★文字起こしした区間のタイムスタンプを出す
        lines.append(f"{idx:02d}\t{s/1000:.2f}-{e/1000:.2f}\t{text}")

    out_dir = os.path.dirname(OUTPUT_PATH) or "."
    os.makedirs(out_dir, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(f"[i] done -> {OUTPUT_PATH}", file=sys.stderr)

if __name__ == "__main__":
    main()
