# 従来の方法（漢字抑制＋無音区間分割＋1発音1行文字起こし）で
# whisperを使って文字起こしを行うスクリプト

import os
import sys
import time
import unicodedata
import numpy as np
import whisper

from pydub import AudioSegment
from pydub.silence import detect_nonsilent

# ====== ここだけ編集してください（元スクリプト踏襲） ======
AUDIO_PATH  = "word/10times_01/sakana.wav"
OUTPUT_PATH = "word/10times_01/sa/sakana_segmented.txt"
LANGUAGE    = "ja"        
MODEL_NAME  = "large-v3"
TEMPERATURE = 0.0

# ★「同じ発音を10回」前提：ここで“10回分”を区切って1回ずつ文字起こしする
EXPECTED_UTTERANCES = 10

# 無音区間分割パラメータ（元スクリプトと同等）
MIN_SILENCE_LEN_MS = 200
SILENCE_THRESH_OFFSET_DB = 14  # sound.dBFS - 14
# ========================================================

# ====== 漢字抑止（元スクリプトの必須処理：同等の構造） ======
def token_has_kanji(token_text: str) -> bool:
    for ch in token_text:
        name = unicodedata.name(ch, "")
        if "CJK UNIFIED IDEOGRAPH" in name:
            return True
        code = ord(ch)
        if (0x4E00 <= code <= 0x9FFF) or (0x3400 <= code <= 0x4DBF) or (0xF900 <= code <= 0xFAFF):
            return True
    return False

def build_suppress_tokens_for_kanji(lang: str):
    tok = whisper.tokenizer.get_tokenizer(multilingual=True, language=lang, task="transcribe")
    suppress = []
    for tid in range(tok.encoding.n_vocab):
        txt = tok.decode([tid])
        if txt and token_has_kanji(txt):
            suppress.append(tid)
    # 重複除去（安定化）
    return sorted(set(suppress))

# ====== AudioSegment -> whisper用 float32 wave ======
def audiosegment_to_whisper_wave(seg: AudioSegment) -> np.ndarray:
    seg = seg.set_channels(1)
    seg = seg.set_frame_rate(16000)
    samples = np.array(seg.get_array_of_samples()).astype(np.float32)
    # pydubは整数PCMなので正規化
    if seg.sample_width == 2:
        samples /= 32768.0
    elif seg.sample_width == 4:
        samples /= 2147483648.0
    else:
        # 8bit等のケース（基本想定外だが落とさない）
        maxv = float(2 ** (8 * seg.sample_width - 1))
        samples /= maxv
    return samples

def _merge_ranges_if_needed(ranges, max_gap_ms: int = 0):
    """検出の揺れで隙間が小さいものを結合（max_gap_ms=0なら結合しない）"""
    if not ranges:
        return []
    merged = [list(ranges[0])]
    for st, ed in ranges[1:]:
        prev = merged[-1]
        if st <= prev[1] + max_gap_ms:
            prev[1] = max(prev[1], ed)
        else:
            merged.append([st, ed])
    return merged

def main():
    if LANGUAGE != "ja":
        raise ValueError('LANGUAGE は前提により "ja" にしてください。')

    model = whisper.load_model(MODEL_NAME)

    # --- 無音で区間分割（10回分を区切るための範囲抽出） ---
    sound = AudioSegment.from_file(AUDIO_PATH)

    silence_thresh = sound.dBFS - SILENCE_THRESH_OFFSET_DB
    ranges = detect_nonsilent(
        sound,
        min_silence_len=MIN_SILENCE_LEN_MS,
        silence_thresh=silence_thresh,
        seek_step=1
    )

    if not ranges:
        # 最低限落とさない（この場合「10回分の区切り」は得られない）
        ranges = [[0, len(sound)]]

    # 必要なら結合（既定は結合なし）
    ranges = _merge_ranges_if_needed(ranges, max_gap_ms=0)

    # 10回分の想定に合わせて採用する範囲を決める
    ranges_used = ranges[:EXPECTED_UTTERANCES] if len(ranges) >= EXPECTED_UTTERANCES else ranges

    lang = "ja"

    # ★必須：漢字トークン抑止リストを作成★
    suppress_tokens = build_suppress_tokens_for_kanji(lang)

    # ★必須：DecodingOptionsに suppress_tokens を渡す★
    options = whisper.DecodingOptions(
        language=lang,
        task="transcribe",
        temperature=TEMPERATURE,
        fp16=False,
        suppress_tokens=suppress_tokens,
        without_timestamps=True,
    )

    # --- 区間ごとに decode（= 1回の発音 = 1行の文字起こし + タイムスタンプ） ---
    lines = []
    lines.append(f"[language] {lang}")
    lines.append(f"[detected_ranges] {len(ranges)}")
    lines.append(f"[used_ranges] {len(ranges_used)} (expected={EXPECTED_UTTERANCES})")
    lines.append(f"[silence] min_silence_len_ms={MIN_SILENCE_LEN_MS} silence_thresh_db={silence_thresh:.2f}")
    lines.append(f"[suppress_tokens_kanji] count={len(suppress_tokens)}")
    lines.append("")
    lines.append("[segments]")

    for i, (st_ms, ed_ms) in enumerate(ranges_used, start=1):
        seg = sound[st_ms:ed_ms]
        wav = audiosegment_to_whisper_wave(seg)

        mel = whisper.log_mel_spectrogram(
            whisper.pad_or_trim(wav),
            n_mels=model.dims.n_mels
        ).to(model.device)

        result = whisper.decode(model, mel, options)

        text = (result.text or "").strip()
        # 文字起こしした“音声時間”のタイムスタンプを必ず出す
        lines.append(f"{i:02d}\t{st_ms/1000:.2f}-{ed_ms/1000:.2f}\t{text}")

    out_dir = os.path.dirname(OUTPUT_PATH) or "."
    os.makedirs(out_dir, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(f"[i] done. -> {OUTPUT_PATH}", file=sys.stderr)

if __name__ == "__main__":
    main()
