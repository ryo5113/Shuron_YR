import os
import sys
import unicodedata
import numpy as np
import whisper

from pydub import AudioSegment
from pydub.silence import detect_nonsilent

# ====== ここだけ編集してください（元スクリプト踏襲） ======
AUDIO_PATH  = "word/1time/shakana.wav"
OUTPUT_PATH = "word/1time/shakana_segmented.txt"
LANGUAGE    = "ja"        # Noneなら自動判定
MODEL_NAME  = "large-v3"
TEMPERATURE = 0.0
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

def build_suppress_tokens_for_kanji(language: str):
    tok = whisper.tokenizer.get_tokenizer(multilingual=True, language=language, task="transcribe")
    n_vocab = tok.encoding.n_vocab
    suppress = []
    for tid in range(n_vocab):
        try:
            s = tok.decode([tid])
        except Exception:
            continue
        if s and token_has_kanji(s):
            suppress.append(tid)
    return sorted(set(suppress))
# ========================================================

def audiosegment_to_whisper_wave(seg: AudioSegment) -> np.ndarray:
    seg = seg.set_channels(1).set_frame_rate(16000)
    samples = np.array(seg.get_array_of_samples())

    if seg.sample_width == 2:          # int16
        wav = samples.astype(np.float32) / 32768.0
    elif seg.sample_width == 4:        # int32
        wav = samples.astype(np.float32) / 2147483648.0
    else:
        wav = samples.astype(np.float32)
        maxv = np.max(np.abs(wav)) + 1e-9
        wav = wav / maxv
    return wav

def main():
    if not os.path.isfile(AUDIO_PATH):
        print(f"入力ファイルが見つかりません: {AUDIO_PATH}", file=sys.stderr)
        sys.exit(1)

    model = whisper.load_model(MODEL_NAME)

    # --- 無音で区間分割（音声は変更しない：区切り情報だけ抽出） ---
    sound = AudioSegment.from_file(AUDIO_PATH)

    min_silence_len = 200
    silence_thresh  = sound.dBFS - 14

    ranges = detect_nonsilent(
        sound,
        min_silence_len=min_silence_len,
        silence_thresh=silence_thresh,
        seek_step=1
    )
    if not ranges:
        ranges = [[0, len(sound)]]

    # --- 言語決定（元スクリプトの流れを踏襲） ---
    lang = LANGUAGE
    if lang is None:
        first_seg = sound[ranges[0][0]:ranges[0][1]]
        wav0 = audiosegment_to_whisper_wave(first_seg)
        mel0 = whisper.log_mel_spectrogram(
            whisper.pad_or_trim(wav0),
            n_mels=model.dims.n_mels
        ).to(model.device)
        _, probs = model.detect_language(mel0)
        lang = max(probs, key=probs.get)

    # ★必須：漢字トークン抑止リストを作成（元スクリプトと同じ役割）★
    suppress_tokens = build_suppress_tokens_for_kanji(lang)

    # ★必須：DecodingOptionsに suppress_tokens を渡す（元スクリプトと同じ）★
    options = whisper.DecodingOptions(
        language=lang,
        task="transcribe",
        temperature=TEMPERATURE,
        fp16=False,
        suppress_tokens=suppress_tokens,
        without_timestamps=True,
    )

    # --- 区間ごとに decode（ここが「回数を明確に分ける」本体） ---
    lines = []
    for i, (st_ms, ed_ms) in enumerate(ranges, start=1):
        seg = sound[st_ms:ed_ms]
        wav = audiosegment_to_whisper_wave(seg)

        mel = whisper.log_mel_spectrogram(
            whisper.pad_or_trim(wav),
            n_mels=model.dims.n_mels
        ).to(model.device)

        result = whisper.decode(model, mel, options)
        text = (result.text or "").strip()
        if text:
            lines.append(f"{i:02d}\t{st_ms/1000:.2f}-{ed_ms/1000:.2f}\t{text}")

    out_dir = os.path.dirname(OUTPUT_PATH) or "."
    os.makedirs(out_dir, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(f"[i] done. language={lang} -> {OUTPUT_PATH}", file=sys.stderr)
    print(f"[i] suppress_tokens(kanji) count = {len(suppress_tokens)}", file=sys.stderr)
    print(f"[i] segments_written = {len(lines)} / detected = {len(ranges)}", file=sys.stderr)

if __name__ == "__main__":
    main()
