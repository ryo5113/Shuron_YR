# Whisperのmodel.transcribe()で特定語の出現時刻を取得するスクリプト

import os
import sys
import unicodedata
import whisper

# ====== ここだけ編集してください ======
AUDIO_PATH  = "word/10times_01/tyakana.wav"
OUTPUT_PATH = "word/10times_01/tya/tyakana_transcribe.txt"
LANGUAGE    = "ja"        # ★前提：日本語（"ja"固定）
MODEL_NAME  = "large-v3"
TEMPERATURE = 0.2
WORD_TIMESTAMPS = True    # True推奨（出現時刻を取るため）
TARGET_WORD = "ちゃかな"  # ★出現時刻を知りたい語
# ================================

# 句読点等の簡易除去（日本語向けに最低限）
PUNCT = set(" \t\n\r、。．，,.!?！？…・「」『』（）()[]{}<>＜＞【】\"'“”‘’：:；;ー-")

def _norm(s: str) -> str:
    s = (s or "").strip()
    s = "".join(ch for ch in s if ch not in PUNCT)
    return s

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
    return sorted(set(suppress))

def extract_occurrences_from_words(words, target: str):
    """
    words: [{'word': str, 'start': float, 'end': float, ...}, ...]
    target: 正規化した文字列（例: "さかな"）
    連結バッファ方式で target に一致した区間の (start,end) を返す
    """
    tgt = _norm(target)
    if not tgt:
        return []

    occ = []
    buf = ""
    buf_start = None
    for w in words:
        ww_raw = w.get("word", "")
        ww = _norm(ww_raw)
        if not ww:
            continue

        if buf == "":
            buf_start = w.get("start", None)

        buf += ww

        # 一致したら確定
        if buf == tgt:
            st = float(buf_start) if buf_start is not None else float(w.get("start", 0.0))
            ed = float(w.get("end", 0.0))
            occ.append((st, ed))
            buf = ""
            buf_start = None
        # targetより長くなったらリセット（単純）
        elif len(buf) > len(tgt):
            buf = ww
            buf_start = w.get("start", None)

            if buf == tgt:
                st = float(buf_start) if buf_start is not None else float(w.get("start", 0.0))
                ed = float(w.get("end", 0.0))
                occ.append((st, ed))
                buf = ""
                buf_start = None

    return occ

def main():
    if LANGUAGE != "ja":
        raise ValueError('LANGUAGE は前提により "ja" にしてください。')

    lang = "ja"
    model = whisper.load_model(MODEL_NAME)

    suppress_tokens = build_suppress_tokens_for_kanji(lang)

    # 1つの音声をまとめて transcribe（タイムスタンプはsegments/wordsで出す）
    result = model.transcribe(
        AUDIO_PATH,
        language=lang,
        task="transcribe",
        fp16=False,
        temperature=TEMPERATURE,
        suppress_tokens=suppress_tokens,
        word_timestamps=WORD_TIMESTAMPS,
        verbose=False,

        # 現行設定の維持
        no_speech_threshold=None,
        condition_on_previous_text=False,

        # 表記ガイド（必要がなければ空文字でもOK）
        initial_prompt="さかな。",
    )

    out_dir = os.path.dirname(OUTPUT_PATH) or "."
    os.makedirs(out_dir, exist_ok=True)

    # words を集約して出現抽出（全体で何回拾えたかを出す）
    all_words = []
    for seg in result.get("segments", []):
        if WORD_TIMESTAMPS and "words" in seg:
            all_words.extend(seg["words"])

    occurrences = extract_occurrences_from_words(all_words, TARGET_WORD)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        f.write(f"[language] {result.get('language')}\n")
        f.write(f"[text] {result.get('text','').strip()}\n\n")

        f.write(f"[suppress_tokens_kanji] count={len(suppress_tokens)}\n")
        f.write("\n")

        # ★文字起こしタイミングのタイムスタンプ（segments単位）を必ず出す
        f.write("[segments_text]\n")
        for i, seg in enumerate(result.get("segments", []), start=1):
            st = seg.get("start", None)
            ed = seg.get("end", None)
            txt = (seg.get("text", "") or "").strip()
            if st is None or ed is None:
                continue
            f.write(f"{i:02d}\t{float(st):.2f}-{float(ed):.2f}\t{txt}\n")
        f.write("\n")

        f.write(f"[target] {TARGET_WORD}\n")
        f.write(f"[occurrences] count={len(occurrences)}\n")
        for i, (st, ed) in enumerate(occurrences, start=1):
            f.write(f"{i:02d}\t{st:.2f}-{ed:.2f}\t{TARGET_WORD}\n")
        f.write("\n")

        # デバッグ用：segmentsとwordsの詳細も維持
        f.write("[segments]\n")
        for i, seg in enumerate(result.get("segments", []), start=1):
            st = seg.get("start", None)
            ed = seg.get("end", None)
            txt = (seg.get("text", "") or "").strip()
            f.write(f"{i:02d}\t{float(st):.2f}-{float(ed):.2f}\t{txt}\n")

            # seg内の主なデバッグ値（存在するものだけ）
            dbg_keys = ["no_speech_prob", "avg_logprob", "compression_ratio", "temperature"]
            dbg = []
            for k in dbg_keys:
                if k in seg:
                    dbg.append(f"{k}={seg.get(k)}")
            if dbg:
                f.write(f"    [seg_debug] " + " ".join(dbg) + "\n")

            # words（タイムスタンプ）
            if WORD_TIMESTAMPS and "words" in seg:
                for w in seg["words"]:
                    wst = w.get("start", None)
                    wed = w.get("end", None)
                    wwd = w.get("word", "")
                    if wst is None or wed is None:
                        continue
                    f.write(f"    - {float(wst):.2f}-{float(wed):.2f}\t{wwd}\n")

    print(f"[i] done. -> {OUTPUT_PATH}", file=sys.stderr)

if __name__ == "__main__":
    main()
