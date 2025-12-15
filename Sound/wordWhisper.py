import os
import sys
import unicodedata
import whisper

# ====== ここだけ編集してください（soundWhisper.pyと同じ） ======
AUDIO_PATH  = "sakana.wav"          # 入力音声ファイル
OUTPUT_PATH = "sakana.txt"          # 出力テキストファイル
LANGUAGE    = "ja"                  # "ja" 推奨（Noneなら自動判定）
MODEL_NAME  = "large-v3"            # 現状踏襲
TEMPERATURE = 0.0                   # 現状踏襲
# ============================================================

def token_has_kanji(token_text: str) -> bool:
    """
    トークン文字列に「漢字（CJK統合漢字・拡張など）」が含まれるか判定。
    Unicode名に 'CJK UNIFIED IDEOGRAPH' が含まれる文字を漢字として扱う。
    """
    for ch in token_text:
        name = unicodedata.name(ch, "")
        if "CJK UNIFIED IDEOGRAPH" in name:
            return True
        # 互換漢字等でUnicode名が異なるケースもあるため、代表範囲も併用
        code = ord(ch)
        if (0x4E00 <= code <= 0x9FFF) or (0x3400 <= code <= 0x4DBF) or (0xF900 <= code <= 0xFAFF):
            return True
    return False

def build_suppress_tokens_for_kanji(language: str):
    """
    Whisperの語彙全体から「漢字を含むトークン」を抽出し、そのIDを suppress_tokens にする。
    tokenizerの仕組みは openai/whisper の tokenizer 実装に基づく。:contentReference[oaicite:3]{index=3}
    """
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

    # 念のため重複除去
    suppress = sorted(set(suppress))
    return suppress

def main():
    if not os.path.isfile(AUDIO_PATH):
        print(f"入力ファイルが見つかりません: {AUDIO_PATH}", file=sys.stderr)
        sys.exit(1)

    # モデルロード
    model = whisper.load_model(MODEL_NAME)

    # 音声読み込み（Whisper側でmono化＆必要なら16kHzへリサンプル）:contentReference[oaicite:4]{index=4}
    audio = whisper.load_audio(AUDIO_PATH)
    audio = whisper.pad_or_trim(audio)

    mel = whisper.log_mel_spectrogram(audio, n_mels=model.dims.n_mels).to(model.device)

    # 漢字トークン抑止リスト作成
    lang = LANGUAGE
    if lang is None:
        # 自動判定する場合：ここでは簡易にdetect_languageを使用
        _, probs = model.detect_language(mel)
        lang = max(probs, key=probs.get)

    suppress_tokens = build_suppress_tokens_for_kanji(lang)

    # デコード（transcribe()ではなく decode() で suppress_tokens を確実に適用）:contentReference[oaicite:5]{index=5}
    options = whisper.DecodingOptions(
        language=lang,
        task="transcribe",
        temperature=TEMPERATURE,
        fp16=False,                 # CPU想定
        suppress_tokens=suppress_tokens,
        without_timestamps=True,
    )

    result = whisper.decode(model, mel, options)
    text = (result.text or "").strip()

    out_dir = os.path.dirname(OUTPUT_PATH) or "."
    os.makedirs(out_dir, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        f.write(text + "\n")

    print(f"[i] done. language={lang} -> {OUTPUT_PATH}", file=sys.stderr)
    print(f"[i] suppress_tokens(kanji) count = {len(suppress_tokens)}", file=sys.stderr)

if __name__ == "__main__":
    main()
