import os
import sys
import torch
import whisper

# ====== ここだけ編集してください ======
AUDIO_PATH  = "ae_v3/first/cleaned_audio.wav"          # 入力音声ファイル
OUTPUT_PATH = "ae_v3/first/output.txt"         # 出力テキストファイル
LANGUAGE    = "ja"                  # 例: "ja" / "en"。Noneなら自動判定
MODEL_NAME  = "large-v3"            # large固定（推奨: large-v3）
TEMPERATURE = 0.0                   # 安定寄り
# =====================================

def main():
    if not os.path.isfile(AUDIO_PATH):
        print(f"入力ファイルが見つかりません: {AUDIO_PATH}", file=sys.stderr)
        sys.exit(1)

    #device = "cuda" if torch.cuda.is_available() else "cpu" #CUDA調整が必要
    device = "cpu"
    print(f"[i] device: {device}", file=sys.stderr)
    print(f"[i] loading model: {MODEL_NAME}", file=sys.stderr)

    # モデル読み込み（large固定）
    model = whisper.load_model(MODEL_NAME, device=device)

    # GPUなら fp16 を有効化
    use_fp16 = (device == "cuda")

    print(f"[i] transcribing... (fp16={use_fp16}, language={LANGUAGE})", file=sys.stderr)
    result = model.transcribe(
        AUDIO_PATH,
        language=LANGUAGE,    # Noneで自動判定
        fp16=use_fp16,
        temperature=TEMPERATURE,
    )

    text = (result.get("text") or "").strip()

    # 出力フォルダを自動作成
    out_dir = os.path.dirname(OUTPUT_PATH) or "."
    os.makedirs(out_dir, exist_ok=True)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        f.write(text + "\n")

    det_lang = result.get("language")
    print(f"[i] done. detected_language={det_lang} -> {OUTPUT_PATH}", file=sys.stderr)

if __name__ == "__main__":
    main()