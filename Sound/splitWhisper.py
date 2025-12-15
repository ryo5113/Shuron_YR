# split_by_ranges_and_whisper.py
from pathlib import Path
from pydub import AudioSegment
import whisper

# ==========
# 入出力（スクリプト内で指定）
# ==========
INPUT_AUDIO_PATH = r"ae_v3/first/cleaned_audio.wav"
OUTPUT_DIR = r"./out_chunks_ranges"
WHISPER_MODEL_NAME = "large"

# ==========
# 区間指定（秒）
# ここに「(開始秒, 終了秒)」を好きな個数だけ列挙（=切り出し個数）
# ==========
SEGMENTS_SEC = [
    (0.5, 2.0),
    (2.5, 4.0),
    (4.5, 6.0),
    (6.5, 8.0),
    (8.5, 10.0),
    # 例：必要なだけ追加
]

def sec_to_ms(sec: float) -> int:
    return int(round(sec * 1000.0))

def main():
    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    audio = AudioSegment.from_file(INPUT_AUDIO_PATH)
    total_ms = len(audio)

    model = whisper.load_model(WHISPER_MODEL_NAME)

    summary_txt_path = out_dir / "transcript_by_ranges.txt"
    with open(summary_txt_path, "w", encoding="utf-8") as fsum:
        fsum.write(f"INPUT: {INPUT_AUDIO_PATH}\n")
        fsum.write(f"WHISPER_MODEL: {WHISPER_MODEL_NAME}\n")
        fsum.write(f"NUM_SEGMENTS: {len(SEGMENTS_SEC)}\n\n")

        for idx, (start_s, end_s) in enumerate(SEGMENTS_SEC):
            start_ms = sec_to_ms(start_s)
            end_ms = sec_to_ms(end_s)

            # 簡易バリデーション
            if start_ms < 0 or end_ms <= start_ms:
                raise ValueError(f"区間が不正です: idx={idx}, ({start_s}, {end_s})")
            if start_ms >= total_ms:
                raise ValueError(f"開始が音声長を超えています: idx={idx}, start={start_s}s, len={total_ms/1000:.3f}s")
            if end_ms > total_ms:
                end_ms = total_ms  # 末尾超えは末尾に丸める

            # pydub: ミリ秒でスライス
            chunk = audio[start_ms:end_ms]

            # 保存
            chunk_wav = out_dir / f"chunk_{idx:04d}_{start_s:.3f}-{end_ms/1000:.3f}.wav"
            chunk.export(chunk_wav, format="wav")

            # Whisper文字起こし
            result = model.transcribe(str(chunk_wav), fp16=False, language="ja")
            text = (result.get("text") or "").strip()

            # チャンク別テキスト
            chunk_txt = out_dir / f"chunk_{idx:04d}_{start_s:.3f}-{end_ms/1000:.3f}.txt"
            with open(chunk_txt, "w", encoding="utf-8") as fc:
                fc.write(text + "\n")

            # まとめ
            fsum.write(f"[{idx:04d}] {start_s:.3f}-{end_ms/1000:.3f}s\n")
            fsum.write(text + "\n\n")

    print(f"Done. Outputs are in: {out_dir.resolve()}")

if __name__ == "__main__":
    main()
