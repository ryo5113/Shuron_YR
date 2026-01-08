import wave
from pathlib import Path

# =========================
# 設定（ここだけ編集）
# =========================
INPUT_WAV = Path(r"word/10times_02/sa/re/cleaned_audio.wav")
OUTPUT_WAV = Path(r"word/10times_02/sa/re/clip_10s.wav")

START_SEC = 2.0          # 切り出し開始秒
END_SEC = 10.0           # 切り出し終了秒（Noneなら START_SEC+MAX_SECONDS）
MAX_SECONDS = 10.0       # 10秒以内に収めたい（Instant Clone向け）
# =========================

def cut_wav_segment(
    input_wav: Path,
    output_wav: Path,
    start_sec: float,
    end_sec: float | None,
    max_seconds: float | None = None,
):
    if start_sec < 0:
        raise ValueError("START_SEC must be >= 0.")

    with wave.open(str(input_wav), "rb") as r:
        nch = r.getnchannels()
        sampwidth = r.getsampwidth()  # bytes/sample
        fr = r.getframerate()         # samples/sec
        nframes = r.getnframes()

        total_sec = nframes / fr

        # end_sec の決定
        if end_sec is None:
            end_sec = start_sec + (max_seconds if max_seconds is not None else total_sec)

        # MAX_SECONDS 制限（超えたら自動トリム）
        if max_seconds is not None and (end_sec - start_sec) > max_seconds:
            end_sec = start_sec + max_seconds

        # 範囲クリップ
        start_sec_clamped = max(0.0, min(start_sec, total_sec))
        end_sec_clamped = max(0.0, min(end_sec, total_sec))

        if end_sec_clamped <= start_sec_clamped:
            raise ValueError(f"Invalid range: start={start_sec_clamped}, end={end_sec_clamped}")

        start_frame = int(round(start_sec_clamped * fr))
        end_frame = int(round(end_sec_clamped * fr))
        cut_frames = end_frame - start_frame

        r.setpos(start_frame)
        frames = r.readframes(cut_frames)

        with wave.open(str(output_wav), "wb") as w:
            w.setnchannels(nch)
            w.setsampwidth(sampwidth)
            w.setframerate(fr)
            w.writeframes(frames)

    print(f"[OK] {input_wav} -> {output_wav}")
    print(f"     range: {start_sec_clamped:.3f}s - {end_sec_clamped:.3f}s "
          f"({end_sec_clamped - start_sec_clamped:.3f}s), sr={fr}, ch={nch}, sampwidth={sampwidth}")

if __name__ == "__main__":
    cut_wav_segment(
        input_wav=INPUT_WAV,
        output_wav=OUTPUT_WAV,
        start_sec=START_SEC,
        end_sec=END_SEC,
        max_seconds=MAX_SECONDS,
    )
