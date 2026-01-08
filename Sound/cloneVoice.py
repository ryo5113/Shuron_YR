import os
import wave
import tempfile
from pathlib import Path

from cartesia import Cartesia


# =========================
# 設定（ここだけ編集）
# =========================
CARTESIA_API_KEY = "sk_car_BUNMtmrNUmhTgeMVXZpLJ2"  # 環境変数推奨 
INPUT_WAV = Path(r"word/10times_02/sa/re/cleaned_audio.wav")          # 入力（PCM WAV想定）
START_SEC = 2.0                         # 切り出し開始秒
MAX_CLIP_SECONDS = 10.0                  # Clone推奨は約5秒 

VOICE_NAME = "my_instant_clone"
VOICE_DESCRIPTION = "created by python"
LANGUAGE = "ja"                         
CLONE_MODE = "similarity"               # PyPI例: "similarity" or "stability" 
ENHANCE = False                         # PyPI例

TTS_TEXT = "さかな。"
OUTPUT_WAV = Path(r"word/10times_02/sa/re/tts_output.wav")
TTS_MODEL_ID = "sonic-3"                
OUT_SAMPLE_RATE = 48000
OUT_ENCODING = "pcm_s16le"              # 例（SDK/Docsで利用されるencodingの一つ）


def cut_wav_segment_to_tempfile(
    input_wav: Path,
    start_sec: float,
    max_seconds: float,
) -> Path:
    """入力WAVから [start_sec, start_sec+max_seconds] を切り出したWAVを一時ファイルに保存して返す。"""
    if start_sec < 0:
        raise ValueError("START_SEC must be >= 0.")

    with wave.open(str(input_wav), "rb") as r:
        nch = r.getnchannels()
        sampwidth = r.getsampwidth()
        fr = r.getframerate()
        nframes = r.getnframes()

        total_sec = nframes / fr
        end_sec = min(total_sec, start_sec + max_seconds)
        start_sec = min(max(0.0, start_sec), total_sec)

        if end_sec <= start_sec:
            raise ValueError(f"Invalid range: start={start_sec}, end={end_sec}")

        start_frame = int(round(start_sec * fr))
        end_frame = int(round(end_sec * fr))
        cut_frames = end_frame - start_frame

        r.setpos(start_frame)
        frames = r.readframes(cut_frames)

    # Windowsでも安全に扱えるよう delete=False
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp_path = Path(tmp.name)
    tmp.close()

    with wave.open(str(tmp_path), "wb") as w:
        w.setnchannels(nch)
        w.setsampwidth(sampwidth)
        w.setframerate(fr)
        w.writeframes(frames)

    return tmp_path


def main():
    if not CARTESIA_API_KEY:
        raise RuntimeError("環境変数 CARTESIA_API_KEY が未設定です。")  # SDK例 

    client = Cartesia(api_key=CARTESIA_API_KEY)

    # 1) 入力WAVからクローン用クリップ（最大5秒）を作る
    clip_path = cut_wav_segment_to_tempfile(
        input_wav=INPUT_WAV,
        start_sec=START_SEC,
        max_seconds=MAX_CLIP_SECONDS,
    )
    print(f"[cut] clip saved: {clip_path}")

    try:
        # 2) Instant Clone（voice_id 作成）
        with open(clip_path, "rb") as f:
            cloned_voice = client.voices.clone(
                clip=f,
                name=VOICE_NAME,
                language=LANGUAGE,
                mode=CLONE_MODE,       
                enhance=ENHANCE,        
                description=VOICE_DESCRIPTION,
            )
        voice_id = cloned_voice.id
        print(f"[clone] voice_id = {voice_id}")  # Clone APIは voice id を返す

        # 3) その voice_id で TTS → WAV保存
        bytes_iter = client.tts.bytes(
            model_id=TTS_MODEL_ID,
            transcript=TTS_TEXT,
            voice={"mode": "id", "id": voice_id},
            language=LANGUAGE,
            output_format={
                "container": "wav",
                "sample_rate": OUT_SAMPLE_RATE,
                "encoding": OUT_ENCODING,
            },
        )  # TTS(Bytes)

        with open(OUTPUT_WAV, "wb") as out:
            for chunk in bytes_iter:
                out.write(chunk)

        print(f"[tts] saved: {OUTPUT_WAV}")

    finally:
        # 一時ファイル削除
        try:
            clip_path.unlink(missing_ok=True)
        except Exception:
            pass


if __name__ == "__main__":
    main()
