from pathlib import Path
from pydub import AudioSegment
from pydub.silence import detect_nonsilent

# ====== ここに処理したいファイルのパスを複数指定 ======
INPUT_FILES = [
    Path(r"sata_ML6/sa/cleaned_audio.wav"),
    Path(r"sata_ML6/sha/cleaned_audio.wav"),
    Path(r"sata_ML6/tha/cleaned_audio.wav"),
    Path(r"sata_ML6/tya/cleaned_audio.wav"),
    Path(r"sata_ML6/ta/cleaned_audio.wav"),
]
# ==================================================

# 出力先（各ファイルごとにサブフォルダを作ってchunkを書き出す）
OUTPUT_ROOT = Path(r"voiced_chunks")

# 無音判定パラメータ（閾値は変えない）
MIN_SILENCE_LEN_MS = 300
SILENCE_THRESH_DBFS = -60.0
KEEP_SILENCE_MS = 0  # 前後に残す無音(ms)。不要なら0のままでOK :contentReference[oaicite:1]{index=1}


def process_one(in_path: Path) -> None:
    if not in_path.exists():
        print(f"[SKIP] not found: {in_path}")
        return

    audio = AudioSegment.from_file(in_path)  # 読み込み :contentReference[oaicite:2]{index=2}

    ranges = detect_nonsilent(
        audio,
        min_silence_len=MIN_SILENCE_LEN_MS,
        silence_thresh=SILENCE_THRESH_DBFS,
    )  # [[start_ms, end_ms], ...] :contentReference[oaicite:3]{index=3}

    out_dir = in_path.parent / f"{in_path.stem}_chunks"
    out_dir.mkdir(parents=True, exist_ok=True)

    if not ranges:
        print(f"[NG] no voiced segments: {in_path.name}")
        return

    for i, (start_ms, end_ms) in enumerate(ranges, start=1):
        s = max(0, start_ms - KEEP_SILENCE_MS)
        e = min(len(audio), end_ms + KEEP_SILENCE_MS)
        chunk = audio[s:e]

        out_path = out_dir / f"chunk_{i:02d}.wav"
        chunk.export(out_path, format="wav")  # 保存(export) :contentReference[oaicite:4]{index=4}

    print(f"[OK] {in_path.name} -> {len(ranges)} chunks ({out_dir})")


def main():
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    for f in INPUT_FILES:
        process_one(f)


if __name__ == "__main__":
    main()
