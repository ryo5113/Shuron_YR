from pathlib import Path
import wave

def wav_duration_seconds(wav_path: Path) -> float:
    with wave.open(str(wav_path), "rb") as wf:
        frames = wf.getnframes()
        rate = wf.getframerate()
        return frames / float(rate)

folder = Path(input("WAVフォルダのパスを入力してください: ").strip().strip('"'))

if not folder.is_dir():
    raise FileNotFoundError(f"フォルダが見つかりません: {folder}")

wav_files = sorted(folder.glob("*.wav"))

if not wav_files:
    print("このフォルダには .wav ファイルがありません。")
else:
    for p in wav_files:
        try:
            dur = wav_duration_seconds(p)
            print(f"{p.name}\t{dur:.3f} sec")
        except wave.Error as e:
            # 圧縮WAVなどで読めない場合にここに来ることがあります
            print(f"{p.name}\t読み取り失敗: {e}")
