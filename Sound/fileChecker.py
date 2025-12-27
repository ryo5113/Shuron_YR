from pathlib import Path
import tkinter as tk
from tkinter import filedialog

from scipy.io import wavfile  # wavfile.read を使う :contentReference[oaicite:2]{index=2}


def pick_root_dir() -> Path:
    root = tk.Tk()
    root.withdraw()
    folder = filedialog.askdirectory(title="ML_wav_dataset（ルート）を選択")  # :contentReference[oaicite:3]{index=3}
    root.destroy()
    return Path(folder) if folder else None


def main():
    root_dir = pick_root_dir()
    if root_dir is None:
        print("キャンセルされました。")
        return

    wav_paths = sorted(root_dir.rglob("*.wav"))  # 再帰的に全wav取得 :contentReference[oaicite:4]{index=4}
    print(f"検査対象 wav 数: {len(wav_paths)}")

    bad = []
    for p in wav_paths:
        try:
            wavfile.read(str(p))  # 読めなければ例外が出る :contentReference[oaicite:5]{index=5}
        except Exception as e:
            bad.append((str(p), repr(e)))
            print("BAD:", p)
            print("  ", e)

    # 結果保存
    out_txt = root_dir / "bad_wavs.txt"
    with out_txt.open("w", encoding="utf-8") as f:
        for path, err in bad:
            f.write(f"{path}\t{err}\n")

    print(f"\n読めないwav数: {len(bad)}")
    print(f"一覧出力: {out_txt}")


if __name__ == "__main__":
    main()
