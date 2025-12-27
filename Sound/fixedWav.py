from pathlib import Path
import subprocess

FFMPEG = "ffmpeg"  # ffmpeg.exe がPATHに通っている想定

def main():
    dataset_root = Path(r"C:\Users\edu01\Documents\GitHub\Shuron_YR\Sound\ML_wav_dataset")
    bad_list = dataset_root / "bad_wavs.txt"
    out_root = dataset_root / "fixed_wavs"
    out_root.mkdir(parents=True, exist_ok=True)

    lines = bad_list.read_text(encoding="utf-8").splitlines()

    for line in lines:
        if not line.strip():
            continue
        in_path_str = line.split("\t", 1)[0]  # 1列目がパス
        in_path = Path(in_path_str)

        # 元の相対パス構造（ラベル/ファイル名.wav）を保って出力
        rel = in_path.relative_to(dataset_root)
        out_path = out_root / rel
        out_path.parent.mkdir(parents=True, exist_ok=True)

        cmd = [
            FFMPEG, "-y",
            "-i", str(in_path),
            "-c:a", "pcm_s16le",
            "-f", "wav",
            str(out_path),
        ]

        print("CONVERT:", in_path, "->", out_path)
        subprocess.run(cmd, check=False)

if __name__ == "__main__":
    main()
