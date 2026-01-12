from pathlib import Path

def rename_wavs_to_sequential(folder: str, digits: int = 2, start: int = 1, prefix: str = "chunkB_",) -> None:
    """
    指定フォルダ内の .wav を、01.wav, 02.wav ... のように連番リネームする。
    - 拡張子は維持（.wav）
    - ファイル内容は変更しない（リネームのみ）
    - 既存ファイル名と衝突しないよう、一旦テンポラリ名にしてから本命名にする
    """
    p = Path(folder)
    if not p.exists() or not p.is_dir():
        raise FileNotFoundError(f"フォルダが見つかりません: {p}")

    # 対象: 拡張子 .wav（大小区別しない）
    wavs = sorted([f for f in p.iterdir() if f.is_file() and f.suffix.lower() == ".wav"])

    if not wavs:
        print("対象のwavファイルが見つかりませんでした。")
        return

    # 1) 衝突回避のため、一旦テンポラリ名に変更
    tmp_map = []
    for i, f in enumerate(wavs, start=start):
        tmp_name = f"__tmp__{i:0{digits}d}__{f.name}"
        tmp_path = f.with_name(tmp_name)
        f.rename(tmp_path)
        tmp_map.append((tmp_path, i))

    # 2) 本命名: chunkB_01.wav, chunkB_02.wav ...
    for tmp_path, i in tmp_map:
        new_name = f"{prefix}{i:0{digits}d}.wav"
        new_path = tmp_path.with_name(new_name)
        if new_path.exists():
            raise FileExistsError(f"リネーム先が既に存在します: {new_path}")
        tmp_path.rename(new_path)

    print(f"完了: {len(wavs)}個のwavを {prefix}{start:0{digits}d}.wav から連番にリネームしました。")


if __name__ == "__main__":
    # ここを自分のフォルダパスに変更してください（例: r"C:\data\wavs" など）
    target_folder = r"C:\Users\edu01\Documents\GitHub\Shuron_YR\Sound\word_Ex1\10times_Ex1_C\tyakana5\cleaned_audio_chunks\voiced"
    rename_wavs_to_sequential(target_folder, digits=2, start=40, prefix="chunkC_")
