import os
import random
import shutil
import hashlib
from pathlib import Path
import tkinter as tk
from tkinter import filedialog

IMG_EXTS = {".png", ".jpg", ".jpeg", ".webp"}
TRAIN_RATIO = 0.7
SEED = 42

def pick_root_dir() -> Path:
    root = tk.Tk()
    root.withdraw()
    folder = filedialog.askdirectory(title="ラベル分け済み画像データセットのルートを選択")
    root.destroy()
    return Path(folder) if folder else None

def list_images(label_dir: Path):
    # 既存スクリプト同様、サブフォルダも含めて再帰で拾う（rglob）
    # ※Path.rglob は Python の pathlib の機能。:contentReference[oaicite:3]{index=3}
    return [p for p in label_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXTS]

def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    # hashlib.sha256 の利用は Python 公式ドキュメントの通り。:contentReference[oaicite:4]{index=4}
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()

def safe_filename(prefix: str, original: str) -> str:
    # ファイル名衝突を避けるためハッシュ先頭を付ける
    return f"{prefix}_{original}"

def main():
    src_root = pick_root_dir()
    if src_root is None:
        print("キャンセルされました。")
        return

    # ルート直下のサブフォルダをラベルとして扱う
    label_dirs = [d for d in src_root.iterdir() if d.is_dir()]
    if not label_dirs:
        print("ルート直下にラベルフォルダが見つかりません。")
        return

    out_root = src_root.parent / (src_root.name + "_yolo_cls")
    train_root = out_root / "train"
    val_root = out_root / "val"

    # 重要: 既存出力があると、前回分が残って枚数が増える/混ざるため削除して作り直す
    # shutil.rmtree は「ディレクトリツリー全体を削除」する関数。:contentReference[oaicite:5]{index=5}
    if out_root.exists():
        shutil.rmtree(out_root)

    train_root.mkdir(parents=True, exist_ok=True)
    val_root.mkdir(parents=True, exist_ok=True)

    random.seed(SEED)

    grand_train = 0
    grand_val = 0

    for ld in sorted(label_dirs):
        label = ld.name
        imgs = list_images(ld)
        if not imgs:
            print(f"[{label}] 画像なし -> skip")
            continue

        # --- (1) 内容ハッシュで重複排除（同一内容の画像は1枚だけ残す） ---
        hash_to_path = {}
        dup_count = 0
        for p in sorted(imgs):
            try:
                h = sha256_file(p)
            except Exception as e:
                print(f"[{label}] hash失敗: {p} ({e}) -> skip")
                continue

            if h in hash_to_path:
                dup_count += 1
                continue
            hash_to_path[h] = p

        unique_hashes = list(hash_to_path.keys())
        random.shuffle(unique_hashes)

        # --- (2) 7:3 に分割（同一ハッシュがtrain/valに跨らない） ---
        n_total = len(unique_hashes)
        n_train = int(n_total * TRAIN_RATIO)
        train_hashes = unique_hashes[:n_train]
        val_hashes = unique_hashes[n_train:]

        # 念のため重複チェック
        if set(train_hashes) & set(val_hashes):
            raise RuntimeError(f"[{label}] train/val に同一画像が跨っています（想定外）")

        (train_root / label).mkdir(parents=True, exist_ok=True)
        (val_root / label).mkdir(parents=True, exist_ok=True)

        # --- (3) コピー（ファイル名衝突回避のためハッシュを付与） ---
        for h in train_hashes:
            src = hash_to_path[h]
            dst = train_root / label / safe_filename(h[:12], src.name)
            # shutil.copy2 はメタデータも含めてコピー。:contentReference[oaicite:6]{index=6}
            shutil.copy2(src, dst)

        for h in val_hashes:
            src = hash_to_path[h]
            dst = val_root / label / safe_filename(h[:12], src.name)
            shutil.copy2(src, dst)

        grand_train += len(train_hashes)
        grand_val += len(val_hashes)

        print(
            f"[{label}] raw={len(imgs)} unique={n_total} "
            f"(dups_removed={dup_count}) train={len(train_hashes)} val={len(val_hashes)}"
        )

    print("\n=== DONE ===")
    print("Output:", out_root)
    print("Train:", grand_train, " Val:", grand_val)
    print("\n期待される構造:")
    print(out_root)
    print("  train/<label>/*.png")
    print("  val/<label>/*.png")

if __name__ == "__main__":
    main()
