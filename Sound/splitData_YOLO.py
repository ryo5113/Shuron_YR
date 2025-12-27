import os
import random
import shutil
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
    return [p for p in label_dir.rglob("*") if p.suffix.lower() in IMG_EXTS]

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
    train_root.mkdir(parents=True, exist_ok=True)
    val_root.mkdir(parents=True, exist_ok=True)

    random.seed(SEED)

    total_train = 0
    total_val = 0

    for ld in sorted(label_dirs):
        label = ld.name
        imgs = list_images(ld)
        if not imgs:
            continue

        random.shuffle(imgs)
        n_train = int(len(imgs) * TRAIN_RATIO)

        train_imgs = imgs[:n_train]
        val_imgs = imgs[n_train:]

        (train_root / label).mkdir(parents=True, exist_ok=True)
        (val_root / label).mkdir(parents=True, exist_ok=True)

        for p in train_imgs:
            dst = train_root / label / p.name
            shutil.copy2(p, dst)

        for p in val_imgs:
            dst = val_root / label / p.name
            shutil.copy2(p, dst)

        total_train += len(train_imgs)
        total_val += len(val_imgs)

        print(f"[{label}] total={len(imgs)} train={len(train_imgs)} val={len(val_imgs)}")

    print("\n=== DONE ===")
    print("Output:", out_root)
    print("Train:", total_train, " Val:", total_val)
    print("\n期待される構造:")
    print(out_root)
    print("  train/<label>/*.png")
    print("  val/<label>/*.png")

if __name__ == "__main__":
    main()
