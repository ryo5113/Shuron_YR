#!/usr/bin/env python3
"""
overlay_images.py — 2つの画像を透過合成するスクリプト

使い方:
  python overlay_images.py path/to/top.png path/to/bottom.jpg -o out.png --alpha 0.5

主なオプション:
  --alpha     上に載せる画像(第1引数)の透過率 (0.0〜1.0, デフォルト0.5)
  --offset    上画像の配置オフセット (x y) 例: --offset 10 20
  --no-resize サイズが異なるときに上画像をリサイズしない（デフォルトは自動で上画像を縮放）
  --swap      画像の上下を入れ替える（第1引数を下、第2引数を上に）
  --out,-o    出力ファイルパス（拡張子でPNG/JPG等が決まります）
"""

from PIL import Image
import argparse
from pathlib import Path
import sys

def apply_alpha(img_rgba: Image.Image, alpha: float) -> Image.Image:
    """画像のアルファチャンネルに透過率を掛ける"""
    r, g, b, a = img_rgba.split()
    # 0〜255 の範囲でスケール
    a = a.point(lambda x: int(x * alpha))
    return Image.merge("RGBA", (r, g, b, a))

def overlay(
    top_path: Path,
    bottom_path: Path,
    out_path: Path,
    alpha: float = 0.5,
    offset=(0, 0),
    resize_top=True,
):
    # 読み込み & RGBA 化
    top = Image.open(top_path).convert("RGBA")
    bottom = Image.open(bottom_path).convert("RGBA")

    # 必要なら上画像を下画像のサイズへ合わせる
    if resize_top and top.size != bottom.size:
        top = top.resize(bottom.size, Image.LANCZOS)

    # 透過率を適用
    if not (0.0 <= alpha <= 1.0):
        raise ValueError("--alpha は 0.0〜1.0 の範囲で指定してください。")
    top = apply_alpha(top, alpha)

    # 合成キャンバス（下画像のコピー）
    canvas = bottom.copy()

    # サイズが一致しない場合はオフセットで貼り付け（alpha 利用のため mask に上画像の A を使う）
    if top.size != canvas.size:
        # top をそのまま貼る（はみ出しは切り落とし）
        canvas.paste(top, offset, mask=top.split()[3])  # 3=Alpha
    else:
        # 同サイズなら alpha_composite でもOK。オフセットがある場合は paste を使う
        if offset == (0, 0):
            canvas = Image.alpha_composite(canvas, top)
        else:
            canvas.paste(top, offset, mask=top.split()[3])

    # 出力
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # PNG なら透過保持、JPG なら自動でRGBに変換
    if out_path.suffix.lower() in [".jpg", ".jpeg", ".webp"]:
        canvas = canvas.convert("RGB")
    canvas.save(out_path)
    return out_path

def parse_args():
    p = argparse.ArgumentParser(description="2つの画像を透過合成します。")
    p.add_argument("top", type=Path, help="上に載せる画像のパス")
    p.add_argument("bottom", type=Path, help="下に敷く画像のパス")
    p.add_argument("--alpha", type=float, default=0.5, help="上画像の透過率 0.0〜1.0 (default: 0.5)")
    p.add_argument("--offset", type=int, nargs=2, default=(0, 0), metavar=("X", "Y"),
                   help="上画像の左上の配置オフセット (default: 0 0)")
    p.add_argument("--no-resize", action="store_true",
                   help="サイズ不一致でも上画像をリサイズせずに貼り付ける")
    p.add_argument("--swap", action="store_true",
                   help="上下の画像を入れ替えて合成する")
    p.add_argument("-o", "--out", type=Path, default=Path("overlay.png"),
                   help="出力先ファイルパス (default: overlay.png)")
    return p.parse_args()

def main():
    args = parse_args()

    # 画像の入れ替えに対応
    top_path, bottom_path = (args.top, args.bottom)
    if args.swap:
        top_path, bottom_path = bottom_path, top_path

    try:
        out = overlay(
            top_path=top_path,
            bottom_path=bottom_path,
            out_path=args.out,
            alpha=args.alpha,
            offset=tuple(args.offset),
            resize_top=not args.no_resize,
        )
        print(f"Saved: {out}")
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
