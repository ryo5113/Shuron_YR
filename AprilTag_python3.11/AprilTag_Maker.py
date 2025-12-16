import os
import cv2

# ============================================
# ★ ここで拡大倍率を設定（例：100倍 → 100）
# ============================================
SCALE = 10   # ← ここを 50 や 200 に変えるだけ！

# 元フォルダ
SRC_DIR = "tag36h11_png"

# 出力フォルダ名を自動生成
DST_DIR = f"tag36h11_pngx{SCALE}"

# フォルダがなければ作成
os.makedirs(DST_DIR, exist_ok=True)

# 対象が PNG ファイルだけになるようにする
files = [f for f in os.listdir(SRC_DIR) if f.lower().endswith(".png")]

print(f"Found {len(files)} PNG files. Expanding {SCALE}x ...")

for filename in files:
    src_path = os.path.join(SRC_DIR, filename)
    dst_path = os.path.join(DST_DIR, filename)

    # 画像読み込み（カラーでもグレースケールでもOK）
    img = cv2.imread(src_path, cv2.IMREAD_UNCHANGED)

    if img is None:
        print(f"Failed to read: {filename}")
        continue

    h, w = img.shape[:2]

    # 劣化なし最近傍補間で拡大（AprilTagは絶対にこれ）
    enlarged = cv2.resize(
        img,
        (w * SCALE, h * SCALE),
        interpolation=cv2.INTER_NEAREST
    )

    # 保存
    cv2.imwrite(dst_path, enlarged)
    print(f"Saved: {dst_path}")

print("✔ 完了: 全画像を拡大し、新フォルダに保存しました！")
