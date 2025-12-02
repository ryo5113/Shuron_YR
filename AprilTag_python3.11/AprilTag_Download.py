import os
import urllib.request

# 出力先フォルダ
SAVE_DIR = "tag36h11_png"
os.makedirs(SAVE_DIR, exist_ok=True)

# GitHub の raw 画像 URL パターン
BASE_URL = "https://raw.githubusercontent.com/AprilRobotics/apriltag-imgs/master/tag36h11/tag36_11_{:05d}.png"

# ID 0〜586（587個）
START_ID = 0
END_ID = 586

for tag_id in range(START_ID, END_ID + 1):
    url = BASE_URL.format(tag_id)
    filename = f"tag36_11_{tag_id:05d}.png"
    save_path = os.path.join(SAVE_DIR, filename)

    try:
        print(f"Downloading {filename} ...")
        urllib.request.urlretrieve(url, save_path)
    except Exception as e:
        print(f"Failed to download ID {tag_id}: {e}")

print("✔ 全 587 個の AprilTag PNG ダウンロードが完了しました！")
