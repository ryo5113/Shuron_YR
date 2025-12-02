import os
import csv
import time
from datetime import datetime

import cv2
import numpy as np
import pyrealsense2 as rs
import mediapipe as mp


# ---- 1. セッション用ディレクトリと CSV の準備 ----

SESSION_TS = datetime.now().strftime("%Y%m%d_%H%M%S")
BASE_DIR = f"session_{SESSION_TS}"
DIR_LOGS = os.path.join(BASE_DIR, "logs")
DIR_IMAGES = os.path.join(BASE_DIR, "images")  # ★ 生画像保存用ディレクトリ

os.makedirs(DIR_LOGS, exist_ok=True)
os.makedirs(DIR_IMAGES, exist_ok=True)         # ★ 追加

CSV_PATH = os.path.join(DIR_LOGS, "lip_depth_points_3d.csv")

# CSVヘッダー:
# capture_id : 手動撮影の通し番号
# landmark_id: MediaPipeのランドマークID（唇のみ）
# x_px, y_px : カラー画像上の2次元座標（ピクセル）
# depth_mm   : RealSenseの深度情報（mm）
# x_cam_mm, y_cam_mm, z_cam_mm : カメラ座標系での3D座標（mm）
with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow([
        "capture_id",
        "landmark_id",
        "x_px",
        "y_px",
        "depth_mm",
        "x_cam_mm",
        "y_cam_mm",
        "z_cam_mm",
    ])

print(f"CSV will be saved to: {CSV_PATH}")


# ---- 2. 唇ランドマークIDの定義 ----
# （必要ならここを実際に使いたいIDに合わせて調整）
LIP_ID_SET = [
    # 上唇
    61, 185, 40, 39, 37, 0, 267, 269, 270, 409,
    291, 306, 308, 415, 310, 311, 312, 13, 82, 81, 80, 191, 78, 76,
    # 下唇
    146, 91, 181, 84, 17, 314, 405, 321, 375, 324, 318, 402, 317, 14, 87, 178, 88, 95
]


# ---- 3. RealSense 初期化 ----

pipeline = rs.pipeline()
config = rs.config()

config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

profile = pipeline.start(config)
align = rs.align(rs.stream.color)

depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()
print(f"Depth scale: {depth_scale} (meters per unit)")

# カラー用のカメラ内部パラメータ取得（3D変換に使用）
color_stream = profile.get_stream(rs.stream.color)
color_stream_profile = rs.video_stream_profile(color_stream)
color_intrinsics = color_stream_profile.get_intrinsics()
print("Color intrinsics:")
print(color_intrinsics)


# ---- 4. MediaPipe FaceMesh 初期化（唇だけ利用） ----

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

print('Press "c" to capture lip landmarks, "q" or ESC to quit.')


def get_depth_mm_at_pixel(depth_frame, x, y):
    """
    指定ピクセル(x, y)の深度をmmで取得。
    get_distanceの戻り値(m)をmmに変換。
    """
    distance_m = depth_frame.get_distance(x, y)
    if distance_m <= 0:
        return None
    return distance_m * 1000.0  # mm


def deproject_to_3d_mm(x_px, y_px, depth_mm):
    """
    2Dピクセル座標(x_px, y_px)と深度(mm)から、
    カメラ座標系での3D座標(X, Y, Z) [mm] を求める。
    RealSenseの deproject_pixel_to_point を使用（単位はmなのでmmに変換）。
    """
    depth_m = depth_mm / 1000.0  # mm -> m
    point_m = rs.rs2_deproject_pixel_to_point(
        color_intrinsics,
        [float(x_px), float(y_px)],
        float(depth_m),
    )
    # point_m = [X_m, Y_m, Z_m] （単位: m）
    X_mm = point_m[0] * 1000.0
    Y_mm = point_m[1] * 1000.0
    Z_mm = point_m[2] * 1000.0
    return X_mm, Y_mm, Z_mm


def main():
    capture_id = 0  # 手動撮影の通し番号

    # CSV を追記モードで開いておく
    with open(CSV_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        try:
            while True:
                frames = pipeline.wait_for_frames()
                aligned_frames = align.process(frames)

                depth_frame = aligned_frames.get_depth_frame()
                color_frame = aligned_frames.get_color_frame()
                if not depth_frame or not color_frame:
                    continue

                color_image = np.asanyarray(color_frame.get_data())
                h, w, _ = color_image.shape

                # MediaPipe用にRGB変換
                rgb_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
                rgb_image.flags.writeable = False
                results = face_mesh.process(rgb_image)
                rgb_image.flags.writeable = True

                lip_points_px = []  # マスク用に唇の2D点を溜める

                # ---- 唇ランドマーク検出＆2D点の取得 ----
                if results.multi_face_landmarks:
                    face_landmarks = results.multi_face_landmarks[0]

                    for idx in LIP_ID_SET:
                        if idx >= len(face_landmarks.landmark):
                            continue
                        lm = face_landmarks.landmark[idx]
                        x_px = int(round(lm.x * w))
                        y_px = int(round(lm.y * h))

                        if 0 <= x_px < w and 0 <= y_px < h:
                            lip_points_px.append((x_px, y_px))

                # ---- 唇以外をマスクして「唇だけ」を表示 ----
                display_image = color_image.copy()
                if len(lip_points_px) >= 3:
                    # 唇点から凸包を作成して、それをマスク領域とする
                    lip_points_np = np.array(lip_points_px, dtype=np.int32)
                    hull = cv2.convexHull(lip_points_np)

                    mask = np.zeros((h, w), dtype=np.uint8)
                    cv2.fillConvexPoly(mask, hull, 255)

                    # マスクを用いて、唇以外を黒塗りにする
                    lip_only = cv2.bitwise_and(display_image, display_image, mask=mask)

                    # 唇上にランドマーク点を描画（任意）
                    for (x_px, y_px) in lip_points_px:
                        cv2.circle(lip_only, (x_px, y_px), 2, (0, 255, 0), -1)

                    display_image = lip_only
                else:
                    # 顔未検出または唇点が少なすぎる場合は、そのままの画像を表示
                    pass

                # ガイドテキスト
                cv2.putText(
                    display_image,
                    'Press "c" to capture, "q"/ESC to quit',
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                cv2.imshow("RealSense + MediaPipe (Lips Only Masked)", display_image)

                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):  # ESC or 'q'
                    break

                # ---- 手動撮影キー 'c' ----
                if key == ord("c"):
                    if not results.multi_face_landmarks:
                        print("No face detected at capture timing.")
                        continue

                    face_landmarks = results.multi_face_landmarks[0]

                    rows = []
                    lip_points_capture = [] 

                    for idx in LIP_ID_SET:
                        if idx >= len(face_landmarks.landmark):
                            continue
                        lm = face_landmarks.landmark[idx]
                        x_px = int(round(lm.x * w))
                        y_px = int(round(lm.y * h))

                        if not (0 <= x_px < w and 0 <= y_px < h):
                            continue

                        depth_mm = get_depth_mm_at_pixel(depth_frame, x_px, y_px)
                        if depth_mm is None:
                            continue

                        # ---- 2D＋深度から3D座標（カメラ座標系, mm）へ変換 ----
                        x_cam_mm, y_cam_mm, z_cam_mm = deproject_to_3d_mm(
                            x_px, y_px, depth_mm
                        )

                        rows.append([
                            capture_id,
                            idx,
                            x_px,
                            y_px,
                            depth_mm,
                            x_cam_mm,
                            y_cam_mm,
                            z_cam_mm,
                        ])
                        lip_points_capture.append((x_px, y_px))

                    if rows:
                        # ★ ここで生画像（フルの color_image）を保存
                        img_name = f"capture_{capture_id:03d}.png"
                        img_path = os.path.join(DIR_IMAGES, img_name)
                        cv2.imwrite(img_path, color_image)
                        print(f"Saved image: {img_path}")
                        
                        # ★ マスク＋ランドマーク描画画像を作成して保存
                        if len(lip_points_capture) >= 3:
                            lip_points_np = np.array(lip_points_capture, dtype=np.int32)
                            hull = cv2.convexHull(lip_points_np)

                            mask_cap = np.zeros((h, w), dtype=np.uint8)
                            cv2.fillConvexPoly(mask_cap, hull, 255)

                            lip_only_cap = cv2.bitwise_and(color_image, color_image, mask=mask_cap)
                            for (x_px, y_px) in lip_points_capture:
                                cv2.circle(lip_only_cap, (x_px, y_px), 2, (0, 255, 0), -1)

                            mask_name = f"capture_{capture_id:03d}_mask.png"
                            mask_path = os.path.join(DIR_IMAGES, mask_name)
                            cv2.imwrite(mask_path, lip_only_cap)
                            print(f"Saved masked image: {mask_path}")
                        else:
                            print("Not enough lip points to save masked image.")

                        writer.writerows(rows)
                        f.flush()
                        print(
                            f"Captured {len(rows)} lip landmarks (with 3D) at capture_id={capture_id}"
                        )
                        capture_id += 1
                    else:
                        print("No valid lip landmarks at capture timing.")

        finally:
            pipeline.stop()
            face_mesh.close()
            cv2.destroyAllWindows()
            print("Finished.")


if __name__ == "__main__":
    main()
