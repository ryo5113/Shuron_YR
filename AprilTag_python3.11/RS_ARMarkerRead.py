import pyrealsense2 as rs
import cv2
import numpy as np
import math
from pupil_apriltags import Detector

# =========================================================
# 1. ユーザー設定
# =========================================================

# 使用するRealSenseのシリアル番号
REALSense_SERIAL = "215322071306"  # 例：カメラ0（基準）用に変更

TAG_SIZE_M = 0.105  # タグのサイズ(m)

# カラー画像の解像度・FPS
COLOR_WIDTH = 640
COLOR_HEIGHT = 480
COLOR_FPS = 30

# =========================================================
# 2. 回転行列 → オイラー角 (roll, pitch, yaw) への変換
# =========================================================
def rotation_matrix_to_euler(R):
    """
    R: 3x3 の回転行列 (numpy.ndarray)
    return: roll(x), pitch(y), yaw(z) [rad]
    ZYX順 (R = Rz(yaw) * Ry(pitch) * Rx(roll)) を想定
    """
    sy = math.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    singular = sy < 1e-6

    if not singular:
        yaw   = math.atan2(R[1, 0], R[0, 0])   # Z
        pitch = math.atan2(-R[2, 0], sy)       # Y
        roll  = math.atan2(R[2, 1], R[2, 2])   # X
    else:
        yaw   = math.atan2(-R[0, 1], R[1, 1])
        pitch = math.atan2(-R[2, 0], sy)
        roll  = 0.0

    return roll, pitch, yaw

# =========================================================
# 3. カメラ内部パラメータを RealSense から取得
# =========================================================
def get_intrinsics_from_realsense(serial):
    """
    RealSense のカラー用 intrinsics を SDK から取得し、
    fx, fy, cx, cy を返す。
    """
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device(serial)
    config.enable_stream(rs.stream.color, COLOR_WIDTH, COLOR_HEIGHT,
                         rs.format.bgr8, COLOR_FPS)

    profile = pipeline.start(config)

    # アクティブプロファイルからカラーの intrinsics を取得
    color_stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
    intr = color_stream.get_intrinsics()

    fx = float(intr.fx)
    fy = float(intr.fy)
    cx = float(intr.ppx)
    cy = float(intr.ppy)

    print("=== RealSense factory intrinsics ===")
    print(f"fx = {fx}")
    print(f"fy = {fy}")
    print(f"cx = {cx}")
    print(f"cy = {cy}")

    # この関数内で一度パイプラインを止める
    pipeline.stop()

    return fx, fy, cx, cy

# =========================================================
# 4. AprilTag Detector の準備
# =========================================================
def create_detector():
    detector = Detector(
        families="tag36h11",
        nthreads=4,
        quad_decimate=1.0,
        quad_sigma=0.0,
        refine_edges=True,
        decode_sharpening=0.25,
        debug=False
    )
    return detector

# =========================================================
# 5. RealSense パイプラインの準備
# =========================================================
def create_realsense_pipeline(serial):
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device(serial)
    config.enable_stream(rs.stream.color, COLOR_WIDTH, COLOR_HEIGHT,
                         rs.format.bgr8, COLOR_FPS)
    pipeline.start(config)
    return pipeline

def debug_pitch_20deg_step(pitch):
    """
    ピッチ角を 20度刻み（誤差±1度）でチェックして、
    条件を満たすときにデバッグ表示を行う。

    pitch_rad : ピッチ角 [rad]
    """
    pitch_deg = math.degrees(pitch)

    # 一番近い 20度刻みの角度を算出
    nearest_20 = round(pitch_deg / 20.0) * 20.0

    # 誤差判定（±1度以内）
    if abs(pitch_deg - nearest_20) <= 1.0:
        print(f"[DEBUG] pitch ≈ {nearest_20:.0f} deg (actual {pitch_deg:.1f} deg)")
        # 将来カメラ撮影などに使うことを想定しているので、
        # 必要ならここから別処理を呼び出せるようにしておく
        return True, pitch_deg, nearest_20

    return False, pitch_deg, nearest_20

# =========================================================
# 6. メインループ
# =========================================================
def main():
    # RealSense SDK から内部パラメータ取得
    fx, fy, cx, cy = get_intrinsics_from_realsense(REALSense_SERIAL)
    camera_params = (fx, fy, cx, cy)

    # AprilTag Detector
    detector = create_detector()

    # RealSense パイプライン開始
    pipeline = create_realsense_pipeline(REALSense_SERIAL)

    try:
        while True:
            # フレーム取得
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue

            # numpy 配列へ変換
            color_image = np.asanyarray(color_frame.get_data())

            # グレースケール
            gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)

            # AprilTag 検出 + 姿勢推定
            results = detector.detect(
                gray,
                estimate_tag_pose=True,
                camera_params=camera_params,
                tag_size=TAG_SIZE_M
            )

            frame_vis = color_image.copy()

            for r in results:
                # 輪郭線描画
                corners = r.corners.astype(int)
                for i in range(4):
                    pt1 = tuple(corners[i])
                    pt2 = tuple(corners[(i + 1) % 4])
                    cv2.line(frame_vis, pt1, pt2, (0, 255, 0), 2)

                # 中心点
                cX, cY = int(r.center[0]), int(r.center[1])
                cv2.circle(frame_vis, (cX, cY), 5, (0, 0, 255), -1)

                # タグID
                tag_id = r.tag_id
                cv2.putText(frame_vis, f"ID:{tag_id}", (cX - 20, cY - 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

                # 姿勢（位置＋回転）
                t = r.pose_t.flatten()  # [tx, ty, tz]
                R = r.pose_R
                roll, pitch, yaw = rotation_matrix_to_euler(R)

                matched, pitch_deg, nearest_20 = debug_pitch_20deg_step(pitch)

                # コンソール出力
                print(f"ID:{tag_id}")
                print(f"  t = [{t[0]:.3f}, {t[1]:.3f}, {t[2]:.3f}]  [m]")
                print(f"  roll={math.degrees(roll):.1f}deg, "
                      f"pitch={math.degrees(pitch):.1f}deg, "
                      f"yaw={math.degrees(yaw):.1f}deg")

                # 画面表示用テキスト
                base_y = cY + 20
                dy = 18
                cv2.putText(frame_vis, f"X:{t[0]:.3f}m", (cX - 80, base_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                cv2.putText(frame_vis, f"Y:{t[1]:.3f}m", (cX - 80, base_y + dy),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                cv2.putText(frame_vis, f"Z:{t[2]:.3f}m", (cX - 80, base_y + 2 * dy),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

                cv2.putText(frame_vis, f"R:{math.degrees(roll):.1f}",
                            (cX + 40, base_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                cv2.putText(frame_vis, f"P:{math.degrees(pitch):.1f}",
                            (cX + 40, base_y + dy),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                cv2.putText(frame_vis, f"Y:{math.degrees(yaw):.1f}",
                            (cX + 40, base_y + 2 * dy),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                
                if matched:
                    cv2.putText(frame_vis, f"PITCH HIT: {nearest_20:.0f} deg",
                                (30, 40), cv2.FONT_HERSHEY_SIMPLEX,
                                1.0, (0, 0, 255), 2)

            cv2.imshow("RealSense AprilTag Pose", frame_vis)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
