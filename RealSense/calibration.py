import pyrealsense2 as rs
import cv2
import numpy as np
import os

# ==============================
# 1. ユーザーが設定すべきパラメータ
# ==============================

# 3台のRealSense D435のシリアル番号
SERIALS = [
    "215322071306",  # カメラ0（基準）
    "913522070157",  # カメラ1
    "108322073166",  # カメラ2
]

# チェッカーボードの内側コーナー数（列, 行）
# 例：9x6 の内側コーナー
pattern_size = (13, 8)

# チェッカーボード1マスの一辺の長さ [m]
# 例：25mm の場合は 0.025
square_size = 0.020

# キャプチャ画像の保存先ディレクトリ
SAVE_DIR = "calib_images"


# ==============================
# 2. RealSense パイプライン準備
# ==============================

def create_pipeline(serial):
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device(serial)
    # ★ 軽量化のため解像度を 640x480, 30fps に変更（流れはそのまま）
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    profile = pipeline.start(config)
    return pipeline, profile


# チェッカーボードの3D座標（Z=0平面上）
def create_object_points(pattern_size, square_size):
    cols, rows = pattern_size
    objp = np.zeros((cols * rows, 3), np.float32)
    # (0,0), (1,0), (2,0), ... → x方向: cols, y方向: rows
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    objp *= square_size
    return objp


def main():
    os.makedirs(SAVE_DIR, exist_ok=True)

    # 3台分のパイプライン開始
    pipelines = []
    profiles = []
    for serial in SERIALS:
        pipeline, profile = create_pipeline(serial)
        pipelines.append(pipeline)
        profiles.append(profile)

    # --- RealSense工場キャリブレーションの内部パラメータ取得 ---
    factory_camera_matrices = []   # RealSenseから取得した内部パラメータ（K）
    factory_dist_coeffs = []       # RealSenseから取得した歪み係数（coeffs）

    for i, profile in enumerate(profiles):
        color_stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
        intr = color_stream.get_intrinsics()
        # カメラ行列 K
        K = np.array([
            [intr.fx, 0,       intr.ppx],
            [0,       intr.fy, intr.ppy],
            [0,       0,       1]
        ], dtype=np.float64)
        # 歪み係数（RealSenseは coeffs に入っている）
        dist = np.array(intr.coeffs, dtype=np.float64)

        factory_camera_matrices.append(K)
        factory_dist_coeffs.append(dist)

        print(f"\n=== Camera {i} RealSense Factory Intrinsics ===")
        print("K_factory:\n", K)
        print("dist_factory (coeffs):\n", dist)
    # --- ここまで 工場キャリブ情報 ---

    # チェッカーボード3D座標
    objp = create_object_points(pattern_size, square_size)

    # 各カメラ用の対応点
    object_points = []            # すべてのビューで共通
    image_points = [[], [], []]   # カメラごとの2Dコーナー点

    print("=== キャリブレーション開始 ===")
    print("・チェッカーボードを3台すべてのカメラに写してください。")
    print("・良い姿勢のときに 'c' キーで1セット保存します。")
    print("・十分な枚数を撮ったら 'q' キーでキャリブレーションを実行します。")

    # ★ 何セット目のキャプチャか（画像ファイル名に使用）
    capture_set_index = 0

    try:
        while True:
            color_images = []

            # 3台からカラー画像取得
            for pipeline in pipelines:
                frames = pipeline.wait_for_frames()
                color_frame = frames.get_color_frame()
                if not color_frame:
                    raise RuntimeError("カラー フレーム取得に失敗しました。")
                color_image = np.asanyarray(color_frame.get_data())
                color_images.append(color_image)

            # 各カメラでチェッカーボード検出
            found_all = True
            corners_list = []
            vis_images = []

            for idx, img in enumerate(color_images):
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)

                if ret:
                    # コーナー精緻化
                    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
                                30, 0.001)
                    corners = cv2.cornerSubPix(
                        gray, corners, (11, 11), (-1, -1), criteria
                    )
                    corners_list.append(corners)
                    vis = cv2.drawChessboardCorners(img.copy(), pattern_size, corners, ret)
                else:
                    found_all = False
                    corners_list.append(None)
                    vis = img.copy()

                vis_images.append(vis)

            # 軽量化のためプレビュー用に少し縮小して結合（元画像は別途保存するのでOK）
            small_images = [cv2.resize(v, (320, 240)) for v in vis_images]
            disp = np.hstack(small_images)

            cv2.putText(disp, f"Captured sets: {len(object_points)}",
                        (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            if not found_all:
                cv2.putText(disp, "Checkerboard NOT found in ALL cameras",
                            (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            else:
                cv2.putText(disp, "Checkerboard found in ALL cameras",
                            (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            cv2.imshow("Multi-Cam Checkerboard (Cam0 | Cam1 | Cam2)", disp)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('c'):
                # 3台すべてで検出できている場合のみ保存
                if found_all:
                    object_points.append(objp)
                    for i in range(3):
                        image_points[i].append(corners_list[i])

                        # ★ 歪み補正前（元画像）を保存
                        raw_name = os.path.join(
                            SAVE_DIR,
                            f"set{capture_set_index:02d}_cam{i}_raw.png"
                        )
                        cv2.imwrite(raw_name, color_images[i])

                    print(f"セット {len(object_points)} を保存しました。（set index = {capture_set_index}）")
                    capture_set_index += 1
                else:
                    print("3台すべてでチェッカーボードが検出されていません。保存しません。")

            elif key == ord('q'):
                print("キャプチャ終了。キャリブレーションを実行します。")
                break

        cv2.destroyAllWindows()

        if len(object_points) < 3:
            print("十分なセットがありません（少なくとも3セット以上が望ましいです）。")
            return

        # 画像サイズ（最後に取得した画像のサイズを使用）
        img_h, img_w = color_images[0].shape[:2]
        image_size = (img_w, img_h)

        # ==============================
        # 3. 各カメラの内部パラメータ推定（OpenCV）
        # ==============================
        camera_matrices = []
        dist_coeffs = []

        for i in range(3):
            ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
                object_points,
                image_points[i],
                image_size,
                None,
                None
            )
            camera_matrices.append(mtx)
            dist_coeffs.append(dist)
            print(f"\n=== Camera {i} (OpenCV estimated intrinsics) ===")
            print("RMS reprojection error:", ret)
            print("Camera matrix (mtx):\n", mtx)
            print("Distortion coeffs (dist):\n", dist)

        # ==============================
        # 4. カメラ0基準での外部パラメータ（R, T）
        # ==============================
        flags = cv2.CALIB_FIX_INTRINSIC
        criteria_stereo = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
                           100, 1e-5)

        # カメラ1 → カメラ0
        ret10, _, _, _, _, R_1_0, T_1_0, _, _ = cv2.stereoCalibrate(
            object_points,
            image_points[1],
            image_points[0],
            camera_matrices[1],
            dist_coeffs[1],
            camera_matrices[0],
            dist_coeffs[0],
            image_size,
            criteria=criteria_stereo,
            flags=flags
        )

        # カメラ2 → カメラ0
        ret20, _, _, _, _, R_2_0, T_2_0, _, _ = cv2.stereoCalibrate(
            object_points,
            image_points[2],
            image_points[0],
            camera_matrices[2],
            dist_coeffs[2],
            camera_matrices[0],
            dist_coeffs[0],
            image_size,
            criteria=criteria_stereo,
            flags=flags
        )

        print("\n=== Camera 1 -> Camera 0 (stereoCalibrate) ===")
        print("Stereo RMS error:", ret10)
        print("R_1_0:\n", R_1_0)
        print("T_1_0:\n", T_1_0)

        print("\n=== Camera 2 -> Camera 0 (stereoCalibrate) ===")
        print("Stereo RMS error:", ret20)
        print("R_2_0:\n", R_2_0)
        print("T_2_0:\n", T_2_0)

        # 4x4 変換行列に拡張
        T_0_0 = np.eye(4, dtype=np.float64)

        T_1_0_mat = np.eye(4, dtype=np.float64)
        T_1_0_mat[:3, :3] = R_1_0
        T_1_0_mat[:3, 3] = T_1_0.ravel()

        T_2_0_mat = np.eye(4, dtype=np.float64)
        T_2_0_mat[:3, :3] = R_2_0
        T_2_0_mat[:3, 3] = T_2_0.ravel()

        print("\n=== 4x4 Transform Matrices (Camera i -> Camera 0) ===")
        print("T_0_to_0:\n", T_0_0)
        print("T_1_to_0:\n", T_1_0_mat)
        print("T_2_to_0:\n", T_2_0_mat)

        # ==============================
        # 5. 歪み補正前後の画像を保存
        # ==============================
        num_sets = len(object_points)
        print(f"\n=== 歪み補正画像の保存を開始します（セット数: {num_sets}） ===")

        for set_idx in range(num_sets):
            for cam_idx in range(3):
                raw_name = os.path.join(
                    SAVE_DIR,
                    f"set{set_idx:02d}_cam{cam_idx}_raw.png"
                )
                if not os.path.exists(raw_name):
                    continue  # 念のため存在チェック

                img_raw = cv2.imread(raw_name)
                if img_raw is None:
                    continue

                K = camera_matrices[cam_idx]
                dist = dist_coeffs[cam_idx]

                # 歪み補正
                img_undist = cv2.undistort(img_raw, K, dist)

                undist_name = os.path.join(
                    SAVE_DIR,
                    f"set{set_idx:02d}_cam{cam_idx}_undist.png"
                )
                cv2.imwrite(undist_name, img_undist)

        print("歪み補正前後の画像を保存しました。")

        # ==============================
        # 6. 必要な情報を npz に保存
        # ==============================
        np.savez(
            "multicam_calibration.npz",
            # OpenCVで推定した内部パラメータ・歪み
            camera_matrices=camera_matrices,
            dist_coeffs=dist_coeffs,
            # RealSense工場キャリブレーションの内部パラメータ・歪み
            factory_camera_matrices=factory_camera_matrices,
            factory_dist_coeffs=factory_dist_coeffs,
            # 外部パラメータ（変換行列）
            T_0_to_0=T_0_0,
            T_1_to_0=T_1_0_mat,
            T_2_to_0=T_2_0_mat,
            # その他情報
            image_size=image_size,
            pattern_size=pattern_size,
            square_size=square_size
        )
        print("\nキャリブレーション結果を multicam_calibration.npz に保存しました。")

    finally:
        for pipeline in pipelines:
            pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
