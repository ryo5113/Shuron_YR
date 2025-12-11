import pyrealsense2 as rs
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from datetime import datetime

# ========== 1. 使用するカメラのシリアル番号 ==========
# 確認したいカメラのシリアル番号をここに設定してください
# 例：カメラ0（基準） "047322070108"
#     カメラ1         "913522070157"
#     カメラ2         "108322073166"
SERIAL = "108322073166"

# 取得するフレーム数（最後のフレームを使用）
NUM_FRAMES = 10


# ========== 2. RealSense パイプライン作成 ==========
def create_pipeline(serial):
    """指定シリアルのRealSenseパイプラインを作成・開始"""
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device(serial)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    profile = pipeline.start(config)
    return pipeline, profile


# ========== 3. フレーム -> Open3D点群 変換 ==========
def frames_to_pointcloud(color_frame, depth_frame, profile):
    """
    RealSenseのカラー/深度フレームから
    Open3Dの PointCloud を生成する
    """
    depth_intrinsics = depth_frame.profile.as_video_stream_profile().get_intrinsics()
    width, height = depth_intrinsics.width, depth_intrinsics.height

    depth_image = np.asanyarray(depth_frame.get_data())
    color_image = np.asanyarray(color_frame.get_data())

    # 深度スケール（1カウントが何mか）
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale_rs = depth_sensor.get_depth_scale()

    # Open3D用画像
    depth_o3d = o3d.geometry.Image(depth_image.astype(np.float32))

    # BGR → RGB（copy()してC連続配列にする）
    color_image_rgb = color_image[:, :, ::-1].copy()
    color_o3d = o3d.geometry.Image(color_image_rgb)

    # 内部パラメータ（RealSenseの intrinsics をそのまま使用）
    intr = o3d.camera.PinholeCameraIntrinsic(
        width,
        height,
        depth_intrinsics.fx,
        depth_intrinsics.fy,
        depth_intrinsics.ppx,
        depth_intrinsics.ppy,
    )

    # RGBD画像から点群生成
    # Open3Dは z = depth_pixel / depth_scale と扱うので、
    # RealSenseの z = raw * depth_scale_rs と一致させるために
    # depth_scale = 1.0 / depth_scale_rs を渡す
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color_o3d,
        depth_o3d,
        depth_scale=1.0 / depth_scale_rs,
        depth_trunc=1.0,  # 1.0mまで
        convert_rgb_to_intensity=False,
    )

    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intr)

    # 座標系補正（Y反転）
    T_flip = np.array(
        [
            [1,  0, 0, 0],
            [0, -1, 0, 0],
            [0,  0, 1, 0],
            [0,  0, 0, 1],
        ],
        dtype=np.float64,
    )
    pcd.transform(T_flip)

    return pcd


# ========== 4. メイン処理（1台用） ==========
def main():
    pipeline = None
    try:
        # パイプライン開始
        pipeline, profile = create_pipeline(SERIAL)

        # フレーム取得（最後のフレームを使用）
        align = rs.align(rs.stream.color)
        depth = None
        color = None
        for _ in range(NUM_FRAMES):
            frames = pipeline.wait_for_frames()
            aligned = align.process(frames)
            depth = aligned.get_depth_frame()
            color = aligned.get_color_frame()
            if not depth or not color:
                raise RuntimeError("フレーム取得に失敗しました")

        # 点群生成（このカメラ座標系のまま）
        pcd = frames_to_pointcloud(color, depth, profile)

        # Open3Dで点群表示
        o3d.visualization.draw_geometries([pcd])

        # XY, XZ, ZY 平面に投影して確認
        points = np.asarray(pcd.points)
        colors = np.asarray(pcd.colors)

        fig, axes = plt.subplots(3, 1, figsize=(6, 12))

        # 1段目: XY 平面
        axes[0].scatter(-points[:, 0], points[:, 1], c=colors, s=0.5)
        axes[0].set_xlabel('X [m]', fontsize=20)
        axes[0].set_ylabel('Y [m]', fontsize=20)
        axes[0].set_title('point cloud (XY)', fontsize=20)
        axes[0].tick_params(axis='both', labelsize=16)
        axes[0].grid(alpha=0.2)

        # 2段目: XZ 平面
        axes[1].scatter(-points[:, 0], points[:, 2], c=colors, s=0.5)
        axes[1].set_xlabel('X [m]', fontsize=20)
        axes[1].set_ylabel('Z [m]', fontsize=20)
        axes[1].set_title('point cloud (XZ)', fontsize=20)
        axes[1].tick_params(axis='both', labelsize=16)
        axes[1].grid(alpha=0.2)

        # 3段目: ZY 平面
        axes[2].scatter(points[:, 2], points[:, 1], c=colors, s=0.5)
        axes[2].set_xlabel('Z [m]', fontsize=20)
        axes[2].set_ylabel('Y [m]', fontsize=20)
        axes[2].set_title('point cloud (ZY)', fontsize=20)
        axes[2].tick_params(axis='both', labelsize=16)
        axes[2].grid(alpha=0.2)

        plt.tight_layout()
        plt.show()

        # PLYとして保存（1台用のファイル名）
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"PLY/face_1cam_{SERIAL}_{timestamp}.ply"
        o3d.io.write_point_cloud(filename, pcd)
        print(f"{filename} として保存しました。")

    finally:
        if pipeline is not None:
            pipeline.stop()


if __name__ == "__main__":
    main()
