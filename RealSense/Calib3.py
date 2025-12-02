import pyrealsense2 as rs
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # プロジェクション指定用

# ========== 1. 使用するカメラのシリアル番号（カメラ0, カメラ1） ==========
SERIALS = [
    "215322071306",  # カメラ0
    "913522070157",  # カメラ1
    "108322073166",  # カメラ2
]

# ========== 2. 外部パラメータ（幾何学的に設定） ==========
# カメラ0座標系を基準とし、
# カメラ1 -> カメラ0 への 4x4 変換行列 T_1_to_0 を手動で設定します。
#
# 例として：
#   - カメラ0,1 はほぼ平行に並べて置き、
#   - カメラ1はカメラ0から +X 方向に baseline_x [m] だけ離れている
# という単純なモデルにしています。
#
# 実際の配置に合わせて baseline_x を修正してください。
# （符号は、どちらを +X とみなすかで変わります）

baseline_x_01 = 0.3  # カメラ間距離 [m]（例：0.15 m）
baseline_x_02 = -0.3 # カメラ間距離 [m]（例：0.15 m）

# カメラ0 -> カメラ0
T_0_to_0 = np.eye(4, dtype=np.float64)

# カメラ1 -> カメラ0
T_1_to_0 = np.eye(4, dtype=np.float64)
T_1_to_0[0, 3] = baseline_x_01  # X方向に平行移動（必要に応じて符号を変えてください）

# カメラ2 -> カメラ0
T_2_to_0 = np.eye(4, dtype=np.float64)
T_2_to_0[0, 3] = baseline_x_02  # X方向に平行移動（必要に応じて符号を変えてください）

EXTRINSICS = [T_0_to_0, T_1_to_0, T_2_to_0] # 外部パラメータリスト


# ========== 3. RealSense パイプライン ==========
def create_pipeline(serial):
    """指定シリアルのRealSenseパイプラインを作成・開始"""
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device(serial)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    profile = pipeline.start(config)
    
    return pipeline, profile

# ========== 4. カラー+深度 → Open3D点群 ==========
def frames_to_pointcloud(color_frame, depth_frame, profile):
    """カラー+深度フレームから Open3D の点群に変換"""
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
        depth_trunc=1.5,              # 1.5 [m]までを有効とする（必要に応じて変更）
        convert_rgb_to_intensity=False,
    )

    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intr)

    # Open3Dの座標系補正（画像座標 → 通常の3D座標）
    pcd.transform([[1, 0, 0, 0],
                   [0, -1, 0, 0],
                   [0, 0, 1, 0],
                   [0, 0, 0, 1]])
    return pcd


# ========== 5. メイン処理 ==========
def main():
    # 3台分のパイプラインとプロファイルを作成
    pipelines = []
    profiles = []
    for serial in SERIALS:
        pipeline, profile = create_pipeline(serial)
        pipelines.append(pipeline)
        profiles.append(profile)

    try:
        # ===== 1フレームだけ取得 =====
        color_frames = []
        depth_frames = []

        for pipeline in pipelines:
            frames = pipeline.wait_for_frames()
            # 深度をカラーにアライン
            align = rs.align(rs.stream.color)
            aligned = align.process(frames)
            depth = aligned.get_depth_frame()
            color = aligned.get_color_frame()
            if not depth or not color:
                raise RuntimeError("フレーム取得に失敗しました")
            depth_frames.append(depth)
            color_frames.append(color)

        # ===== 各カメラから点群生成 =====
        pcds = []
        for i in range(len(SERIALS)):  # 0,1
            pcd = frames_to_pointcloud(color_frames[i], depth_frames[i], profiles[i])
            # カメラi座標系 → カメラ0座標系への変換を適用
            T_i_to_0 = EXTRINSICS[i]
            pcd.transform(T_i_to_0)
            pcds.append(pcd)

        # ===== 点群の結合 =====
        merged_pcd = o3d.geometry.PointCloud()
        for p in pcds:
            merged_pcd += p

        # 必要に応じてダウンサンプリング
        merged_pcd = merged_pcd.voxel_down_sample(voxel_size=0.003)

        # ===== 表示 =====
        o3d.visualization.draw_geometries([merged_pcd])

        points = np.asarray(merged_pcd.points)
        colors = np.asarray(merged_pcd.colors)

        fig, axes = plt.subplots(3, 1, figsize=(6, 12))

        # 1段目: XY 平面
        axes[0].scatter(points[:, 0], points[:, 1], c=colors, s=0.5)
        axes[0].set_xlabel('X [m]')
        axes[0].set_ylabel('Y [m]')
        axes[0].set_title('XY plane')
        axes[0].grid(alpha=0.2)

        # 2段目: XZ 平面
        axes[1].scatter(points[:, 0], points[:, 2], c=colors, s=0.5)
        axes[1].set_xlabel('X [m]')
        axes[1].set_ylabel('Z [m]')
        axes[1].set_title('XZ plane')
        axes[1].grid(alpha=0.2)

        # 3段目: YZ 平面
        axes[2].scatter(points[:, 2], points[:, 1], c=colors, s=0.5)
        axes[2].set_xlabel('Z [m]')
        axes[2].set_ylabel('Y [m]')
        axes[2].set_title('ZY plane')
        axes[2].grid(alpha=0.2)

        plt.tight_layout()
        plt.show()

        # PLYとして保存
        o3d.io.write_point_cloud("face_3cams_geom_merged.ply", merged_pcd)
        print("face_3cams_geom_merged.ply として保存しました。")

    finally:
        for pipeline in pipelines:
            pipeline.stop()


if __name__ == "__main__":
    main()
