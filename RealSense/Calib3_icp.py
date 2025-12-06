import pyrealsense2 as rs
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # プロジェクション指定用
from datetime import datetime

# ========== 1. 使用するカメラのシリアル番号（カメラ0, カメラ1, カメラ2） ==========
SERIALS = [
    "047322070108",  # カメラ0（基準）
    "913522070157",  # カメラ1
    "108322073166",  # カメラ2
]

# ========== 2. 外部パラメータ（幾何学的に設定：ICPの初期値として使用） ==========
# カメラ0座標系を基準とし、
# カメラ1,2 -> カメラ0 への 4x4 変換行列 T_i_to_0 を手動で設定します。
# 実際の配置に合わせて baseline_x_01, baseline_x_02 を修正してください。

baseline_x_01 = 0.3   # カメラ0 → カメラ1 のX方向距離 [m]
baseline_x_02 = -0.3  # カメラ0 → カメラ2 のX方向距離 [m]

T_0_to_0 = np.eye(4, dtype=np.float64)

T_1_to_0 = np.eye(4, dtype=np.float64)
T_1_to_0[0, 3] = baseline_x_01

T_2_to_0 = np.eye(4, dtype=np.float64)
T_2_to_0[0, 3] = baseline_x_02

EXTRINSICS = [T_0_to_0, T_1_to_0, T_2_to_0]

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
        depth_trunc=1.0,              # 1 [m]までを有効とする（必要に応じて変更）
        convert_rgb_to_intensity=False,
    )

    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intr)

    # Open3Dの座標系補正（画像座標 → 通常の3D座標）
    pcd.transform([[1, 0, 0, 0],
                   [0, -1, 0, 0],
                   [0, 0, 1, 0],
                   [0, 0, 0, 1]])
    return pcd

# ========== 5. ICP（カメラi → カメラ0）の推定 ==========
def icp_to_cam0(source_pcd, target_pcd, init_trans, voxel_size=0.005):
    """
    source_pcd を target_pcd（ここではカメラ0）に
    ICPで合わせる変換行列を求める。
    init_trans に EXTRINSICS[i] を渡して初期値とする。
    """
    # ダウンサンプリング
    #source_down = source_pcd.voxel_down_sample(voxel_size)
    #target_down = target_pcd.voxel_down_sample(voxel_size)

    # 法線推定（Point-to-plane ICP 用）
    radius = voxel_size * 2.0
    # source_down.estimate_normals(
    #     o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=30))
    # target_down.estimate_normals(
    #     o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=30))

    #source_pcd.paint_uniform_color([1, 0, 0])
    #target_pcd.paint_uniform_color([0, 1, 0])
    
    # ダウンサンプリングなしの場合
    source_pcd.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=30))
    target_pcd.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=30))

    # 距離しきい値
    max_correspondence_distance_coarse = voxel_size * 10.0
    max_correspondence_distance_fine = voxel_size * 1.0

    # # 粗いICP
    # icp_coarse = o3d.pipelines.registration.registration_icp(
    #     source_down, target_down,
    #     max_correspondence_distance_coarse, init_trans,
    #     o3d.pipelines.registration.TransformationEstimationPointToPlane()
    # )

    # # 細かいICP
    # icp_fine = o3d.pipelines.registration.registration_icp(
    #     source_down, target_down,
    #     max_correspondence_distance_fine, icp_coarse.transformation,
    #     o3d.pipelines.registration.TransformationEstimationPointToPlane()
    # )

    # 粗いICP(ダウンサンプリングなし)
    icp_coarse = o3d.pipelines.registration.registration_icp(
        source_pcd, target_pcd,
        max_correspondence_distance_coarse, init_trans,
        o3d.pipelines.registration.TransformationEstimationPointToPlane()
    )

    # 細かいICP
    icp_fine = o3d.pipelines.registration.registration_icp(
        source_pcd, target_pcd,
        max_correspondence_distance_fine, icp_coarse.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPlane()
    )

    print("ICP fitness:", icp_fine.fitness, "rmse:", icp_fine.inlier_rmse)
    return icp_fine.transformation

# ========== 6. メイン処理 ==========
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

        # ===== 各カメラから点群生成（まだ変換しない：各カメラ座標系のまま） =====
        pcds = []
        for i in range(len(SERIALS)):
            pcd = frames_to_pointcloud(color_frames[i], depth_frames[i], profiles[i])
            pcds.append(pcd)

        # ===== カメラ0を基準に、1・2をICPで合わせる =====
        base_pcd = pcds[0]

        # カメラ1 → カメラ0
        T_1_to_0_icp = icp_to_cam0(pcds[1], base_pcd, T_1_to_0)
        # カメラ2 → カメラ0
        T_2_to_0_icp = icp_to_cam0(pcds[2], base_pcd, T_2_to_0)

        # ===== 変換を適用してマージ =====
        pcd0_aligned = base_pcd
        pcd1_aligned = pcds[1].transform(T_1_to_0_icp)
        pcd2_aligned = pcds[2].transform(T_2_to_0_icp)

        merged_pcd = o3d.geometry.PointCloud()
        merged_pcd += pcd0_aligned
        merged_pcd += pcd1_aligned
        merged_pcd += pcd2_aligned

        # 必要に応じてダウンサンプリング
        #merged_pcd = merged_pcd.voxel_down_sample(voxel_size=0.003)

        # ===== Open3Dで表示 =====
        o3d.visualization.draw_geometries([merged_pcd])

        # ===== Matplotlib で XY / XZ / ZY 平面のプロット =====
        points = np.asarray(merged_pcd.points)
        colors = np.asarray(merged_pcd.colors)

        fig, axes = plt.subplots(3, 1, figsize=(6, 12))

        plt.rcParams["font.size"] = 20

        # 1段目: XY 平面
        axes[0].scatter(-points[:, 0], points[:, 1], c=colors, s=0.5)
        axes[0].set_xlabel('X [m]', fontsize=20)
        axes[0].set_ylabel('Y [m]', fontsize=20)
        axes[0].set_title('mouth shape(XY)')
        axes[0].tick_params(axis='both', labelsize=20)
        axes[0].grid(alpha=0.2)

        # 2段目: XZ 平面
        axes[1].scatter(-points[:, 0], points[:, 2], c=colors, s=0.5)
        axes[1].set_xlabel('X [m]', fontsize=20)
        axes[1].set_ylabel('Z [m]', fontsize=20)
        axes[1].set_title('mouth shape(XZ)')
        axes[1].tick_params(axis='both', labelsize=20)
        axes[1].grid(alpha=0.2)

        # 3段目: ZY 平面
        axes[2].scatter(points[:, 2], points[:, 1], c=colors, s=0.5)
        axes[2].set_xlabel('Z [m]', fontsize=20)
        axes[2].set_ylabel('Y [m]', fontsize=20)
        axes[2].set_title('mouth shape(ZY)')
        axes[2].tick_params(axis='both', labelsize=20)
        axes[2].grid(alpha=0.2)

        plt.tight_layout()
        plt.show()

        # PLYとして保存
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"RealSense/PLY/face_3cams_geom_merged_{timestamp}.ply"
        o3d.io.write_point_cloud(filename, merged_pcd)
        print(f"{filename} として保存しました。")
    
    finally:
        for pipeline in pipelines:
            pipeline.stop()

if __name__ == "__main__":
    main()
