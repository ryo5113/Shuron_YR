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

# 各カメラで取得するフレーム数（最後のフレームを使用）
NUM_FRAMES = 100

# ========== 2. 外部パラメータ（幾何学的に設定：ICPの初期値として使用） ==========
# カメラ0座標系を基準とし、
# カメラ1,2 -> カメラ0 への 4x4 変換行列 T_i_to_0 を手動で設定します。
# 実際の配置に合わせて値を調整してください。

def make_extrinsic(tx, ty, tz, angle_deg):
    """カメラ0座標系を基準とした外部パラメータ行列を作成"""
    T = np.eye(4, dtype=np.float64)
    angle = np.deg2rad(angle_deg)

    # y軸周りの回転（ピッチ）
    R = np.array(
        [
            [ np.cos(angle), 0.0, np.sin(angle)],
            [ 0.0,           1.0, 0.0          ],
            [-np.sin(angle), 0.0, np.cos(angle)],
        ],
        dtype=np.float64,
    )

    T[:3, :3] = R
    T[0, 3] = tx
    T[1, 3] = ty
    T[2, 3] = tz
    return T

# カメラ0は原点・回転なし
T_0_to_0 = np.eye(4, dtype=np.float64)

# カメラ1設定（左側に配置、内向き45度)
T_1_to_0 = make_extrinsic(-0.26, 0.0, 0.15, 45.0)

# カメラ2設定（右側に配置、内向き-45度）
T_2_to_0 = make_extrinsic(0.23, 0.0, 0.15, -45.0)

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

# ========== 4. フレーム -> Open3D点群 変換 ==========
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
    # RealSense: 画像座標系
    # Open3D標準の右手系に合わせる
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

# ========== 5. ICPで source_pcd をカメラ0座標系へ合わせる ==========
def icp_to_cam0(source_pcd, target_pcd, init_trans, voxel_size=0.005):
    """
    source_pcd を target_pcd（ここではカメラ0）に
    ICPで合わせる変換行列を求める。
    init_trans に EXTRINSICS[i] を渡して初期値とする。
    """
    # ダウンサンプリング
    # source_down = source_pcd.voxel_down_sample(voxel_size)
    # target_down = target_pcd.voxel_down_sample(voxel_size)

    radius = voxel_size * 2.0
    # source_down.estimate_normals(
    #     o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=30))
    # target_down.estimate_normals(
    #     o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=30))

    # source_pcd.paint_uniform_color([1, 0, 0])
    # target_pcd.paint_uniform_color([0, 1, 0])
    
    # ダウンサンプリングなしの場合
    source_pcd.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=30))
    target_pcd.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=30))

    # 距離しきい値
    max_correspondence_distance_coarse = voxel_size * 10.0
    max_correspondence_distance_fine = voxel_size * 1.0

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
    # --- RealSenseパイプラインを3台分作成 ---
    pipelines = []
    profiles = []
    try:
        for serial in SERIALS:
            pipeline, profile = create_pipeline(serial)
            pipelines.append(pipeline)
            profiles.append(profile)

        # ===== 複数フレーム取得（最後のフレームを使用） =====
        color_frames = []
        depth_frames = []

        for pipeline in pipelines:
            align = rs.align(rs.stream.color)
            depth = None
            color = None
            for _ in range(NUM_FRAMES):
                frames = pipeline.wait_for_frames()
                # 深度をカラーにアライン
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

        # ===== 変換を適用して1つの点群にマージ =====
        pcd0_aligned = base_pcd
        pcd1_aligned = pcds[1].transform(T_1_to_0_icp)
        pcd2_aligned = pcds[2].transform(T_2_to_0_icp)

        merged_pcd = o3d.geometry.PointCloud()
        merged_pcd += pcd0_aligned
        merged_pcd += pcd1_aligned
        merged_pcd += pcd2_aligned

        # ===== Open3Dビューワで3D表示 =====
        o3d.visualization.draw_geometries([merged_pcd])

        # ===== Matplotlibで XY, XZ, ZY 平面に投影して確認 =====
        points = np.asarray(merged_pcd.points)
        colors = np.asarray(merged_pcd.colors)

        fig, axes = plt.subplots(3, 1, figsize=(6, 12))

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
        filename = f"PLY/face_3cams_geom_merged_{timestamp}.ply"
        o3d.io.write_point_cloud(filename, merged_pcd)
        print(f"{filename} として保存しました。")
    
    finally:
        for pipeline in pipelines:
            pipeline.stop()

if __name__ == "__main__":
    main()
