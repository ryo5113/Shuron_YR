import pyrealsense2 as rs
import numpy as np
import open3d as o3d

# ========== 1. 使用するカメラのシリアル番号 ==========
SERIALS = [
    "215322071306",  # カメラ0
    "913522070157",  # カメラ1
    "108322073166",  # カメラ2
]

# キャリブレーション結果ファイル
CALIB_FILE = "multicam_calibration.npz"


def load_extrinsics_from_npz(path):
    """multicam_calibration.npz から 4x4 変換行列を読み込む"""
    data = np.load(path, allow_pickle=True)

    T_0_to_0 = data["T_0_to_0"]  # 4x4
    T_1_to_0 = data["T_1_to_0"]  # 4x4
    T_2_to_0 = data["T_2_to_0"]  # 4x4

    return [T_0_to_0, T_1_to_0, T_2_to_0]


def create_pipeline(serial):
    """指定シリアルのRealSenseパイプラインを作成・開始"""
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device(serial)
    # 解像度・FPSは必要に応じて変更
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    profile = pipeline.start(config)
    return pipeline, profile


def frames_to_pointcloud(color_frame, depth_frame, profile):
    """カラー+深度フレームから Open3D の点群に変換"""
    depth_intrinsics = depth_frame.profile.as_video_stream_profile().get_intrinsics()
    width, height = depth_intrinsics.width, depth_intrinsics.height

    depth_image = np.asanyarray(depth_frame.get_data())
    color_image = np.asanyarray(color_frame.get_data())

    # 深度スケールの取得
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale_rs = depth_sensor.get_depth_scale()

    # Open3Dの画像オブジェクトに変換
    depth_o3d = o3d.geometry.Image(depth_image.astype(np.float32))

    color_image_rgb = color_image[:, :, ::-1].copy()
    color_o3d = o3d.geometry.Image(color_image_rgb)

    # 内部パラメータ
    intr = o3d.camera.PinholeCameraIntrinsic(
        width,
        height,
        depth_intrinsics.fx,
        depth_intrinsics.fy,
        depth_intrinsics.ppx,
        depth_intrinsics.ppy,
    )

    # RGBD画像から点群生成
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color_o3d,
        depth_o3d,
        depth_scale=1.0 / depth_scale_rs,       # 既にメートルに変換済みとして扱う
        depth_trunc=1.5,                        # 距離の上限（必要に応じて変更）
        convert_rgb_to_intensity=False,
    )

    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intr)
    # Open3Dの座標系は画像と上下反転しているので補正
    pcd.transform([[1, 0, 0, 0],
                   [0, -1, 0, 0],
                   [0, 0, -1, 0],
                   [0, 0, 0, 1]])
    return pcd


def main():
    # ==== キャリブレーション結果の読み込み ====
    EXTRINSICS = load_extrinsics_from_npz(CALIB_FILE)

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
            # 深度にカラーをアライン
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
        for i in range(len(SERIALS)):
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

        # PLYとして保存
        o3d.io.write_point_cloud("face_3cams_merged.ply", merged_pcd)
        print("face_3cams_merged.ply として保存しました。")

    finally:
        # パイプライン停止
        for pipeline in pipelines:
            pipeline.stop()


if __name__ == "__main__":
    main()
