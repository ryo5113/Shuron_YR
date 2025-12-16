import pyrealsense2 as rs
import numpy as np
import open3d as o3d
from datetime import datetime
import cv2
import math
from pupil_apriltags import Detector
import os
from concurrent.futures import ThreadPoolExecutor

# =========================================================
# 3Cam_rot.py 側（既存設定）
# =========================================================
SERIALS = [
    "047322070108",  # カメラ0（基準） :contentReference[oaicite:6]{index=6}
    "913522070157",  # カメラ1
    "108322073166",  # カメラ2
]
NUM_FRAMES = 50  # 最後のフレームを使用 :contentReference[oaicite:7]{index=7}

def make_extrinsic(tx, ty, tz, angle_deg):
    T = np.eye(4, dtype=np.float64)
    angle = np.deg2rad(angle_deg)
    # y軸周りの回転（ピッチ） :contentReference[oaicite:8]{index=8}
    R = np.array([
        [ np.cos(angle), 0.0, np.sin(angle)],
        [ 0.0,           1.0, 0.0          ],
        [-np.sin(angle), 0.0, np.cos(angle)],
    ], dtype=np.float64)
    T[:3, :3] = R
    T[0, 3] = tx
    T[1, 3] = ty
    T[2, 3] = tz
    return T

T_0_to_0 = np.eye(4, dtype=np.float64)
T_1_to_0 = make_extrinsic(-0.25, 0.0, 0.15, 45.0)   # :contentReference[oaicite:9]{index=9}
T_2_to_0 = make_extrinsic( 0.24, 0.0, 0.15, -45.0)  # :contentReference[oaicite:10]{index=10}

def create_pipeline(serial):
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device(serial)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    profile = pipeline.start(config)  # :contentReference[oaicite:11]{index=11}
    return pipeline, profile

def frames_to_pointcloud(color_frame, depth_frame, profile):
    depth_intrinsics = depth_frame.profile.as_video_stream_profile().get_intrinsics()
    width, height = depth_intrinsics.width, depth_intrinsics.height

    depth_image = np.asanyarray(depth_frame.get_data())
    color_image = np.asanyarray(color_frame.get_data())

    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale_rs = depth_sensor.get_depth_scale()  # :contentReference[oaicite:12]{index=12}

    depth_o3d = o3d.geometry.Image(depth_image.astype(np.float32))
    color_image_rgb = color_image[:, :, ::-1].copy()  # BGR->RGB :contentReference[oaicite:13]{index=13}
    color_o3d = o3d.geometry.Image(color_image_rgb)

    intr = o3d.camera.PinholeCameraIntrinsic(
        width, height,
        depth_intrinsics.fx, depth_intrinsics.fy,
        depth_intrinsics.ppx, depth_intrinsics.ppy,
    )  # :contentReference[oaicite:14]{index=14}

    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color_o3d,
        depth_o3d,
        depth_scale=1.0 / depth_scale_rs,  # :contentReference[oaicite:15]{index=15}
        depth_trunc=1.0,
        convert_rgb_to_intensity=False,
    )

    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intr)

    # 座標系補正（Y反転） :contentReference[oaicite:16]{index=16}
    T_flip = np.array([
        [1,  0, 0, 0],
        [0, -1, 0, 0],
        [0,  0, 1, 0],
        [0,  0, 0, 1],
    ], dtype=np.float64)
    pcd.transform(T_flip)

    return pcd

def icp_to_cam0(source_pcd, target_pcd, init_trans, voxel_size=0.005):
    radius = voxel_size * 2.0

    source_pcd.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=30)
    )
    target_pcd.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=30)
    )

    max_correspondence_distance_coarse = voxel_size * 10.0
    max_correspondence_distance_fine = voxel_size * 1.0

    icp_coarse = o3d.pipelines.registration.registration_icp(
        source_pcd, target_pcd,
        max_correspondence_distance_coarse, init_trans,
        o3d.pipelines.registration.TransformationEstimationPointToPlane()
    )  # :contentReference[oaicite:17]{index=17}

    icp_fine = o3d.pipelines.registration.registration_icp(
        source_pcd, target_pcd,
        max_correspondence_distance_fine, icp_coarse.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPlane()
    )  # :contentReference[oaicite:18]{index=18}

    print("ICP fitness:", icp_fine.fitness, "rmse:", icp_fine.inlier_rmse)
    return icp_fine.transformation

# =========================================================
# RS_ARMarkerRead.py 側（既存仕様）
# =========================================================
TAG_SIZE_M = 0.021  # :contentReference[oaicite:19]{index=19}
TARGET_PITCH_DEG = [-60.0, -40.0, -20.0, 0.0, 20.0, 40.0, 60.0]
PITCH_TOL_DEG = 1.0
HOLD_FRAMES = 15

def create_detector():
    return Detector(
        families="tag36h11",
        nthreads=4,
        quad_decimate=1.0,
        quad_sigma=0.0,
        refine_edges=True,
        decode_sharpening=0.25,
        debug=False
    )  # :contentReference[oaicite:20]{index=20}

def rotation_matrix_to_euler(R):
    # ZYX順 (R = Rz(yaw) * Ry(pitch) * Rx(roll)) 想定 :contentReference[oaicite:21]{index=21}
    sy = math.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    singular = sy < 1e-6
    if not singular:
        yaw   = math.atan2(R[1, 0], R[0, 0])
        pitch = math.atan2(-R[2, 0], sy)
        roll  = math.atan2(R[2, 1], R[2, 2])
    else:
        yaw   = math.atan2(-R[0, 1], R[1, 1])
        pitch = math.atan2(-R[2, 0], sy)
        roll  = 0.0
    return roll, pitch, yaw

def match_pitch_targets(pitch_rad):
    pitch_deg = math.degrees(pitch_rad)
    nearest_20 = round(pitch_deg / 20.0) * 20.0  # 20度刻み :contentReference[oaicite:22]{index=22}
    if abs(pitch_deg - nearest_20) <= PITCH_TOL_DEG and (-60.0 <= nearest_20 <= 60.0):
        return True, pitch_deg, nearest_20
    return False, pitch_deg, nearest_20

def get_color_intrinsics_from_profile(profile):
    color_stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
    intr = color_stream.get_intrinsics()
    return (float(intr.fx), float(intr.fy), float(intr.ppx), float(intr.ppy))

# =========================================================
# 実行オプション（あなたの要望：重い場合はPLYのみOK）
# =========================================================
SAVE_ONLY_PLY = True  # True: PLY保存のみ（Open3D表示/Matplotlib投影なし）
SHOW_CAM0_WINDOW = True  # カメラ0の検出状況を表示したい場合 True

def capture_and_process_3cams(pipelines, profiles, pitch_label_deg):
    color_frames = [None] * len(pipelines)
    depth_frames = [None] * len(pipelines)

    # 各カメラごとの align を用意（既存と同じく depth->color アライン）:contentReference[oaicite:2]{index=2}
    aligns = [rs.align(rs.stream.color) for _ in pipelines]

    def grab_one(i):
        # 1回だけ frameset を取得して返す
        return pipelines[i].wait_for_frames()

    # NUM_FRAMES 回まわして「最後のフレーム」を採用する点は既存のまま :contentReference[oaicite:3]{index=3}
    with ThreadPoolExecutor(max_workers=len(pipelines)) as ex:
        for _ in range(NUM_FRAMES):
            futures = [ex.submit(grab_one, i) for i in range(len(pipelines))]
            framesets = [f.result() for f in futures]

            # 3台分の frameset をほぼ同時刻に取得した後、各台で align を適用
            for i, fs in enumerate(framesets):
                aligned = aligns[i].process(fs)
                depth = aligned.get_depth_frame()
                color = aligned.get_color_frame()
                if not depth or not color:
                    raise RuntimeError("フレーム取得に失敗しました")
                depth_frames[i] = depth
                color_frames[i] = color

    # 点群生成 :contentReference[oaicite:25]{index=25}
    pcds = []
    for i in range(len(SERIALS)):
        pcds.append(frames_to_pointcloud(color_frames[i], depth_frames[i], profiles[i]))

    # ICPでcam1/cam2をcam0へ :contentReference[oaicite:26]{index=26}
    base_pcd = pcds[0]
    T_1_to_0_icp = icp_to_cam0(pcds[1], base_pcd, T_1_to_0)
    T_2_to_0_icp = icp_to_cam0(pcds[2], base_pcd, T_2_to_0)

    # マージ :contentReference[oaicite:27]{index=27}
    pcd0_aligned = base_pcd
    pcd1_aligned = pcds[1].transform(T_1_to_0_icp)
    pcd2_aligned = pcds[2].transform(T_2_to_0_icp)

    merged_pcd = o3d.geometry.PointCloud()
    merged_pcd += pcd0_aligned
    merged_pcd += pcd1_aligned
    merged_pcd += pcd2_aligned

    # PLY保存（角度ラベル入りにする）
    os.makedirs("PLY", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"PLY/ply/face_3cams_geom_merged_{int(pitch_label_deg)}deg_{timestamp}.ply"
    o3d.io.write_point_cloud(filename, merged_pcd)  # :contentReference[oaicite:28]{index=28}
    print(f"[SAVE] {filename}")

    # 重い表示は必要ならここでOFF/ON
    if not SAVE_ONLY_PLY:
        o3d.visualization.draw_geometries([merged_pcd])  # :contentReference[oaicite:29]{index=29}

def main():
    pipelines = []
    profiles = []
    detector = create_detector()

    hold_count = 0
    hold_target = None  # 20/40/60 のいずれか

    try:
        # 3台起動 :contentReference[oaicite:30]{index=30}
        for serial in SERIALS:
            pipeline, profile = create_pipeline(serial)
            pipelines.append(pipeline)
            profiles.append(profile)

        # cam0 intrinsics を AprilTag姿勢推定に使う（camera_params） :contentReference[oaicite:31]{index=31}
        camera_params = get_color_intrinsics_from_profile(profiles[0])

        print("[INFO] Running...  Stop with Ctrl+C (KeyboardInterrupt).")

        while True:
            # cam0で AprilTag 姿勢推定 :contentReference[oaicite:32]{index=32}
            frames0 = pipelines[0].wait_for_frames()
            color0 = frames0.get_color_frame()
            if not color0:
                continue

            color_image0 = np.asanyarray(color0.get_data())
            gray0 = cv2.cvtColor(color_image0, cv2.COLOR_BGR2GRAY)

            results = detector.detect(
                gray0,
                estimate_tag_pose=True,
                camera_params=camera_params,
                tag_size=TAG_SIZE_M
            )  # :contentReference[oaicite:33]{index=33}

            matched_any = False
            matched_target = None
            frame_vis = color_image0.copy()

            for r in results:
                R = r.pose_R
                roll, pitch, yaw = rotation_matrix_to_euler(R)
                ok, pitch_deg, nearest_20 = match_pitch_targets(pitch)

                if ok:
                    matched_any = True
                    matched_target = nearest_20

                if SHOW_CAM0_WINDOW:
                    # 画面表示は RS_ARMarkerRead.py の方針に合わせて最低限 :contentReference[oaicite:34]{index=34}
                    cv2.putText(frame_vis, f"pitch={math.degrees(pitch):.1f}",
                                (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                break  # 1タグで十分

            # 30フレーム連続成立でトリガ
            if matched_any:
                if hold_target == matched_target:
                    hold_count += 1
                else:
                    hold_target = matched_target
                    hold_count = 1
            else:
                hold_target = None
                hold_count = 0

            if SHOW_CAM0_WINDOW:
                if hold_target is not None:
                    cv2.putText(frame_vis, f"HOLD {hold_target:.0f}deg {hold_count}/{HOLD_FRAMES}",
                                (30, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                cv2.imshow("Cam0 AprilTag Pose (Trigger)", frame_vis)

                # qでも終了可能（任意）
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            if hold_target is not None and hold_count >= HOLD_FRAMES:
                print(f"[TRIGGER] pitch={hold_target:.0f} deg held {HOLD_FRAMES} frames -> capture")
                capture_and_process_3cams(pipelines, profiles, pitch_label_deg=hold_target)
                # 無制限撮影のためリセットして継続
                hold_target = None
                hold_count = 0

    except KeyboardInterrupt:
        print("\n[INFO] Stopped by Ctrl+C (KeyboardInterrupt).")

    finally:
        for p in pipelines:
            try:
                p.stop()
            except Exception:
                pass
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass

if __name__ == "__main__":
    main()
