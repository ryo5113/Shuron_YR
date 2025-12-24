import pyrealsense2 as rs
import numpy as np
import open3d as o3d
from datetime import datetime
import cv2
import math
from pupil_apriltags import Detector
import os
import copy
from concurrent.futures import ThreadPoolExecutor
import time
from collections import deque

# === 追加: MediaPipe ===
import mediapipe as mp


# =========================================================
# 追加: 3D-CNN 教師データ用（口周辺点群 -> 64^3 RGBボクセル）
# =========================================================
VOXEL_GRID_SIZE = 64
SAVE_MOUTH_VOXEL = True  # True: 口周辺点群から 64^3 RGB ボクセル(.npz)を保存

def voxelize_rgb_mean(mouth_pcd: o3d.geometry.PointCloud,
                      min_xyz: np.ndarray,
                      max_xyz: np.ndarray,
                      grid: int = VOXEL_GRID_SIZE) -> np.ndarray:
    """
    口周辺点群（Open3D PointCloud, colorsあり）を、AABB[min_xyz, max_xyz] の範囲で
    (C=3, D=grid, H=grid, W=grid) のRGB平均ボクセルへ変換する。

    戻り値:
      voxel_rgb: np.ndarray float32 shape (3, grid, grid, grid)
                 ※空ボクセルは0のまま
    """
    if not mouth_pcd.has_colors():
        raise ValueError("mouth_pcd に colors がありません（RGB前提）")

    pts = np.asarray(mouth_pcd.points, dtype=np.float64)   # (N,3)
    cols = np.asarray(mouth_pcd.colors, dtype=np.float64)  # (N,3) 通常 0..1

    min_xyz = np.asarray(min_xyz, dtype=np.float64)
    max_xyz = np.asarray(max_xyz, dtype=np.float64)

    extent = max_xyz - min_xyz
    extent = np.maximum(extent, 1e-9)  # 0除算回避

    # bbox内を [0,1] に正規化 → [0, grid-1] のインデックスへ
    uvw = (pts - min_xyz) / extent
    ijk = np.floor(uvw * (grid - 1)).astype(np.int32)

    # bbox外を除外（数値誤差対策）
    m = np.all((ijk >= 0) & (ijk < grid), axis=1)
    ijk = ijk[m]
    cols = cols[m]

    acc = np.zeros((grid, grid, grid, 3), dtype=np.float64)
    cnt = np.zeros((grid, grid, grid, 1), dtype=np.float64)

    x, y, z = ijk[:, 0], ijk[:, 1], ijk[:, 2]
    np.add.at(acc, (x, y, z, slice(None)), cols)
    np.add.at(cnt, (x, y, z, 0), 1.0)

    nonzero = cnt[..., 0] > 0
    acc[nonzero] /= cnt[nonzero]

    # PyTorch Conv3d向けの軸: (C, D, H, W) = (3, Z, Y, X)
    voxel = acc.transpose(3, 2, 1, 0).astype(np.float32)
    return voxel

# =========================================================
# 3Cam_rot.py 側（既存設定）
# =========================================================
SERIALS = [
    "047322070108",  # カメラ0（基準）
    "913522070157",  # カメラ1
    "108322073166",  # カメラ2
]
NUM_FRAMES = 30  # 最後のフレームを使用

def make_extrinsic(tx, ty, tz, angle_deg):
    T = np.eye(4, dtype=np.float64)
    angle = np.deg2rad(angle_deg)
    # y軸周りの回転
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
T_1_to_0 = make_extrinsic(-0.29, 0.0, 0.20,  45.0)
T_2_to_0 = make_extrinsic( 0.285, 0.0, 0.20, -45.0)

# 点群で使っている座標系補正（Y反転）を 3D特徴点にも合わせるために共有
T_FLIP = np.array([
    [1,  0, 0, 0],
    [0, -1, 0, 0],
    [0,  0, 1, 0],
    [0,  0, 0, 1],
], dtype=np.float64)

def apply_manual_color_settings(profile, exposure=None, gain=None, white_balance=None):
    """
    profile: pipeline.start(config) の戻り値
    exposure: int/float (例: 8000)
    gain: int/float (例: 16)
    white_balance: int/float (例: 4500)
    """
    dev = profile.get_device()

    # RGBセンサーを探す（環境によりインデックス固定は危険なので総当たり）
    for s in dev.query_sensors():
        name = s.get_info(rs.camera_info.name)
        if "RGB" not in name and "Color" not in name:
            continue

        # 自動露出OFF → 露出/ゲイン固定
        if exposure is not None and s.supports(rs.option.enable_auto_exposure):
            s.set_option(rs.option.enable_auto_exposure, 0)  # 0=False
        if exposure is not None and s.supports(rs.option.exposure):
            s.set_option(rs.option.exposure, float(exposure))
        if gain is not None and s.supports(rs.option.gain):
            s.set_option(rs.option.gain, float(gain))

        # 自動WB OFF → WB固定
        if white_balance is not None and s.supports(rs.option.enable_auto_white_balance):
            s.set_option(rs.option.enable_auto_white_balance, 0)  # 0=False
        if white_balance is not None and s.supports(rs.option.white_balance):
            s.set_option(rs.option.white_balance, float(white_balance))

        # RGBセンサーに適用したら終了
        break

def create_pipeline(serial):
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device(serial)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    profile = pipeline.start(config)
    # ★追加：RGBの明度を手動で揃える（全台同じ値にする）
    apply_manual_color_settings(
        profile,
        exposure=300,        # ←ここはあなたが揃えたい値
        gain=16,              # ←必要なら
        white_balance=4500    # ←必要なら
    )
    return pipeline, profile

def frames_to_pointcloud(color_frame, depth_frame, profile, apply_flip=True, return_raw=False):
    depth_intrinsics = depth_frame.profile.as_video_stream_profile().get_intrinsics()
    width, height = depth_intrinsics.width, depth_intrinsics.height

    depth_image = np.asanyarray(depth_frame.get_data())
    color_image = np.asanyarray(color_frame.get_data())

    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale_rs = depth_sensor.get_depth_scale()

    depth_o3d = o3d.geometry.Image(depth_image.astype(np.float32))
    color_image_rgb = color_image[:, :, ::-1].copy()  # BGR->RGB
    color_o3d = o3d.geometry.Image(color_image_rgb)

    intr = o3d.camera.PinholeCameraIntrinsic(
        width, height,
        depth_intrinsics.fx, depth_intrinsics.fy,
        depth_intrinsics.ppx, depth_intrinsics.ppy,
    )

    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color_o3d,
        depth_o3d,
        depth_scale=1.0 / depth_scale_rs,
        depth_trunc=1.0,
        convert_rgb_to_intensity=False,
    )

    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intr)

    # return_raw=True の場合、座標変換前の点群を退避（各カメラの「そのまま」PLY保存用）
    pcd_raw = copy.deepcopy(pcd) if return_raw else None

    # 座標系補正（Y反転）※従来どおり ICP/結合 用
    if apply_flip:
        pcd.transform(T_FLIP)

    if return_raw:
        return pcd, pcd_raw
    return pcd

def icp_to_cam0(source_pcd, target_pcd, init_trans, source_cam_index, voxel_size=0.005):
    #source_ds = source_pcd.voxel_down_sample(voxel_size) # ダウンサンプリング
    #target_ds = target_pcd.voxel_down_sample(voxel_size)
    radius = voxel_size * 2.0
    # ダウンサンプリングなしの場合はpcdを直接使う

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
    )

    icp_fine = o3d.pipelines.registration.registration_icp(
        source_pcd, target_pcd,
        max_correspondence_distance_fine, icp_coarse.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPlane()
    )

    print(f"[ICP] Cam{source_cam_index} -> Cam0 | fitness: {icp_fine.fitness:.6f}  rmse: {icp_fine.inlier_rmse:.6f}")
    return icp_fine.transformation

# =========================================================
# RS_ARMarkerRead.py 側（既存仕様）
# =========================================================
TAG_SIZE_M = 0.041
TARGET_PITCH_DEG = [-60.0, -40.0, -20.0, 0.0, 20.0, 40.0, 60.0]
PITCH_TOL_DEG = 1.0
HOLD_FRAMES = 30

def create_detector():
    return Detector(
        families="tag36h11",
        nthreads=4,
        quad_decimate=1.0,
        quad_sigma=0.0,
        refine_edges=True,
        decode_sharpening=0.25,
        debug=False
    )

def rotation_matrix_to_euler(R):
    # ZYX順 (R = Rz(yaw) * Ry(pitch) * Rx(roll)) 想定
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
    nearest_20 = round(pitch_deg / 20.0) * 20.0  # 20度刻み
    if abs(pitch_deg - nearest_20) <= PITCH_TOL_DEG and (-60.0 <= nearest_20 <= 60.0):
        return True, pitch_deg, nearest_20
    return False, pitch_deg, nearest_20

def get_color_intrinsics_from_profile(profile):
    color_stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
    intr = color_stream.get_intrinsics()
    return (float(intr.fx), float(intr.fy), float(intr.ppx), float(intr.ppy))

def get_color_intrinsics_struct(profile):
    """MediaPipe→3D変換用に intrinsics 構造体そのものを取得"""
    color_stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
    return color_stream.get_intrinsics()

# =========================================================
# MediaPipe 用設定（唇4点）
# =========================================================

# Face Mesh を1回だけ作成
mp_face_mesh = mp.solutions.face_mesh
FACE_MESH = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

# 唇ランドマークID（Face Mesh 468点版）
LIP_UPPER_ID = 0   # 上唇中央付近
LIP_LOWER_ID = 17   # 下唇中央付近
LIP_LEFT_ID  = 61   # 口角（片側）
LIP_RIGHT_ID = 291  # 口角（反対側）

from mediapipe import solutions as mp_solutions
mp_drawing = mp_solutions.drawing_utils
mp_drawing_styles = mp_solutions.drawing_styles

def detect_lip_3d_for_camera(color_frame, depth_frame, profile, T_cam_to_cam0, cam_index):
    """
    1台のカメラについて:
    - RGB画像にMediaPipe Face Meshを適用して唇4点の2D座標を取得
    - depthから3D座標(そのカメラ座標系)を求める
    - 点群と同じ座標系になるようYを反転
    - T_cam_to_cam0でカメラ0座標系へ変換
    戻り値:
      { "ok": bool,
        "camera_index": int,
        "points_cam0": {"upper": np.array(3), "lower": ..., "left": ..., "right": ...}
      }
    """
    color_image = np.asanyarray(color_frame.get_data())  # BGR
    h, w, _ = color_image.shape
    rgb_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)

    results = FACE_MESH.process(rgb_image)
    if not results.multi_face_landmarks:
        return {"ok": False, "camera_index": cam_index}

    face_landmarks = results.multi_face_landmarks[0]

    # ★ここで描画用画像を作る（BGRでOK）
    annotated_image = color_image.copy()
    mp_drawing.draw_landmarks(
        image=annotated_image,
        landmark_list=face_landmarks,
        connections=mp_face_mesh.FACEMESH_LIPS,  # 唇周りだけ描画
        landmark_drawing_spec=None,
        connection_drawing_spec=mp_drawing_styles
            .get_default_face_mesh_contours_style()
    )

    # 4点分のピクセル座標を取得
    def lm_to_pixel(lm_id):
        lm = face_landmarks.landmark[lm_id]
        u = int(round(lm.x * w))
        v = int(round(lm.y * h))
        if u < 0 or u >= w or v < 0 or v >= h:
            return None
        return (u, v)

    pix_upper = lm_to_pixel(LIP_UPPER_ID)
    pix_lower = lm_to_pixel(LIP_LOWER_ID)
    pix_left  = lm_to_pixel(LIP_LEFT_ID)
    pix_right = lm_to_pixel(LIP_RIGHT_ID)

    # ===== 追加: 指定IDのプロット描画 =====
    def draw_id_point(img, pix, color, r=4):
        if pix is None:
            return
        u, v = pix
        cv2.circle(img, (u, v), r, color, -1)

    draw_id_point(annotated_image, pix_upper,  (0, 255, 255))  # BGR
    draw_id_point(annotated_image, pix_lower,  (255, 255, 0))
    draw_id_point(annotated_image, pix_left,   (0, 255, 0))
    draw_id_point(annotated_image, pix_right,  (0, 0, 255))
    # =====================================

    if any(p is None for p in [pix_upper, pix_lower, pix_left, pix_right]):
        return {"ok": False, "camera_index": cam_index}

    intr = get_color_intrinsics_struct(profile)

    def pixel_to_cam0(pix):
        u, v = pix
        # depth [m]
        z_m = depth_frame.get_distance(u, v)
        if z_m <= 0:
            return None
        # RealSenseカメラ座標系での3D点 (X, Y, Z)
        X, Y, Z = rs.rs2_deproject_pixel_to_point(intr, [u, v], z_m)
        # 点群と同じ座標系に合わせるため Y反転
        p_cam = np.array([X, -Y, Z, 1.0], dtype=np.float64)
        # カメラ0座標系に変換
        p0 = T_cam_to_cam0 @ p_cam
        return p0[:3]

    p_upper = pixel_to_cam0(pix_upper)
    p_lower = pixel_to_cam0(pix_lower)
    p_left  = pixel_to_cam0(pix_left)
    p_right = pixel_to_cam0(pix_right)

    if any(p is None for p in [p_upper, p_lower, p_left, p_right]):
        return {"ok": False, "camera_index": cam_index}

    points_cam0 = {
        "upper": p_upper,
        "lower": p_lower,
        "left":  p_left,
        "right": p_right,
    }

    return {
        "ok": True,
        "camera_index": cam_index,
        "points_cam0": points_cam0,
        "annotated_image": annotated_image,
    }

def compute_lip_metrics(points_cam0):
    """
    points_cam0: {"upper": np.array(3), "lower":..., "left":..., "right":...} （すべてカメラ0座標系）
    要望どおり:
      幅   = 左右口角のX座標の差（絶対値）
      高さ = 上下唇のY座標の差（絶対値）
      奥行 = (上下唇のZ座標のうち大きい値) - (左右口角のZ座標のうち小さい値)
    を計算して返す。
    """
    up = points_cam0["upper"]
    lo = points_cam0["lower"]
    le = points_cam0["left"]
    ri = points_cam0["right"]

    width = abs(ri[0] - le[0])
    height = abs(up[1] - lo[1])

    z_ul_min = min(up[2], lo[2])
    z_lr_max = max(le[2], ri[2])
    depth =  z_lr_max - z_ul_min

    return {
        "width":  float(width),
        "height": float(height),
        "depth":  float(depth),
    }

# =========================================================
# 実行オプション
# =========================================================
SAVE_ONLY_PLY = True      # True: PLY保存のみ（Open3D表示/Matplotlib投影なし）
SHOW_CAM0_WINDOW = True   # カメラ0の検出状況を表示したい場合 True

def capture_and_process_3cams(pipelines, profiles, pitch_label_deg):
    color_frames = [None] * len(pipelines)
    depth_frames = [None] * len(pipelines)

    aligns = [rs.align(rs.stream.color) for _ in pipelines]

    def grab_one(i):
        return pipelines[i].wait_for_frames()

    # NUM_FRAMES 回まわして「最後のフレーム」を採用
    with ThreadPoolExecutor(max_workers=len(pipelines)) as ex:
        for _ in range(NUM_FRAMES):
            futures = [ex.submit(grab_one, i) for i in range(len(pipelines))]
            framesets = [f.result() for f in futures]

            for i, fs in enumerate(framesets):
                aligned = aligns[i].process(fs)
                depth = aligned.get_depth_frame()
                color = aligned.get_color_frame()
                if not depth or not color:
                    raise RuntimeError("フレーム取得に失敗しました")
                depth_frames[i] = depth
                color_frames[i] = color

    # PLY保存（角度ラベル入り）
    os.makedirs("PLY/ml", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 点群生成（ICP/結合用は従来どおりY反転済み）
    # 併せて、各カメラの「座標変換なし」点群も取得してPLY保存する
    pcds = []
    raw_pcds = []
    for i in range(len(SERIALS)):
        pcd, pcd_raw = frames_to_pointcloud(
            color_frames[i], depth_frames[i], profiles[i],
            apply_flip=True, return_raw=True
        )
        pcds.append(pcd)
        raw_pcds.append(pcd_raw)

    # 各カメラ raw PLY を保存（座標変換なし）
    os.makedirs("PLY/ml/raw_face", exist_ok=True)
    for i, pcd_raw in enumerate(raw_pcds):
        raw_path = f"PLY/ml/raw_face/face_cam{i}_raw_{int(pitch_label_deg)}deg_{timestamp}.ply"
        o3d.io.write_point_cloud(raw_path, pcd_raw)
        print(f"[SAVE] {raw_path}")

    # ICPでcam1/cam2をcam0へ
    base_pcd = pcds[0]
    T_1_to_0_icp = icp_to_cam0(pcds[1], base_pcd, T_1_to_0, source_cam_index=1)
    T_2_to_0_icp = icp_to_cam0(pcds[2], base_pcd, T_2_to_0, source_cam_index=2)

    # マージ（従来どおり：各カメラのRGBを保持）
    pcd0_aligned = copy.deepcopy(base_pcd)
    pcd1_aligned = copy.deepcopy(pcds[1])
    pcd1_aligned.transform(T_1_to_0_icp)
    pcd2_aligned = copy.deepcopy(pcds[2])
    pcd2_aligned.transform(T_2_to_0_icp)

    merged_pcd = o3d.geometry.PointCloud()
    merged_pcd += pcd0_aligned
    merged_pcd += pcd1_aligned
    merged_pcd += pcd2_aligned

    # マージ（追加：カメラごとに色を固定して「どのカメラ由来か」分かるようにする）
    cam_colors = {
        0: np.array([1.0, 0.0, 0.0], dtype=np.float64),  # Cam0 = Red
        1: np.array([0.0, 1.0, 0.0], dtype=np.float64),  # Cam1 = Green
        2: np.array([0.0, 0.0, 1.0], dtype=np.float64),  # Cam2 = Blue
    }

    def recolor_pointcloud(pcd, rgb01):
        pts = np.asarray(pcd.points)
        if pts.size == 0:
            return pcd
        cols = np.tile(rgb01.reshape(1, 3), (pts.shape[0], 1))
        pcd.colors = o3d.utility.Vector3dVector(cols)
        return pcd

    pcd0_c = recolor_pointcloud(copy.deepcopy(pcd0_aligned), cam_colors[0])
    pcd1_c = recolor_pointcloud(copy.deepcopy(pcd1_aligned), cam_colors[1])
    pcd2_c = recolor_pointcloud(copy.deepcopy(pcd2_aligned), cam_colors[2])

    merged_pcd_camcolor = o3d.geometry.PointCloud()
    merged_pcd_camcolor += pcd0_c
    merged_pcd_camcolor += pcd1_c
    merged_pcd_camcolor += pcd2_c

    # PLY保存（従来どおりの結合PLY）
    filename = f"PLY/ml/face_3cams_geom_merged_{int(pitch_label_deg)}deg_{timestamp}.ply"
    o3d.io.write_point_cloud(filename, merged_pcd)
    print(f"[SAVE] {filename}")

    # PLY保存（追加：カメラ色付き結合PLY）
    filename_camcolor = f"PLY/ml/face_3cams_geom_merged_camcolor_{int(pitch_label_deg)}deg_{timestamp}.ply"
    o3d.io.write_point_cloud(filename_camcolor, merged_pcd_camcolor)
    print(f"[SAVE] {filename_camcolor}")



    # ==== 追加: MediaPipe による唇4点3D＋幅/高さ/奥行のテキスト出力 ====

    lip_results = []

    for cam_idx in range(len(SERIALS)):
        if cam_idx == 0:
            T_cam_to_0 = np.eye(4, dtype=np.float64)
        elif cam_idx == 1:
            T_cam_to_0 = T_1_to_0_icp
        else:
            T_cam_to_0 = T_2_to_0_icp

        res = detect_lip_3d_for_camera(
            color_frames[cam_idx],
            depth_frames[cam_idx],
            profiles[cam_idx],
            T_cam_to_0,
            cam_index=cam_idx
        )
        lip_results.append(res)

    # 0度・±20度 → カメラ0を最優先
    # ±40度以上 → 角度の符号に応じて 1 or 2 を最優先
    if 0 <= pitch_label_deg <= 21.0:
        # 正面〜20度まではカメラ0優先
        camera_priority = [0, 2]
    elif pitch_label_deg > 21.0:
        # 20度〜60度まではカメラ1優先
        camera_priority = [2, 0]
    if -21.0 <= pitch_label_deg < 0:
        # -20度〜正面まではカメラ0優先
        camera_priority = [0, 1]
    elif pitch_label_deg < -21.0:
        # -60度〜-20度まではカメラ2優先
        camera_priority = [1, 0]

    # 優先順位: カメラ0 → カメラ1 → カメラ2
    selected = None
    for idx in camera_priority:
        if idx < len(lip_results) and lip_results[idx].get("ok"):
            selected = lip_results[idx]
            break

    if selected is None or not selected.get("ok"):
        print("[LIP] MediaPipeによる唇4点検出に失敗しました。")
        # 映像にも「失敗」を表示（例：カメラ0のカラー画像を使う）
        try:
            debug_img = np.asanyarray(color_frames[0].get_data()).copy()  # Cam0の画像
            cv2.putText(debug_img,
                        "LIP NG: MediaPipe failed",
                        (30, 110),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 0, 255),
                        2)
            # 約3秒間(10回×100ms)表示し続ける
            for _ in range(10):
                cv2.imshow("Cam0 AprilTag Pose (Trigger)", debug_img)
                # 'q' が押されたら中断
                if cv2.waitKey(100) & 0xFF == ord('q'):
                    break
        except Exception as e:
            print(f"[LIP] show error failed: {e}")
    else:
        pts = selected["points_cam0"]
        metrics = compute_lip_metrics(pts)

        print(f"[LIP] 使用カメラ: Cam{selected['camera_index']} (カメラ0座標系に変換済み)")
        print("[LIP] 3D座標 (カメラ0座標系, 単位[m])")
        print(f"  upper: {pts['upper']}")
        print(f"  lower: {pts['lower']}")
        print(f"  left : {pts['left']}")
        print(f"  right: {pts['right']}")

        print("[LIP METRICS] 唇形状指標 (カメラ0座標系)")
        print(f"  幅   (左右口角X差)       : {metrics['width']:.6f} [m]")
        print(f"  高さ (上下唇Y差)         : {metrics['height']:.6f} [m]")
        print(f"  奥行 ( max(Z_left, Z_right) - min(Z_upper, Z_lower)): {metrics['depth']:.6f} [m]")

        os.makedirs("PLY/ml/lip_metrics", exist_ok=True)
        txt_path = f"PLY/ml/lip_metrics/lip_metrics_{int(pitch_label_deg)}deg_{timestamp}.txt"
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(f"pitch_label_deg: {pitch_label_deg}\n")
            f.write(f"camera_index: {selected['camera_index']}\n")
            f.write("points_cam0 (X,Y,Z in meters)\n")
            f.write(f"  upper: {pts['upper']}\n")
            f.write(f"  lower: {pts['lower']}\n")
            f.write(f"  left : {pts['left']}\n")
            f.write(f"  right: {pts['right']}\n")
            f.write("\n[LIP METRICS]\n")
            f.write(f"width : {metrics['width']:.6f}  # 左右口角X差 [m]\n")
            f.write(f"height: {metrics['height']:.6f}  # 上下唇Y差 [m]\n")
            f.write(f"depth : {metrics['depth']:.6f}  # max(Z_left, Z_right) - min(Z_upper, Z_lower) [m]\n")

        print(f"[LIP] 唇形状指標をテキスト保存しました: {txt_path}")

        # ==== 追加: 口周辺点群の切り出し（AABB crop） ====
        # 4点からAABBを作り、少しマージンを付けて口周辺のみ抽出
        pts4 = np.stack([pts["upper"], pts["lower"], pts["left"], pts["right"]], axis=0)

        min_xyz = pts4.min(axis=0)
        max_xyz = pts4.max(axis=0)

        # マージン（メートル）：データに合わせて調整する前提
        # 「幅・高さ」を基準に可変マージン（最低1cm）
        mx = max(metrics["width"]  * 0.5, 0.01)
        my = max(metrics["height"] * 0.5, 0.01)
        mz = max(metrics["width"]  * 0.5, 0.01)

        min_xyz = min_xyz - np.array([mx, my, mz], dtype=np.float64)
        max_xyz = max_xyz + np.array([mx, my, mz], dtype=np.float64)

        bbox = o3d.geometry.AxisAlignedBoundingBox(min_xyz, max_xyz)

        mouth_pcd = merged_pcd.crop(bbox)
        mouth_pcd_camcolor = merged_pcd_camcolor.crop(bbox)

        os.makedirs("PLY/ml/mouth", exist_ok=True)
        # 従来どおり（RGB保持）
        mouth_filename = f"PLY/ml/mouth/mouth_{int(pitch_label_deg)}deg_{timestamp}.ply"
        o3d.io.write_point_cloud(mouth_filename, mouth_pcd)
        print(f"[SAVE] mouth pcd: {mouth_filename}")

        # 追加（カメラ色付き）
        mouth_filename_camcolor = f"PLY/ml/mouth/mouth_camcolor_{int(pitch_label_deg)}deg_{timestamp}.ply"
        o3d.io.write_point_cloud(mouth_filename_camcolor, mouth_pcd_camcolor)
        print(f"[SAVE] mouth pcd (camcolor): {mouth_filename_camcolor}")

        # ==== 追加: 3D-CNN教師データ用 64^3 RGBボクセルの保存 ====
        if SAVE_MOUTH_VOXEL:
            voxel = voxelize_rgb_mean(mouth_pcd, min_xyz=min_xyz, max_xyz=max_xyz, grid=VOXEL_GRID_SIZE)

            os.makedirs("PLY/ml/mouth_voxel64_rgb", exist_ok=True)
            voxel_path = f"PLY/ml/mouth_voxel64_rgb/mouth_voxel64rgb_{int(pitch_label_deg)}deg_{timestamp}.npz"
            np.savez_compressed(
                voxel_path,
                voxel=voxel,                 # (3,64,64,64) float32, RGB平均
                bbox_min=min_xyz.astype(np.float32),
                bbox_max=max_xyz.astype(np.float32),
                pitch_label_deg=float(pitch_label_deg),
                camera_index=int(selected["camera_index"]),
            )
            print(f"[SAVE] mouth voxel (64^3 RGB) : {voxel_path}")
        # ===============================================


        # ★ここから画像保存
        annotated = selected.get("annotated_image", None)
        if annotated is not None:
            os.makedirs("PLY/ml/mediapipe_img", exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            img_path = f"PLY/ml/mediapipe_img/lip_cam{selected['camera_index']}_{ts}.png"
            cv2.imwrite(img_path, annotated)
            print(f"[LIP] MediaPipe描画画像を保存しました: {img_path}")

    # ===============================================

    if not SAVE_ONLY_PLY:
        o3d.visualization.draw_geometries([merged_pcd_camcolor])

def main():
    pipelines = []
    profiles = []
    detector = create_detector()
    pitch_hist = deque(maxlen=10)  # 直近10フレームのpitch[deg]

    hold_count = 0
    hold_target = None

    try:
        # 3台起動
        for serial in SERIALS:
            pipeline, profile = create_pipeline(serial)
            pipelines.append(pipeline)
            profiles.append(profile)

        # cam0 intrinsics を AprilTag 姿勢推定に使う
        camera_params = get_color_intrinsics_from_profile(profiles[0])

        print("[INFO] Running...  Stop with Ctrl+C (KeyboardInterrupt).")

        while True:
            # cam0で AprilTag 姿勢推定
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
            )

            matched_any = False
            matched_target = None
            frame_vis = color_image0.copy()

            for r in results:
                R = r.pose_R
                roll, pitch, yaw = rotation_matrix_to_euler(R)
                pitch = -pitch  # 頭の回転方向に合わせて符号反転
                pitch_deg = math.degrees(pitch)
                pitch_hist.append(pitch_deg)
                # 直近10フレームの移動平均（初期は要素数が10未満でも平均）
                pitch_deg_smooth = sum(pitch_hist) / len(pitch_hist)
                ok, pitch_deg_raw, nearest_20 = match_pitch_targets(math.radians(pitch_deg_smooth))

                if ok:
                    matched_any = True
                    matched_target = nearest_20

                if SHOW_CAM0_WINDOW:
                    cv2.putText(frame_vis, f"pitch_angle={pitch_deg_smooth:.2f}",
                                (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                break

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

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            if hold_target is not None and hold_count >= HOLD_FRAMES:
                print(f"[TRIGGER] pitch={hold_target:.0f} deg held {HOLD_FRAMES} frames -> capture")
                capture_and_process_3cams(pipelines, profiles, pitch_label_deg=hold_target)
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
