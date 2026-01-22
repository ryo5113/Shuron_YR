import pyrealsense2 as rs
import numpy as np
import open3d as o3d
from datetime import datetime
import cv2
import math
import joblib
from pupil_apriltags import Detector
import os
import copy
from concurrent.futures import ThreadPoolExecutor
import time
from collections import deque, defaultdict

# === 追加: MediaPipe ===
import mediapipe as mp

# === 追加: SVM推論（faceTrain_SVM.py の保存形式を読む想定）===
#SVM_MODEL_PATH = r"./PLY_dataset_YR/25dens/ply_svm_model.joblib"  # あなたのモデルパスに合わせる
SVM_MODEL_PATH = r"./AA/ply_svm_model.joblib"  # あなたのモデルパスに合わせる
SVM_PAYLOAD = None

def occupancy_grid_features(points: np.ndarray, grid: int) -> np.ndarray:
    pts = points.astype(np.float64, copy=True)
    pts -= pts.mean(axis=0, keepdims=True)
    max_abs = np.max(np.abs(pts))
    if max_abs > 0:
        pts /= max_abs
    pts = np.clip(pts, -1.0, 1.0)
    idx = ((pts + 1.0) * 0.5 * grid).astype(np.int64)
    idx = np.clip(idx, 0, grid - 1)
    occ = np.zeros((grid, grid, grid), dtype=np.uint8)
    occ[idx[:, 0], idx[:, 1], idx[:, 2]] = 1
    return occ.reshape(-1).astype(np.float64)

def predict_mouth_label_from_pcd(mouth_pcd: o3d.geometry.PointCloud):
    global SVM_PAYLOAD
    if SVM_PAYLOAD is None:
        raise RuntimeError("SVM_PAYLOAD is None. main()でjoblib.loadしてください。")
    if mouth_pcd is None or len(mouth_pcd.points) == 0:
        return None, None

    pts = np.asarray(mouth_pcd.points, dtype=np.float64)
    grid = int(SVM_PAYLOAD.get("grid", 10))
    feat = occupancy_grid_features(pts, grid=grid).reshape(1, -1)

    model = SVM_PAYLOAD["model"]
    label_order = SVM_PAYLOAD["label_order"]

    # --- predict_proba が使える場合（％表示）---
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(feat)[0]
        pred_idx = int(np.argmax(proba))
        pred_label = str(label_order[pred_idx])
        pred_percent = float(proba[pred_idx] * 100.0)
        percent_dict = {str(label_order[i]): float(proba[i] * 100.0) for i in range(len(label_order))}
        return pred_label, pred_percent, percent_dict

    # --- 使えない場合（％にせずスコア表示）---
    pred_id = int(model.predict(feat)[0])
    pred_label = str(label_order[pred_id])
    pred_score = None
    try:
        scores = model.decision_function(feat)
        if scores.ndim == 1:
            pred_score = float(scores[0])
        else:
            pred_score = float(scores[0, pred_id])
    except Exception:
        pass

    return pred_label, pred_score, None

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
    static_image_mode=True,
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
        "face_landmarks": face_landmarks,
    }

def build_outer_lip_polygon(face_landmarks, w, h):
    """
    戻り値: np.ndarray shape=(N,2) int32 (外周ポリゴン)
    """
    edges = mp.solutions.face_mesh.FACEMESH_LIPS  # set of (i,j)

    # 隣接リスト
    adj = defaultdict(set)
    for a, b in edges:
        adj[a].add(b)
        adj[b].add(a)

    # 連結成分ごとに「ループの順序」を作る
    visited = set()
    polys = []

    for start in adj.keys():
        if start in visited:
            continue

        # BFSで連結成分を抽出
        comp = []
        q = deque([start])
        visited.add(start)
        while q:
            n = q.popleft()
            comp.append(n)
            for nb in adj[n]:
                if nb not in visited:
                    visited.add(nb)
                    q.append(nb)

        # ループ順序化（基本: 各ノードの次数が2なら単純サイクル）
        # comp 内の任意のノードから辿って順序を作る
        s = comp[0]
        nbs = list(adj[s])
        if len(nbs) < 1:
            continue

        ordered = [s]
        prev = None
        cur = s
        nxt = nbs[0]

        # 最大長の安全策（無限ループ防止）
        for _ in range(len(comp) + 5):
            ordered.append(nxt)
            prev, cur = cur, nxt
            cand = [x for x in adj[cur] if x != prev]
            if not cand:
                break
            nxt = cand[0]
            if nxt == s:
                break

        # 終端が start に戻ったら閉路
        if ordered[-1] == s:
            ordered = ordered[:-1]

        # ピクセル座標ポリゴンへ
        poly = []
        for idx in ordered:
            lm = face_landmarks.landmark[idx]
            u = int(round(lm.x * w))
            v = int(round(lm.y * h))
            poly.append([u, v])
        poly = np.array(poly, dtype=np.int32)

        # 面積がある程度あるものだけ
        if len(poly) >= 3 and abs(cv2.contourArea(poly.reshape(-1, 1, 2))) > 1.0:
            polys.append(poly)

    if not polys:
        return None

    # 面積最大 = 外周とみなす
    areas = [abs(cv2.contourArea(p.reshape(-1, 1, 2))) for p in polys]
    outer = polys[int(np.argmax(areas))]
    return outer

def crop_pcd_by_lip_polygon_project(
    merged_pcd,               # open3d.geometry.PointCloud (cam0座標系, Y反転済み)
    lip_poly_px,              # (N,2) int32 (画像座標の唇外周ポリゴン)
    color_intrinsics,         # rs.intrinsics (fx, fy, ppx, ppy, width, height)
    T_cam_to_cam0,            # 4x4 (そのカメラ -> cam0)  ※Y反転座標系で求めたもの
    depth_frame=None,
    depth_tol_m=0.01,         # 奥行き許容範囲（depth_frameがある場合）
    mask_dilate_px=0,         # 外周を少し厚く含めたい場合（0でOK）
    debug_bgr=None,           # デバッグ用に投影画像を返す場合のBGR画像（Noneで投影画像不要）
    debug_save_path=None      # デバッグ用に投影画像を保存する場合のパス（Noneで保存不要）
):
    if lip_poly_px is None or len(lip_poly_px) < 3:
        return None

    w = color_intrinsics.width
    h = color_intrinsics.height

    # ポリゴンmask
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [lip_poly_px.reshape(-1, 1, 2)], 255)

    if mask_dilate_px > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*mask_dilate_px+1, 2*mask_dilate_px+1))
        mask = cv2.dilate(mask, k)

    # cam0 -> cam へ（T_cam_to_cam0 の逆）
    T_cam0_to_cam = np.linalg.inv(T_cam_to_cam0)

    pts0 = np.asarray(merged_pcd.points)  # (N,3) cam0座標系(Y反転済)
    if pts0.size == 0:
        return None

    pts0_h = np.hstack([pts0, np.ones((pts0.shape[0], 1), dtype=np.float64)])  # (N,4)
    pts_cam = (T_cam0_to_cam @ pts0_h.T).T[:, :3]  # (N,3) cam座標系(Y反転済)

    X = pts_cam[:, 0]
    Y = pts_cam[:, 1]
    Z = pts_cam[:, 2]

    valid = Z > 1e-6
    X = X[valid]; Y = Y[valid]; Z = Z[valid]
    valid_idx = np.where(valid)[0]

    fx, fy, cx, cy = color_intrinsics.fx, color_intrinsics.fy, color_intrinsics.ppx, color_intrinsics.ppy

    # Y反転座標系なので v の式が「cy - fy*(Y/Z)」
    u = (fx * (X / Z) + cx).astype(np.int32)
    v = (cy - fy * (Y / Z)).astype(np.int32)

    in_img = (u >= 0) & (u < w) & (v >= 0) & (v < h)
    u = u[in_img]; v = v[in_img]
    idx = valid_idx[in_img]
    Z_img = Z[in_img]  # depth_frame と比較するために保持

    inside = mask[v, u] > 0
    inside_i = np.where(inside)[0]
    if inside_i.size == 0:
        return None

    u_in = u[inside_i]
    v_in = v[inside_i]
    z_in = Z_img[inside_i]
    idx_in = idx[inside_i]

    pix_key = (v_in.astype(np.int64) * w + u_in.astype(np.int64))
    uniq, inv = np.unique(pix_key, return_inverse=True)

    minz = np.full((uniq.shape[0],), np.inf, dtype=np.float64)
    np.minimum.at(minz, inv, z_in)

    front_band_m = 0.003  # 例: 3mm。唇表面の厚み/ノイズ分だけ許容
    is_front = z_in <= (minz[inv] + front_band_m)

    keep = is_front
    keep_idx = idx_in[keep].tolist()
    if not keep_idx:
        return None

    # ここからデバッグ画像保存（任意）
    if debug_bgr is not None and debug_save_path is not None:
        vis = cv2.bitwise_and(debug_bgr, debug_bgr, mask=mask)

        # 1) 唇外周ポリゴンを描く
        cv2.polylines(vis, [lip_poly_px.reshape(-1, 1, 2)], isClosed=True, color=(0, 255, 255), thickness=2)

        os.makedirs(os.path.dirname(debug_save_path), exist_ok=True)
        cv2.imwrite(debug_save_path, vis)

    if not keep_idx:
        return None

    # Open3Dで抽出
    mouth_pcd = merged_pcd.select_by_index(keep_idx)
    return mouth_pcd

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

def keep_largest_cluster_dbscan(pcd, eps=0.006, min_points=30):
    """
    Open3D DBSCANでクラスタリングし、最大クラスタのみ残す。
    eps: 近傍半径[m], min_points: クラスタ最小点数
    """
    if pcd is None or len(pcd.points) == 0:
        return pcd

    labels = np.asarray(pcd.cluster_dbscan(eps=eps, min_points=min_points, print_progress=False))
    if labels.size == 0:
        return pcd

    valid = labels >= 0
    if not np.any(valid):
        # 全点がノイズ扱い(-1)になった場合は、削り過ぎ防止のためそのまま返す
        return pcd

    # 最大クラスタID
    counts = np.bincount(labels[valid])
    largest = int(np.argmax(counts))

    keep_idx = np.where(labels == largest)[0].tolist()
    return pcd.select_by_index(keep_idx)

# =========================================================
# 実行オプション
# =========================================================
SAVE_ONLY_PLY = True      # True: PLY保存のみ（Open3D表示/Matplotlib投影なし）
SHOW_CAM0_WINDOW = True   # カメラ0の検出状況を表示したい場合 True

def capture_and_process_3cams(pipelines, profiles, pitch_label_deg, tag_R, tag_t):
    color_frames = [None] * len(pipelines)
    depth_frames = [None] * len(pipelines)

    aligns = [rs.align(rs.stream.color) for _ in pipelines]

    def grab_one(i):
        return pipelines[i].wait_for_frames()
    
    def make_T_from_Rt(R, t):
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = np.asarray(R, dtype=np.float64)
        T[:3, 3]  = np.asarray(t, dtype=np.float64).reshape(3)
        return T

    def transform_xyz(xyz, T):
        xyz = np.asarray(xyz, dtype=np.float64).reshape(3)
        p = np.ones(4, dtype=np.float64)
        p[:3] = xyz
        q = T @ p
        return q[:3]

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
    os.makedirs("PLY/pre", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Tag姿勢の記録 
    os.makedirs("PLY/pre/tag_pose", exist_ok=True)

    roll, pitch, yaw = rotation_matrix_to_euler(tag_R)
    # main側と同じ符号系に揃えたいなら pitch を反転した値も保存しておく
    pitch_deg_raw = math.degrees(pitch)
    pitch_deg_flipped = -pitch_deg_raw

    pose_path = f"PLY/pre/tag_pose/tag_pose_{timestamp}.txt"
    with open(pose_path, "w", encoding="utf-8") as f:
        f.write(f"pitch_label_deg(arg): {pitch_label_deg}\n")
        f.write("R_tag (Cam->Tag):\n")
        f.write(np.array2string(np.asarray(tag_R), precision=8, suppress_small=False))
        f.write("\n")
        f.write(f"t_tag (Cam->Tag) [m]: {np.asarray(tag_t).reshape(3)}\n")
        f.write(f"euler_deg (raw) roll,pitch,yaw: {math.degrees(roll):.6f}, {pitch_deg_raw:.6f}, {math.degrees(yaw):.6f}\n")
        f.write(f"pitch_deg_flipped(main style): {pitch_deg_flipped:.6f}\n")

    print(f"[SAVE] Tag pose: {pose_path}")

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
    os.makedirs("PLY/pre/raw_face", exist_ok=True)
    for i, pcd_raw in enumerate(raw_pcds):
        raw_path = f"PLY/pre/raw_face/face_cam{i}_raw_{int(pitch_label_deg)}deg_{timestamp}.ply"
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

    # tag_R, tag_t から Cam0->Tag 変換を作る（仮定：Cam→Tag）
    T_cam0_to_tag_raw = make_T_from_Rt(tag_R, tag_t)
    T_cam0_to_tag = T_cam0_to_tag_raw @ T_FLIP  # Y反転補正

    # PLY保存（従来どおりの結合PLY）
    filename = f"PLY/pre/face_3cams_geom_merged_{int(pitch_label_deg)}deg_{timestamp}.ply"
    o3d.io.write_point_cloud(filename, merged_pcd)
    print(f"[SAVE] {filename}")

    # PLY保存（追加：カメラ色付き結合PLY）
    filename_camcolor = f"PLY/pre/face_3cams_geom_merged_camcolor_{int(pitch_label_deg)}deg_{timestamp}.ply"
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

        # Tag座標系へ
        pts_tag = {k: transform_xyz(v, T_cam0_to_tag) for k, v in pts.items()}

        # 「唇中心」を原点に（ここでは4点平均を唇中心と定義）
        lip_center_tag = (pts_tag["upper"] + pts_tag["lower"] + pts_tag["left"] + pts_tag["right"]) / 4.0

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

        os.makedirs("PLY/pre/lip_metrics", exist_ok=True)
        txt_path = f"PLY/pre/lip_metrics/lip_metrics_{int(pitch_label_deg)}deg_{timestamp}.txt"
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

        face_landmarks = selected["face_landmarks"]
        cam_index = selected["camera_index"]
        debug_bgr = np.asanyarray(color_frames[cam_index].get_data()).copy()  # BGR

        # そのカメラの color intrinsics（aligned color frameのprofileから取る）
        color_intr = color_frames[cam_index].profile.as_video_stream_profile().get_intrinsics()

        # 唇外周ポリゴン（画像座標）
        h, w, _ = np.asanyarray(color_frames[cam_index].get_data()).shape
        lip_poly = build_outer_lip_polygon(face_landmarks, w, h)

        # cam_index の T_cam_to_cam0 を用意（あなたの既存変数に合わせてください）
        # cam0: np.eye(4), cam1: T_1_to_0_refined, cam2: T_2_to_0_refined のような形
        if cam_index == 0:
            T_cam_to_cam0 = np.eye(4, dtype=np.float64)
        elif cam_index == 1:
            T_cam_to_cam0 = T_1_to_0_icp
        else:
            T_cam_to_cam0 = T_2_to_0_icp

        dbg_dir = "PLY/pre/lip_mask_debug"
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        debug_path = f"{dbg_dir}/lipmask_cam{cam_index}_{int(pitch_label_deg)}deg_{ts}.png"
        
        mouth_pcd = crop_pcd_by_lip_polygon_project(
            merged_pcd=merged_pcd,
            lip_poly_px=lip_poly,
            color_intrinsics=color_intr,
            T_cam_to_cam0=T_cam_to_cam0,
            depth_frame=depth_frames[cam_index],
            depth_tol_m=0.01,
            mask_dilate_px=0,
            debug_bgr=debug_bgr,
            debug_save_path=debug_path
        )
        print(f"[LIP] 唇マスク投影デバッグ画像を保存しました: {debug_path}")

        mouth_pcd_camcolor = crop_pcd_by_lip_polygon_project(
            merged_pcd=merged_pcd_camcolor,
            lip_poly_px=lip_poly,
            color_intrinsics=color_intr,
            T_cam_to_cam0=T_cam_to_cam0,
            depth_frame=depth_frames[cam_index],
            depth_tol_m=0.01,
            mask_dilate_px=0
        )

        if mouth_pcd is None or len(mouth_pcd.points) == 0:
            print("[LIP] mouth_pcd is empty (polygon crop). skip save.")
            return
        
        # 最大クラスタのみ残す
        mouth_pcd = keep_largest_cluster_dbscan(mouth_pcd, eps=0.006, min_points=30)
        mouth_pcd_camcolor = keep_largest_cluster_dbscan(mouth_pcd_camcolor, eps=0.006, min_points=30)

        # クラスタリング後に0点になる可能性があるので再チェック
        if mouth_pcd is None or len(mouth_pcd.points) == 0:
            print("[LIP] mouth_pcd became empty after clustering. skip save.")
            return

        def transform_pcd_points(pcd, T):
            pts = np.asarray(pcd.points)
            pts_h = np.hstack([pts, np.ones((len(pts), 1), dtype=np.float64)])
            pts2 = (T @ pts_h.T).T[:, :3]
            pcd2 = o3d.geometry.PointCloud()
            pcd2.points = o3d.utility.Vector3dVector(pts2)
            if pcd.has_colors():
                pcd2.colors = pcd.colors
            if pcd.has_normals():
                pcd2.normals = pcd.normals
            return pcd2

        # 口点群（Cam0座標系）のはずなので、そのまま mouth_pcd_cam0 として扱う
        mouth_pcd_cam0 = mouth_pcd
        mouth_pcd_cam0_camcolor = mouth_pcd_camcolor

        mouth_pcd_tag = transform_pcd_points(mouth_pcd_cam0, T_cam0_to_tag)
        mouth_pcd_camcolor_tag = transform_pcd_points(mouth_pcd_cam0_camcolor, T_cam0_to_tag)

        # 唇中心を原点に（Tag座標上で）
        pts = np.asarray(mouth_pcd_tag.points)
        pts_centered = pts - lip_center_tag.reshape(1, 3)

        # 口の横幅方向をX軸に
        x_axis = pts_tag["right"] - pts_tag["left"]
        x_axis = x_axis / (np.linalg.norm(x_axis) + 1e-9)

        # 口の縦方向をY軸に（X軸と直交化）
        y_axis = pts_tag["upper"] - pts_tag["lower"]
        y_axis = y_axis - np.dot(y_axis, x_axis) * x_axis
        y_axis = y_axis / (np.linalg.norm(y_axis) + 1e-9)

        # 右手系のZ軸
        z_axis = np.cross(x_axis, y_axis)
        z_axis = z_axis / (np.linalg.norm(z_axis) + 1e-9)

        # 回転行列（列に各軸）
        R_mouth = np.stack([x_axis, y_axis, z_axis], axis=1)

        os.makedirs("PLY/pre/mouth_pose", exist_ok=True)

        roll_m, pitch_m, yaw_m = rotation_matrix_to_euler(R_mouth)
        mouth_pose_path = f"PLY/pre/mouth_pose/mouth_pose_{timestamp}.txt"

        # ★追加：撮影時点のARマーカー姿勢（Cam->Tag）も同じtxtに残す
        roll_t, pitch_t, yaw_t = rotation_matrix_to_euler(tag_R)
        tag_roll_deg  = math.degrees(roll_t)
        tag_pitch_deg = math.degrees(pitch_t)
        tag_yaw_deg   = math.degrees(yaw_t)

        # main側と同じ符号系に揃えたいなら pitch を反転した値も併記
        tag_pitch_deg_flipped = -tag_pitch_deg

        with open(mouth_pose_path, "w", encoding="utf-8") as f:
            f.write(f"pitch_label_deg(arg): {pitch_label_deg}\n")
            f.write(f"camera_index(lip source): {selected['camera_index']}\n")

            # 口中心（Tag座標）
            f.write("lip_center_tag (Tag coord) [m]:\n")
            f.write(np.array2string(lip_center_tag.reshape(3), precision=8, suppress_small=False))
            f.write("\n\n")

            # 口姿勢（Tag座標に対する口ローカル軸）
            f.write("R_mouth (Tag->Mouth axes as columns):\n")
            f.write(np.array2string(R_mouth, precision=8, suppress_small=False))
            f.write("\n")

            f.write(f"euler_deg_from_R_mouth (roll,pitch,yaw): "
                    f"{math.degrees(roll_m):.6f}, {math.degrees(pitch_m):.6f}, {math.degrees(yaw_m):.6f}\n")
            
            f.write("\n[TAG POSE]\n")
            f.write("R_tag (Cam->Tag):\n")
            f.write(np.array2string(np.asarray(tag_R), precision=8, suppress_small=False))
            f.write("\n")
            f.write(f"t_tag (Cam->Tag) [m]: {np.asarray(tag_t).reshape(3)}\n")
            f.write(f"euler_deg_tag (roll,pitch,yaw): {tag_roll_deg:.6f}, {tag_pitch_deg:.6f}, {tag_yaw_deg:.6f}\n")
            f.write(f"tag_pitch_deg_flipped(main style): {tag_pitch_deg_flipped:.6f}\n")

        print(f"[SAVE] Mouth pose: {mouth_pose_path}")

        # 「口ローカル座標」へ： p_local = R^T * p_centered
        pts_centered = (R_mouth.T @ pts_centered.T).T

        mouth_pcd_tag_centered = o3d.geometry.PointCloud()
        mouth_pcd_tag_centered.points = o3d.utility.Vector3dVector(pts_centered)
        if mouth_pcd_tag.has_colors():
            mouth_pcd_tag_centered.colors = mouth_pcd_tag.colors

        # camcolor側も同様に中心原点化（色は保持したまま点だけ平行移動）
        pts_c = np.asarray(mouth_pcd_camcolor_tag.points)
        pts_c_centered = pts_c - lip_center_tag.reshape(1, 3)

        mouth_pcd_camcolor_tag_centered = o3d.geometry.PointCloud()
        mouth_pcd_camcolor_tag_centered.points = o3d.utility.Vector3dVector(pts_c_centered)
        if mouth_pcd_camcolor_tag.has_colors():
            mouth_pcd_camcolor_tag_centered.colors = mouth_pcd_camcolor_tag.colors

        if mouth_pcd is None or len(mouth_pcd.points) == 0:
            print("[LIP] mouth_pcd is empty (polygon crop). skip save.")
        else:
            os.makedirs("PLY/pre/mouth", exist_ok=True)

        # === 追加: SVMでリアルタイム分類（mouth_pcd_tag_centered を使用）===
        pred_label, pred_value, detail = predict_mouth_label_from_pcd(mouth_pcd_tag_centered)
        os.makedirs("PLY/pre/pred", exist_ok=True)
        pred_txt = f"PLY/pre/pred/pred_{int(pitch_label_deg)}deg_{timestamp}.txt"
        with open(pred_txt, "w", encoding="utf-8") as pf:
            pf.write(f"pitch_label_deg: {pitch_label_deg}\n")
            pf.write(f"camera_index(lip source): {selected['camera_index']}\n")
            pf.write(f"pred_label: {pred_label}\n")

            if detail is not None:
                # predict_proba が使えた → %表示
                pf.write(f"pred_percent: {pred_value:.2f}\n")
                pf.write("class_percent:\n")
                for k, v in detail.items():
                    pf.write(f"  {k}: {v:.2f}\n")
            else:
                # predict_proba が使えない → スコア表示
                if pred_value is not None:
                    pf.write(f"decision_score: {pred_value:.6f}\n")

        print(f"[PRED] {pred_label}  (saved: {pred_txt})")

        mouth_filename = f"PLY/pre/mouth/mouth_{int(pitch_label_deg)}deg_{timestamp}.ply"
        o3d.io.write_point_cloud(mouth_filename, mouth_pcd_tag_centered)  # ←ここが重要（変換後を保存）
        print(f"[SAVE] mouth pcd: {mouth_filename}")

        mouth_filename_camcolor = f"PLY/pre/mouth/mouth_camcolor_{int(pitch_label_deg)}deg_{timestamp}.ply"
        o3d.io.write_point_cloud(mouth_filename_camcolor, mouth_pcd_camcolor_tag_centered)  # ←変換後を保存
        print(f"[SAVE] mouth pcd (camcolor): {mouth_filename_camcolor}")

        # ★ここから画像保存
        annotated = selected.get("annotated_image", None)
        if annotated is not None:
            os.makedirs("PLY/pre/mediapipe_img", exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            img_path = f"PLY/pre/mediapipe_img/lip_cam{selected['camera_index']}_{ts}.png"
            cv2.imwrite(img_path, annotated)
            print(f"[LIP] MediaPipe描画画像を保存しました: {img_path}")

    # ===============================================


        # 追加: 呼び出し元へ保存パスと予測を返す（必要なら利用）
        try:
            return {
                "mouth_ply": mouth_filename if 'mouth_filename' in locals() else None,
                "mediapipe_img": img_path if 'img_path' in locals() else None,
                "pred_label": pred_label if 'pred_label' in locals() else None,
            }
        except Exception:
            return None

            if not SAVE_ONLY_PLY:
                o3d.visualization.draw_geometries([merged_pcd_camcolor])

def main():
    pipelines = []
    profiles = []
    detector = create_detector()
    pitch_hist = deque(maxlen=10)  # 直近10フレームのpitch[deg]

    try:
        # 3台起動
        for serial in SERIALS:
            pipeline, profile = create_pipeline(serial)
            pipelines.append(pipeline)
            profiles.append(profile)

        # cam0 intrinsics を AprilTag 姿勢推定に使う
        camera_params = get_color_intrinsics_from_profile(profiles[0])

        # === 追加: SVMモデル読み込み（faceTrain_SVM.py の出力）===
        global SVM_PAYLOAD
        if not os.path.exists(SVM_MODEL_PATH):
            raise FileNotFoundError(f"SVM model not found: {SVM_MODEL_PATH}")
        SVM_PAYLOAD = joblib.load(SVM_MODEL_PATH)
        print(f"[INFO] Loaded SVM model: {SVM_MODEL_PATH}")
        print(f"       label_order={SVM_PAYLOAD.get('label_order')}, grid={SVM_PAYLOAD.get('grid')}")


        print("[INFO] Running...  Stop with Ctrl+C (KeyboardInterrupt).")

        is_processing = False
        last_pred_text = "PRED: --"
        last_pred_time = 0.0
        PRED_SHOW_SEC = 10.0  # 何秒表示するか

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
                R_tag = r.pose_R
                t_tag = r.pose_t  #（3要素の並進ベクトルの想定）
                roll, pitch, yaw = rotation_matrix_to_euler(R_tag)
                pitch = -pitch  # 頭の回転方向に合わせて符号反転
                # 角度（deg）
                roll_deg  = math.degrees(roll)
                pitch_deg = math.degrees(pitch)
                yaw_deg   = math.degrees(yaw)

                # （必要なら）Pitchだけ平滑化は残す：既存の pitch_hist を使う
                pitch_hist.append(pitch_deg)
                pitch_deg_smooth = sum(pitch_hist) / len(pitch_hist)

                # 20度刻み判定をやめ、Tagが見えたら撮影可能にする
                matched_any = True

                break

            # 撮影可能条件（最低限）
            # - Tagが検出されている（resultsからR_tag, t_tagが取れている）
            # - pitchが指定範囲内（例：-40〜40）  ※あなたの要件
            CAPTURE_MIN_DEG = -60.0
            CAPTURE_MAX_DEG =  60.0

            # pitch_deg_smooth は上で計算済み（直近平均）
            capture_ready = (matched_any and (CAPTURE_MIN_DEG <= pitch_deg_smooth <= CAPTURE_MAX_DEG))

            if SHOW_CAM0_WINDOW:
                # 撮影可否：Tagが見えていれば READY
                capture_ready = matched_any
                status = "READY" if capture_ready else "NG"

                cv2.putText(frame_vis, f"CAPTURE: {status}",
                            (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                            (0, 255, 0) if capture_ready else (0, 0, 255), 2)
                
                if is_processing:
                    cv2.putText(frame_vis, "PROCESSING... DO NOT MOVE",
                                (30, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

                if matched_any:
                    # 表示するPitchは、既存の平滑化後を使う（必要なければ pitch_deg にしてOK）
                    cv2.putText(frame_vis, f"R:{roll_deg:+.1f}  P:{pitch_deg_smooth:+.1f}  Y:{yaw_deg:+.1f} [deg]",
                                (30, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                else:
                    cv2.putText(frame_vis, "R:--  P:--  Y:-- [deg]",
                                (30, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

                now = time.time()
                if now - last_pred_time <= PRED_SHOW_SEC:
                    cv2.putText(frame_vis, last_pred_text,
                                (30, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
                cv2.imshow("Cam0 AprilTag Pose (Trigger)", frame_vis)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break

                # 手動撮影キー（例：'c'）
                if key == ord('c'):
                    if capture_ready and (not is_processing):
                        is_processing = True

                        overlay = frame_vis.copy()
                        cv2.putText(overlay, "CAPTURED. PROCESSING... DO NOT MOVE",
                                    (30, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                        cv2.imshow("Cam0 AprilTag Pose (Trigger)", overlay)
                        cv2.waitKey(1)

                        print(f"[TRIGGER] manual capture | pitch={pitch_deg_smooth:.2f} deg")
                        # 記録する姿勢＝この時点の Tag の R,t を capture 側へ渡す
                        # pitch_label_deg は「記録したい角度」として、ここでは pitch_deg_smooth を渡す
                        result = capture_and_process_3cams(
                            pipelines, profiles,
                            pitch_label_deg=pitch_deg_smooth,
                            tag_R=R_tag, tag_t=t_tag
                        )
                        if result and result.get("pred_label") is not None:
                            last_pred_text = f"PRED: {result['pred_label']}"
                            last_pred_time = time.time()
                            print(f"[PRED] result label: {result['pred_label']}")
                            # UIに一瞬表示（約1秒）
                            try:
                                overlay2 = frame_vis.copy()
                                cv2.putText(overlay2, f"PRED: {result['pred_label']}",
                                            (30, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
                                for _ in range(10):
                                    cv2.imshow("Cam0 AprilTag Pose (Trigger)", overlay2)
                                    if cv2.waitKey(100) & 0xFF == ord('q'):
                                        break
                            except Exception:
                                pass

                        is_processing = False
                    else:
                        print("[TRIGGER] manual capture ignored (not ready)")

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
