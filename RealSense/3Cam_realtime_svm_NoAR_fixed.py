import os
import time
import math
import copy
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict, deque

import cv2
import joblib
import numpy as np
import open3d as o3d
import pyrealsense2 as rs

import mediapipe as mp
from mediapipe import solutions as mp_solutions

# =========================================================
# SVM
# =========================================================
SVM_MODEL_PATH = r"./PLY_dataset_all/ply_svm_model.joblib"  # 必要に応じて変更
SVM_PAYLOAD = None


def occupancy_grid_features(points: np.ndarray, grid: int) -> np.ndarray:
    """10x10x10 等の占有ボクセル特徴量（points は任意座標系、ここで中心化+正規化）"""
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
    """戻り値は常に (pred_label, pred_value, detail) の3つ"""
    global SVM_PAYLOAD
    if SVM_PAYLOAD is None:
        raise RuntimeError("SVM_PAYLOAD is None. main()でjoblib.loadしてください。")
    if mouth_pcd is None or len(mouth_pcd.points) == 0:
        return None, None, None

    pts = np.asarray(mouth_pcd.points, dtype=np.float64)
    grid = int(SVM_PAYLOAD.get("grid", 10))
    feat = occupancy_grid_features(pts, grid=grid).reshape(1, -1)

    model = SVM_PAYLOAD["model"]
    label_order = SVM_PAYLOAD["label_order"]

    # probability が使える場合 → %
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(feat)[0]
        pred_idx = int(np.argmax(proba))
        pred_label = str(label_order[pred_idx])
        pred_percent = float(proba[pred_idx] * 100.0)
        percent_dict = {str(label_order[i]): float(proba[i] * 100.0) for i in range(len(label_order))}
        return pred_label, pred_percent, percent_dict

    # 使えない場合 → decision score
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
# RealSense / ICP
# =========================================================
SERIALS = [
    "047322070108",  # Cam0
    "913522070157",  # Cam1
    "108322073166",  # Cam2
]
NUM_FRAMES = 30


def make_extrinsic(tx, ty, tz, angle_deg):
    T = np.eye(4, dtype=np.float64)
    angle = np.deg2rad(angle_deg)
    R = np.array([
        [np.cos(angle), 0.0, np.sin(angle)],
        [0.0, 1.0, 0.0],
        [-np.sin(angle), 0.0, np.cos(angle)],
    ], dtype=np.float64)
    T[:3, :3] = R
    T[0, 3] = tx
    T[1, 3] = ty
    T[2, 3] = tz
    return T


T_0_to_0 = np.eye(4, dtype=np.float64)
T_1_to_0 = make_extrinsic(-0.29, 0.0, 0.20, 45.0)
T_2_to_0 = make_extrinsic(0.285, 0.0, 0.20, -45.0)

# 点群側のY反転（既存スクリプト踏襲）
T_FLIP = np.array(
    [[1, 0, 0, 0],
     [0, -1, 0, 0],
     [0, 0, 1, 0],
     [0, 0, 0, 1]],
    dtype=np.float64,
)


def apply_manual_color_settings(profile, exposure=None, gain=None, white_balance=None):
    dev = profile.get_device()
    for s in dev.query_sensors():
        name = s.get_info(rs.camera_info.name)
        if "RGB" not in name and "Color" not in name:
            continue

        if exposure is not None and s.supports(rs.option.enable_auto_exposure):
            s.set_option(rs.option.enable_auto_exposure, 0)
        if exposure is not None and s.supports(rs.option.exposure):
            s.set_option(rs.option.exposure, float(exposure))
        if gain is not None and s.supports(rs.option.gain):
            s.set_option(rs.option.gain, float(gain))

        if white_balance is not None and s.supports(rs.option.enable_auto_white_balance):
            s.set_option(rs.option.enable_auto_white_balance, 0)
        if white_balance is not None and s.supports(rs.option.white_balance):
            s.set_option(rs.option.white_balance, float(white_balance))
        break


def create_pipeline(serial):
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device(serial)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    profile = pipeline.start(config)

    apply_manual_color_settings(profile, exposure=300, gain=16, white_balance=4500)
    return pipeline, profile


def frames_to_pointcloud(color_frame, depth_frame, profile, apply_flip=True, return_raw=False):
    depth_intrinsics = depth_frame.profile.as_video_stream_profile().get_intrinsics()
    width, height = depth_intrinsics.width, depth_intrinsics.height

    depth_image = np.asanyarray(depth_frame.get_data())
    color_image = np.asanyarray(color_frame.get_data())

    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()

    depth_o3d = o3d.geometry.Image(depth_image.astype(np.float32))
    color_rgb = color_image[:, :, ::-1].copy()  # BGR->RGB
    color_o3d = o3d.geometry.Image(color_rgb)

    intr = o3d.camera.PinholeCameraIntrinsic(
        width, height,
        depth_intrinsics.fx, depth_intrinsics.fy,
        depth_intrinsics.ppx, depth_intrinsics.ppy,
    )

    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color_o3d,
        depth_o3d,
        depth_scale=1.0 / depth_scale,
        depth_trunc=1.0,
        convert_rgb_to_intensity=False,
    )

    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intr)
    pcd_raw = copy.deepcopy(pcd) if return_raw else None

    if apply_flip:
        pcd.transform(T_FLIP)

    if return_raw:
        return pcd, pcd_raw
    return pcd


def icp_to_cam0(source_pcd, target_pcd, init_trans, source_cam_index, voxel_size=0.005):
    radius = voxel_size * 2.0

    source_pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=30))
    target_pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=30))

    max_corr_coarse = voxel_size * 10.0
    max_corr_fine = voxel_size * 1.0

    icp_coarse = o3d.pipelines.registration.registration_icp(
        source_pcd, target_pcd,
        max_corr_coarse, init_trans,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
    )

    icp_fine = o3d.pipelines.registration.registration_icp(
        source_pcd, target_pcd,
        max_corr_fine, icp_coarse.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
    )

    print(f"[ICP] Cam{source_cam_index} -> Cam0 | fitness: {icp_fine.fitness:.6f}  rmse: {icp_fine.inlier_rmse:.6f}")
    return icp_fine.transformation


def get_color_intrinsics_struct(profile):
    color_stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
    return color_stream.get_intrinsics()


# =========================================================
# MediaPipe
# =========================================================
mp_face_mesh = mp.solutions.face_mesh
FACE_MESH = mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

mp_drawing = mp_solutions.drawing_utils
mp_drawing_styles = mp_solutions.drawing_styles

LIP_UPPER_ID = 0
LIP_LOWER_ID = 17
LIP_LEFT_ID = 61
LIP_RIGHT_ID = 291


def detect_lip_3d_for_camera(color_frame, depth_frame, profile, T_cam_to_cam0, cam_index):
    color_image = np.asanyarray(color_frame.get_data())  # BGR
    h, w, _ = color_image.shape
    rgb_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)

    results = FACE_MESH.process(rgb_image)
    if not results.multi_face_landmarks:
        return {"ok": False, "camera_index": cam_index}

    face_landmarks = results.multi_face_landmarks[0]

    annotated_image = color_image.copy()
    mp_drawing.draw_landmarks(
        image=annotated_image,
        landmark_list=face_landmarks,
        connections=mp_face_mesh.FACEMESH_LIPS,
        landmark_drawing_spec=None,
        connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style(),
    )

    def lm_to_pixel(lm_id):
        lm = face_landmarks.landmark[lm_id]
        u = int(round(lm.x * w))
        v = int(round(lm.y * h))
        if u < 0 or u >= w or v < 0 or v >= h:
            return None
        return (u, v)

    pix_upper = lm_to_pixel(LIP_UPPER_ID)
    pix_lower = lm_to_pixel(LIP_LOWER_ID)
    pix_left = lm_to_pixel(LIP_LEFT_ID)
    pix_right = lm_to_pixel(LIP_RIGHT_ID)

    if any(p is None for p in [pix_upper, pix_lower, pix_left, pix_right]):
        return {"ok": False, "camera_index": cam_index}

    intr = get_color_intrinsics_struct(profile)

    def pixel_to_cam0(pix):
        u, v = pix
        z_m = depth_frame.get_distance(u, v)
        if z_m <= 0:
            return None
        X, Y, Z = rs.rs2_deproject_pixel_to_point(intr, [u, v], z_m)
        p_cam = np.array([X, -Y, Z, 1.0], dtype=np.float64)  # Y反転
        p0 = T_cam_to_cam0 @ p_cam
        return p0[:3]

    p_upper = pixel_to_cam0(pix_upper)
    p_lower = pixel_to_cam0(pix_lower)
    p_left = pixel_to_cam0(pix_left)
    p_right = pixel_to_cam0(pix_right)

    if any(p is None for p in [p_upper, p_lower, p_left, p_right]):
        return {"ok": False, "camera_index": cam_index}

    points_cam0 = {"upper": p_upper, "lower": p_lower, "left": p_left, "right": p_right}

    return {
        "ok": True,
        "camera_index": cam_index,
        "points_cam0": points_cam0,
        "annotated_image": annotated_image,
        "face_landmarks": face_landmarks,
    }


def build_outer_lip_polygon(face_landmarks, w, h):
    edges = mp.solutions.face_mesh.FACEMESH_LIPS
    adj = defaultdict(set)
    for a, b in edges:
        adj[a].add(b)
        adj[b].add(a)

    visited = set()
    polys = []

    for start in adj.keys():
        if start in visited:
            continue

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

        s = comp[0]
        nbs = list(adj[s])
        if len(nbs) < 1:
            continue

        ordered = [s]
        prev = None
        cur = s
        nxt = nbs[0]

        for _ in range(len(comp) + 5):
            ordered.append(nxt)
            prev, cur = cur, nxt
            cand = [x for x in adj[cur] if x != prev]
            if not cand:
                break
            nxt = cand[0]
            if nxt == s:
                break

        if ordered[-1] == s:
            ordered = ordered[:-1]

        poly = []
        for idx in ordered:
            lm = face_landmarks.landmark[idx]
            u = int(round(lm.x * w))
            v = int(round(lm.y * h))
            poly.append([u, v])
        poly = np.array(poly, dtype=np.int32)

        if len(poly) >= 3 and abs(cv2.contourArea(poly.reshape(-1, 1, 2))) > 1.0:
            polys.append(poly)

    if not polys:
        return None

    areas = [abs(cv2.contourArea(p.reshape(-1, 1, 2))) for p in polys]
    return polys[int(np.argmax(areas))]


def crop_pcd_by_lip_polygon_project(
    merged_pcd,
    lip_poly_px,
    color_intrinsics,
    T_cam_to_cam0,
    mask_dilate_px=0,
    debug_bgr=None,
    debug_save_path=None,
    front_band_m=0.003,
):
    if lip_poly_px is None or len(lip_poly_px) < 3:
        return None

    w = color_intrinsics.width
    h = color_intrinsics.height

    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [lip_poly_px.reshape(-1, 1, 2)], 255)
    if mask_dilate_px > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * mask_dilate_px + 1, 2 * mask_dilate_px + 1))
        mask = cv2.dilate(mask, k)

    # cam0 -> cam
    T_cam0_to_cam = np.linalg.inv(T_cam_to_cam0)

    pts0 = np.asarray(merged_pcd.points)
    if pts0.size == 0:
        return None

    pts0_h = np.hstack([pts0, np.ones((pts0.shape[0], 1), dtype=np.float64)])
    pts_cam = (T_cam0_to_cam @ pts0_h.T).T[:, :3]

    X = pts_cam[:, 0]
    Y = pts_cam[:, 1]
    Z = pts_cam[:, 2]

    valid = Z > 1e-6
    X = X[valid]
    Y = Y[valid]
    Z = Z[valid]
    valid_idx = np.where(valid)[0]

    fx, fy, cx, cy = color_intrinsics.fx, color_intrinsics.fy, color_intrinsics.ppx, color_intrinsics.ppy

    u = (fx * (X / Z) + cx).astype(np.int32)
    v = (cy - fy * (Y / Z)).astype(np.int32)  # Y反転座標系

    in_img = (u >= 0) & (u < w) & (v >= 0) & (v < h)
    u = u[in_img]
    v = v[in_img]
    idx = valid_idx[in_img]
    Z_img = Z[in_img]

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

    is_front = z_in <= (minz[inv] + float(front_band_m))

    keep_idx = idx_in[is_front].tolist()
    if not keep_idx:
        return None

    if debug_bgr is not None and debug_save_path is not None:
        vis = cv2.bitwise_and(debug_bgr, debug_bgr, mask=mask)
        cv2.polylines(vis, [lip_poly_px.reshape(-1, 1, 2)], isClosed=True, color=(0, 255, 255), thickness=2)
        os.makedirs(os.path.dirname(debug_save_path), exist_ok=True)
        cv2.imwrite(debug_save_path, vis)

    return merged_pcd.select_by_index(keep_idx)


def keep_largest_cluster_dbscan(pcd, eps=0.006, min_points=30):
    if pcd is None or len(pcd.points) == 0:
        return pcd

    labels = np.asarray(pcd.cluster_dbscan(eps=eps, min_points=min_points, print_progress=False))
    if labels.size == 0:
        return pcd

    valid = labels >= 0
    if not np.any(valid):
        return pcd

    counts = np.bincount(labels[valid])
    largest = int(np.argmax(counts))
    keep_idx = np.where(labels == largest)[0].tolist()
    return pcd.select_by_index(keep_idx)


def median_depth_in_polygon(depth_frame, depth_scale, lip_poly_px):
    if lip_poly_px is None or len(lip_poly_px) < 3:
        return math.inf

    depth_image = np.asanyarray(depth_frame.get_data())  # uint16
    h, w = depth_image.shape

    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [lip_poly_px.reshape(-1, 1, 2)], 255)

    vals = depth_image[mask > 0]
    vals = vals[vals > 0]
    if vals.size == 0:
        return math.inf

    return float(np.median(vals) * float(depth_scale))


# =========================================================
# Capture + process (No AR)
# =========================================================
SHOW_CAM0_WINDOW = True
MIN_LIP_POINTS = 300  # cam0が「映っている」と判断する最低点数


def capture_and_process_3cams(pipelines, profiles, capture_id: str):
    """ARマーカー無し。capture_id は時間ラベル（例: YYYYmmdd_HHMMSS）。"""

    color_frames = [None] * len(pipelines)
    depth_frames = [None] * len(pipelines)
    aligns = [rs.align(rs.stream.color) for _ in pipelines]

    def grab_one(i):
        return pipelines[i].wait_for_frames()

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

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # ===== 点群生成（Y反転済み）+ raw保存 =====
    pcds = []
    raw_pcds = []
    for i in range(len(SERIALS)):
        pcd, pcd_raw = frames_to_pointcloud(
            color_frames[i], depth_frames[i], profiles[i], apply_flip=True, return_raw=True
        )
        pcds.append(pcd)
        raw_pcds.append(pcd_raw)

    os.makedirs("PLY/pre/raw_face", exist_ok=True)
    for i, pcd_raw in enumerate(raw_pcds):
        raw_path = f"PLY/pre/raw_face/face_cam{i}_raw_{capture_id}_{timestamp}.ply"
        o3d.io.write_point_cloud(raw_path, pcd_raw)
        print(f"[SAVE] {raw_path}")

    # ===== ICPでcam1/cam2をcam0へ =====
    base_pcd = pcds[0]
    T_1_to_0_icp = icp_to_cam0(pcds[1], base_pcd, T_1_to_0, source_cam_index=1)
    T_2_to_0_icp = icp_to_cam0(pcds[2], base_pcd, T_2_to_0, source_cam_index=2)

    pcd0_aligned = copy.deepcopy(base_pcd)
    pcd1_aligned = copy.deepcopy(pcds[1])
    pcd1_aligned.transform(T_1_to_0_icp)
    pcd2_aligned = copy.deepcopy(pcds[2])
    pcd2_aligned.transform(T_2_to_0_icp)

    merged_pcd = o3d.geometry.PointCloud()
    merged_pcd += pcd0_aligned
    merged_pcd += pcd1_aligned
    merged_pcd += pcd2_aligned

    os.makedirs("PLY/pre", exist_ok=True)
    merged_path = f"PLY/pre/face_3cams_geom_merged_{capture_id}_{timestamp}.ply"
    o3d.io.write_point_cloud(merged_path, merged_pcd)
    print(f"[SAVE] {merged_path}")

    # ===== MediaPipe（各カメラ）: 4点+外周ポリゴン+デバッグ画像 =====
    lip_results = []
    for cam_idx in range(len(SERIALS)):
        if cam_idx == 0:
            T_cam_to_0 = np.eye(4, dtype=np.float64)
        elif cam_idx == 1:
            T_cam_to_0 = T_1_to_0_icp
        else:
            T_cam_to_0 = T_2_to_0_icp

        res = detect_lip_3d_for_camera(
            color_frames[cam_idx], depth_frames[cam_idx], profiles[cam_idx], T_cam_to_0, cam_index=cam_idx
        )
        lip_results.append(res)

    # ===== カメラ選択ルール =====
    # 基本はCam0。
    # Cam0の唇点群が不足する場合のみ、各カメラの唇領域の「深度中央値」が最小（最も近い）なカメラを採用。

    # まず各カメラで外周ポリゴンを作り、口点群を抽出して点数を確認
    per_cam = []
    for cam_idx in range(len(SERIALS)):
        res = lip_results[cam_idx]
        if not res.get("ok"):
            per_cam.append({"ok": False, "camera_index": cam_idx})
            continue

        face_landmarks = res["face_landmarks"]
        bgr = np.asanyarray(color_frames[cam_idx].get_data()).copy()
        h, w, _ = bgr.shape
        lip_poly = build_outer_lip_polygon(face_landmarks, w, h)
        if lip_poly is None:
            per_cam.append({"ok": False, "camera_index": cam_idx})
            continue

        if cam_idx == 0:
            T_cam_to_cam0 = np.eye(4, dtype=np.float64)
        elif cam_idx == 1:
            T_cam_to_cam0 = T_1_to_0_icp
        else:
            T_cam_to_cam0 = T_2_to_0_icp

        color_intr = color_frames[cam_idx].profile.as_video_stream_profile().get_intrinsics()

        debug_path = f"PLY/pre/lip_mask_debug/lipmask_cam{cam_idx}_{capture_id}_{timestamp}.png"

        mouth_pcd = crop_pcd_by_lip_polygon_project(
            merged_pcd=merged_pcd,
            lip_poly_px=lip_poly,
            color_intrinsics=color_intr,
            T_cam_to_cam0=T_cam_to_cam0,
            mask_dilate_px=0,
            debug_bgr=bgr,
            debug_save_path=debug_path,
            front_band_m=0.003,
        )

        # 深度中央値（このカメラの aligned depth から計算）
        depth_scale = profiles[cam_idx].get_device().first_depth_sensor().get_depth_scale()
        med_depth = median_depth_in_polygon(depth_frames[cam_idx], depth_scale, lip_poly)

        per_cam.append({
            "ok": True,
            "camera_index": cam_idx,
            "lip_poly": lip_poly,
            "mouth_pcd": mouth_pcd,
            "median_depth": med_depth,
            "annotated_image": res.get("annotated_image"),
            "points_cam0": res.get("points_cam0"),
        })

    selected = None
    cam0_info = per_cam[0] if len(per_cam) > 0 else None
    if cam0_info and cam0_info.get("ok") and cam0_info.get("mouth_pcd") is not None and len(cam0_info["mouth_pcd"].points) >= MIN_LIP_POINTS:
        selected = cam0_info
    else:
        # cam0不足 → 深度中央値が最小のカメラ
        candidates = [c for c in per_cam if c.get("ok") and c.get("mouth_pcd") is not None and len(c["mouth_pcd"].points) > 0]
        if candidates:
            candidates.sort(key=lambda x: x.get("median_depth", math.inf))
            selected = candidates[0]

    if selected is None:
        print("[LIP] 口領域点群の抽出に失敗しました。")
        return {"pred_label": None}

    sel_cam = int(selected["camera_index"])
    mouth_pcd_cam0 = selected["mouth_pcd"]

    # 口点群の大きなクラスタだけ残す（背景残り対策：あなたがBを選んだ処理）
    mouth_pcd_cam0 = keep_largest_cluster_dbscan(mouth_pcd_cam0, eps=0.006, min_points=30)

    # ===== 唇中心 & 口ローカル軸で整列（Tag無しでも可能） =====
    pts4 = selected.get("points_cam0")
    if pts4 is None:
        # 念のため：4点が無い場合は重心中心化のみ
        lip_center_cam0 = np.mean(np.asarray(mouth_pcd_cam0.points), axis=0)
        R_mouth = np.eye(3, dtype=np.float64)
    else:
        lip_center_cam0 = (pts4["upper"] + pts4["lower"] + pts4["left"] + pts4["right"]) / 4.0

        x_axis = pts4["right"] - pts4["left"]
        y_axis = pts4["upper"] - pts4["lower"]
        x_n = np.linalg.norm(x_axis)
        y_n = np.linalg.norm(y_axis)
        if x_n < 1e-9 or y_n < 1e-9:
            R_mouth = np.eye(3, dtype=np.float64)
        else:
            x_axis = x_axis / x_n
            y_axis = y_axis / y_n
            z_axis = np.cross(x_axis, y_axis)
            z_n = np.linalg.norm(z_axis)
            if z_n < 1e-9:
                R_mouth = np.eye(3, dtype=np.float64)
            else:
                z_axis = z_axis / z_n
                # 再直交化
                y_axis = np.cross(z_axis, x_axis)
                y_axis = y_axis / (np.linalg.norm(y_axis) + 1e-12)
                R_mouth = np.column_stack([x_axis, y_axis, z_axis])  # cam0 -> mouth(local)

    pts = np.asarray(mouth_pcd_cam0.points, dtype=np.float64)
    pts_centered = pts - lip_center_cam0.reshape(1, 3)
    pts_local = (R_mouth.T @ pts_centered.T).T

    mouth_pcd_local = o3d.geometry.PointCloud()
    mouth_pcd_local.points = o3d.utility.Vector3dVector(pts_local)

    # ===== 保存（口PLY + MediaPipe描画画像） =====
    os.makedirs("PLY/pre/mouth", exist_ok=True)
    mouth_ply = f"PLY/pre/mouth/mouth_{capture_id}_{timestamp}.ply"
    o3d.io.write_point_cloud(mouth_ply, mouth_pcd_local)
    print(f"[SAVE] {mouth_ply}  (cam={sel_cam}, pts={len(mouth_pcd_local.points)})")

    os.makedirs("PLY/pre/mediapipe", exist_ok=True)
    ann = selected.get("annotated_image")
    if ann is not None:
        mp_path = f"PLY/pre/mediapipe/mediapipe_cam{sel_cam}_{capture_id}_{timestamp}.png"
        cv2.imwrite(mp_path, ann)
        print(f"[SAVE] {mp_path}")

    # ===== 推論 =====
    pred_label, pred_value, detail = predict_mouth_label_from_pcd(mouth_pcd_local)

    os.makedirs("PLY/pre/pred", exist_ok=True)
    pred_txt = f"PLY/pre/pred/pred_{capture_id}_{timestamp}.txt"
    with open(pred_txt, "w", encoding="utf-8") as pf:
        pf.write(f"capture_id: {capture_id}\n")
        pf.write(f"selected_camera_index: {sel_cam}\n")
        pf.write(f"pred_label: {pred_label}\n")

        if detail is not None:
            pf.write(f"pred_percent: {pred_value:.2f}\n")
            pf.write("class_percent:\n")
            for k, v in detail.items():
                pf.write(f"  {k}: {v:.2f}\n")
        else:
            if pred_value is not None:
                pf.write(f"decision_score: {pred_value:.6f}\n")

    print(f"[PRED] {pred_label}  (saved: {pred_txt})")

    return {"pred_label": pred_label, "pred_value": pred_value, "detail": detail}


# =========================================================
# main (No AR)
# =========================================================

def main():
    global SVM_PAYLOAD

    # SVMロード
    SVM_PAYLOAD = joblib.load(SVM_MODEL_PATH)

    pipelines = []
    profiles = []

    try:
        for s in SERIALS:
            p, prof = create_pipeline(s)
            pipelines.append(p)
            profiles.append(prof)

        print("[INFO] Running...  Stop with Ctrl+C (KeyboardInterrupt).")

        is_processing = False
        last_pred_text = "PRED: --"
        last_pred_time = 0.0
        PRED_SHOW_SEC = 3.0

        align0 = rs.align(rs.stream.color)

        while True:
            # Cam0の表示用フレーム
            fs0 = pipelines[0].wait_for_frames()
            fs0 = align0.process(fs0)
            color0 = fs0.get_color_frame()
            if not color0:
                continue

            frame_vis = np.asanyarray(color0.get_data()).copy()

            capture_ready = not is_processing
            status = "READY" if capture_ready else "NG"

            if SHOW_CAM0_WINDOW:
                cv2.putText(
                    frame_vis,
                    f"CAPTURE: {status} (press 'c')",
                    (30, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0) if capture_ready else (0, 0, 255),
                    2,
                )

                if is_processing:
                    cv2.putText(
                        frame_vis,
                        "PROCESSING... DO NOT MOVE",
                        (30, 110),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 0, 255),
                        2,
                    )

                now = time.time()
                if now - last_pred_time <= PRED_SHOW_SEC:
                    cv2.putText(
                        frame_vis,
                        last_pred_text,
                        (30, 150),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.9,
                        (0, 255, 255),
                        2,
                    )

                cv2.imshow("Cam0 (No AR)", frame_vis)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

            if key == ord('c'):
                if capture_ready:
                    is_processing = True

                    overlay = frame_vis.copy()
                    cv2.putText(
                        overlay,
                        "CAPTURED. PROCESSING... DO NOT MOVE",
                        (30, 110),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 0, 255),
                        2,
                    )
                    cv2.imshow("Cam0 (No AR)", overlay)
                    cv2.waitKey(1)

                    capture_id = datetime.now().strftime("%Y%m%d_%H%M%S")  # pitch_label_deg の代わりに時間
                    print(f"[TRIGGER] manual capture | id={capture_id}")

                    result = capture_and_process_3cams(pipelines, profiles, capture_id=capture_id)

                    if result and result.get("pred_label") is not None:
                        pv = result.get("pred_value")
                        detail = result.get("detail")
                        if detail is not None and pv is not None:
                            last_pred_text = f"PRED: {result['pred_label']}  {pv:.1f}%"
                        elif pv is not None:
                            last_pred_text = f"PRED: {result['pred_label']}  score={pv:.3f}"
                        else:
                            last_pred_text = f"PRED: {result['pred_label']}"
                        last_pred_time = time.time()

                    is_processing = False

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

