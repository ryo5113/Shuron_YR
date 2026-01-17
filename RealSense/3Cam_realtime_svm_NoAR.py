import pyrealsense2 as rs
import numpy as np
import open3d as o3d
from datetime import datetime
import cv2
import math
import joblib
import os
import copy
from concurrent.futures import ThreadPoolExecutor
import time

# === MediaPipe ===
import mediapipe as mp

# =========================================================
# SVM推論（faceTrain_SVM.py の保存形式を読む想定）
# =========================================================
SVM_MODEL_PATH = r"./PLY_dataset_all/ply_svm_model.joblib"  # あなたのモデルパスに合わせる
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
        return None, None, None

    pts = np.asarray(mouth_pcd.points, dtype=np.float64)
    grid = int(SVM_PAYLOAD.get("grid", 10))
    feat = occupancy_grid_features(pts, grid=grid).reshape(1, -1)

    model = SVM_PAYLOAD["model"]
    label_order = SVM_PAYLOAD["label_order"]

    # --- predict_proba が使える場合（％表示）---
    if hasattr(model, "predict_proba"):
        try:
            proba = model.predict_proba(feat)[0]
            pred_idx = int(np.argmax(proba))
            pred_label = str(label_order[pred_idx])
            pred_percent = float(proba[pred_idx] * 100.0)
            percent_dict = {str(label_order[i]): float(proba[i] * 100.0) for i in range(len(label_order))}
            return pred_label, pred_percent, percent_dict
        except Exception:
            pass

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
# RealSense / Open3D 点群作成
# =========================================================
T_FLIP = np.eye(4, dtype=np.float64)
T_FLIP[1, 1] = -1.0  # Y反転

def frames_to_pointcloud(color_frame, depth_frame, profile, apply_flip=True, return_raw=False):
    """
    align済の color/depth から Open3D PointCloud を作成
    apply_flip=True なら Y反転して「あなたの既存処理と同じ座標系」に揃える
    """
    depth_image = np.asanyarray(depth_frame.get_data())
    color_image = np.asanyarray(color_frame.get_data())

    depth_intrinsics = profile.get_stream(rs.stream.depth).as_video_stream_profile().get_intrinsics()
    color_o3d = o3d.geometry.Image(cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB))
    depth_o3d = o3d.geometry.Image(depth_image)

    # depth scale
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale_rs = float(depth_sensor.get_depth_scale())

    width, height = depth_intrinsics.width, depth_intrinsics.height
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

def get_color_intrinsics_struct(profile):
    color_stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
    return color_stream.get_intrinsics()

def get_depth_scale_from_profile(profile) -> float:
    depth_sensor = profile.get_device().first_depth_sensor()
    return float(depth_sensor.get_depth_scale())

# =========================================================
# MediaPipe (FaceMesh lips)
# =========================================================
mp_face_mesh = mp.solutions.face_mesh
FACE_MESH = mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

from mediapipe import solutions as mp_solutions
mp_drawing = mp_solutions.drawing_utils
mp_drawing_styles = mp_solutions.drawing_styles

LIP_UPPER_ID = 0
LIP_LOWER_ID = 17
LIP_LEFT_ID  = 61
LIP_RIGHT_ID = 291

from collections import defaultdict, deque

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
    outer = polys[int(np.argmax(areas))]
    return outer

def compute_mean_depth_in_polygon(depth_frame, depth_scale_m, lip_poly_px, w, h):
    if lip_poly_px is None or len(lip_poly_px) < 3:
        return float("nan")

    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [lip_poly_px.reshape(-1, 1, 2)], 255)

    depth_u16 = np.asanyarray(depth_frame.get_data()).astype(np.float32)
    depth_m = depth_u16 * float(depth_scale_m)

    m = (mask > 0) & (depth_m > 0)
    if not np.any(m):
        return float("nan")

    return float(np.mean(depth_m[m]))

def detect_lip_3d_for_camera(color_frame, depth_frame, profile, T_cam_to_cam0, cam_index):
    """
    - MediaPipeで唇4点+外周ポリゴンを取得
    - depthで4点を3D化し、Y反転→T_cam_to_cam0でcam0座標へ
    - 外周ポリゴン内の平均深度(mean_depth_m)も計算（カメラ選択用）
    """
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
        connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style()
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
    pix_left  = lm_to_pixel(LIP_LEFT_ID)
    pix_right = lm_to_pixel(LIP_RIGHT_ID)

    def draw_id_point(img, pix, color, r=4):
        if pix is None:
            return
        u, v = pix
        cv2.circle(img, (u, v), r, color, -1)

    draw_id_point(annotated_image, pix_upper, (0, 255, 255))
    draw_id_point(annotated_image, pix_lower, (255, 255, 0))
    draw_id_point(annotated_image, pix_left,  (0, 255, 0))
    draw_id_point(annotated_image, pix_right, (0, 0, 255))

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
    p_left  = pixel_to_cam0(pix_left)
    p_right = pixel_to_cam0(pix_right)

    if any(p is None for p in [p_upper, p_lower, p_left, p_right]):
        return {"ok": False, "camera_index": cam_index}

    points_cam0 = {"upper": p_upper, "lower": p_lower, "left": p_left, "right": p_right}

    lip_poly = build_outer_lip_polygon(face_landmarks, w, h)
    depth_scale_m = get_depth_scale_from_profile(profile)
    mean_depth_m = compute_mean_depth_in_polygon(depth_frame, depth_scale_m, lip_poly, w, h)

    return {
        "ok": True,
        "camera_index": cam_index,
        "points_cam0": points_cam0,
        "annotated_image": annotated_image,
        "face_landmarks": face_landmarks,
        "lip_poly_px": lip_poly,
        "mean_depth_m": mean_depth_m,
        "img_wh": (w, h),
    }

def crop_pcd_by_lip_polygon_project(
    merged_pcd,
    lip_poly_px,
    color_intrinsics,
    T_cam_to_cam0,
    depth_frame=None,
    depth_tol_m=0.01,
    mask_dilate_px=0,
    debug_bgr=None,
    debug_save_path=None
):
    if lip_poly_px is None or len(lip_poly_px) < 3:
        return None

    w = color_intrinsics.width
    h = color_intrinsics.height

    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [lip_poly_px.reshape(-1, 1, 2)], 255)

    if mask_dilate_px > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*mask_dilate_px+1, 2*mask_dilate_px+1))
        mask = cv2.dilate(mask, k)

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
    X = X[valid]; Y = Y[valid]; Z = Z[valid]
    valid_idx = np.where(valid)[0]

    fx, fy, cx, cy = color_intrinsics.fx, color_intrinsics.fy, color_intrinsics.ppx, color_intrinsics.ppy

    u = (fx * (X / Z) + cx).astype(np.int32)
    v = (cy - fy * (Y / Z)).astype(np.int32)  # Y反転座標系

    in_img = (u >= 0) & (u < w) & (v >= 0) & (v < h)
    u = u[in_img]; v = v[in_img]
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

    # 1ピクセルに複数点が来る場合、最前面帯だけ残す（背景混入を弱める）
    pix_key = (v_in.astype(np.int64) * w + u_in.astype(np.int64))
    uniq, inv = np.unique(pix_key, return_inverse=True)

    minz = np.full((uniq.shape[0],), np.inf, dtype=np.float64)
    np.minimum.at(minz, inv, z_in)

    front_band_m = 0.003  # 3mm帯
    is_front = z_in <= (minz[inv] + front_band_m)

    keep = is_front

    # depth_frame とも整合を取る場合はここをON（必要時のみ）
    if depth_frame is not None:
        d_in = np.array([depth_frame.get_distance(int(uu), int(vv)) for uu, vv in zip(u_in, v_in)], dtype=np.float64)
        ok_depth = (d_in > 0) & (np.abs(z_in - d_in) <= depth_tol_m)
        keep = keep & ok_depth

    keep_idx = idx_in[keep].tolist()
    if not keep_idx:
        return None

    if debug_bgr is not None and debug_save_path is not None:
        vis = cv2.bitwise_and(debug_bgr, debug_bgr, mask=mask)
        cv2.polylines(vis, [lip_poly_px.reshape(-1, 1, 2)], isClosed=True, color=(0, 255, 255), thickness=2)
        os.makedirs(os.path.dirname(debug_save_path), exist_ok=True)
        cv2.imwrite(debug_save_path, vis)

    return merged_pcd.select_by_index(keep_idx)

def build_mouth_axes_from_4pts(points_cam0: dict):
    up = points_cam0["upper"]
    lo = points_cam0["lower"]
    le = points_cam0["left"]
    ri = points_cam0["right"]

    x = (ri - le)
    nx = np.linalg.norm(x)
    if nx < 1e-9:
        return None
    x = x / nx

    y0 = (up - lo)
    # y0 から x 成分を除去して直交化
    y = y0 - np.dot(y0, x) * x
    ny = np.linalg.norm(y)
    if ny < 1e-9:
        return None
    y = y / ny

    z = np.cross(x, y)
    nz = np.linalg.norm(z)
    if nz < 1e-9:
        return None
    z = z / nz

    # 再直交化（数値誤差対策）
    y = np.cross(z, x)
    y = y / (np.linalg.norm(y) + 1e-12)

    R = np.stack([x, y, z], axis=1)  # columns = axes
    return R

# =========================================================
# 実行オプション
# =========================================================
SAVE_ONLY_PLY = True
SHOW_CAM0_WINDOW = True

NUM_FRAMES = 10  # フレーム安定化用
MAX_RANGE_M = 1.0  # 1m以内点群だけ
MIN_MOUTH_POINTS_FOR_CAM0 = 800  # cam0が「欠ける」の判定に使う最低点数

def capture_and_process_3cams(pipelines, profiles, capture_id_str: str):
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
    os.makedirs("PLY/pre/raw_face", exist_ok=True)

    # --- 各カメラの点群（cam座標, Y反転済）を作成 ---
    pcds = []
    pcds_raw = []
    for i in range(len(pipelines)):
        pcd_i, pcd_raw_i = frames_to_pointcloud(color_frames[i], depth_frames[i], profiles[i], apply_flip=True, return_raw=True)
        # 1m以内に制限
        pts = np.asarray(pcd_i.points)
        if pts.size > 0:
            m = (pts[:, 2] > 1e-6) & (pts[:, 2] < MAX_RANGE_M)
            pcd_i = pcd_i.select_by_index(np.where(m)[0])
            pcd_raw_i = pcd_raw_i.select_by_index(np.where(m)[0])
        pcds.append(pcd_i)
        pcds_raw.append(pcd_raw_i)

        raw_path = f"PLY/pre/raw_face/face_cam{i}_raw_{capture_id_str}_{timestamp}.ply"
        o3d.io.write_point_cloud(raw_path, pcd_raw_i)
        print(f"[SAVE] {raw_path}")

    # --- ICPで Cam1/Cam2 -> Cam0 ---
    T_cam0_to_cam0 = np.eye(4, dtype=np.float64)

    init = np.eye(4, dtype=np.float64)
    T_cam1_to_cam0 = icp_to_cam0(pcds[1], pcds[0], init, source_cam_index=1) if len(pipelines) > 1 else np.eye(4)
    T_cam2_to_cam0 = icp_to_cam0(pcds[2], pcds[0], init, source_cam_index=2) if len(pipelines) > 2 else np.eye(4)

    T_cam_to_cam0_list = [T_cam0_to_cam0, T_cam1_to_cam0, T_cam2_to_cam0]

    # --- 統合点群（cam0座標系） ---
    merged_pcd = copy.deepcopy(pcds[0])
    merged_pcd_camcolor = copy.deepcopy(pcds[0])

    for i in range(1, len(pipelines)):
        p = copy.deepcopy(pcds[i])
        p.transform(T_cam_to_cam0_list[i])
        merged_pcd += p

        pc = copy.deepcopy(pcds[i])
        pc.transform(T_cam_to_cam0_list[i])
        merged_pcd_camcolor += pc

    # --- MediaPipeを3台に適用して候補を作る ---
    lip_results = []
    for i in range(len(pipelines)):
        res = detect_lip_3d_for_camera(
            color_frames[i],
            depth_frames[i],
            profiles[i],
            T_cam_to_cam0_list[i],
            cam_index=i
        )
        lip_results.append(res)

    # =========================================================
    # カメラ選択ルール（ARなし版）
    #   1) 原則 cam0
    #   2) cam0 が欠ける（口点群が少ない等）なら、平均深度が最小（最も近い）カメラを採用
    # =========================================================
    selected = None

    # cam0が使えるならまず採用
    if lip_results[0].get("ok", False):
        selected0 = lip_results[0]
        intr0 = get_color_intrinsics_struct(profiles[0])
        mouth0 = crop_pcd_by_lip_polygon_project(
            merged_pcd=merged_pcd,
            lip_poly_px=selected0.get("lip_poly_px", None),
            color_intrinsics=intr0,
            T_cam_to_cam0=T_cam_to_cam0_list[0],
            depth_frame=depth_frames[0],
            depth_tol_m=0.01,
            mask_dilate_px=0,
        )
        if mouth0 is not None and len(mouth0.points) >= MIN_MOUTH_POINTS_FOR_CAM0:
            selected = selected0

    if selected is None:
        ok_cands = [r for r in lip_results if r.get("ok", False)]
        ok_cands = [r for r in ok_cands if np.isfinite(r.get("mean_depth_m", float("nan")))]
        if ok_cands:
            selected = min(ok_cands, key=lambda r: r["mean_depth_m"])

    if selected is None:
        print("[LIP] No valid MediaPipe detection on any camera. abort.")
        return None

    sel_cam = int(selected["camera_index"])
    print(f"[LIP] selected camera: cam{sel_cam} (mean_depth={selected.get('mean_depth_m')})")

    # --- 選択カメラのポリゴンで統合点群を切り出し ---
    intr_sel = get_color_intrinsics_struct(profiles[sel_cam])
    mouth_pcd = crop_pcd_by_lip_polygon_project(
        merged_pcd=merged_pcd,
        lip_poly_px=selected.get("lip_poly_px", None),
        color_intrinsics=intr_sel,
        T_cam_to_cam0=T_cam_to_cam0_list[sel_cam],
        depth_frame=depth_frames[sel_cam],
        depth_tol_m=0.01,
        mask_dilate_px=0,
        debug_bgr=np.asanyarray(color_frames[sel_cam].get_data()),
        debug_save_path=None
    )

    mouth_pcd_camcolor = crop_pcd_by_lip_polygon_project(
        merged_pcd=merged_pcd_camcolor,
        lip_poly_px=selected.get("lip_poly_px", None),
        color_intrinsics=intr_sel,
        T_cam_to_cam0=T_cam_to_cam0_list[sel_cam],
        depth_frame=depth_frames[sel_cam],
        depth_tol_m=0.01,
        mask_dilate_px=0
    )

    if mouth_pcd is None or len(mouth_pcd.points) == 0:
        print("[LIP] mouth_pcd is empty (polygon crop).")
        return None

    # --- 中心化（唇4点平均：cam0座標系）---
    pts4 = selected["points_cam0"]
    lip_center_cam0 = (pts4["upper"] + pts4["lower"] + pts4["left"] + pts4["right"]) / 4.0

    pts = np.asarray(mouth_pcd.points, dtype=np.float64)
    pts_c = pts - lip_center_cam0

    pts_camcolor = np.asarray(mouth_pcd_camcolor.points, dtype=np.float64) if mouth_pcd_camcolor is not None else pts
    pts_c2 = pts_camcolor - lip_center_cam0

    # --- 口局所座標へ回転（4点から口軸を推定）---
    R_mouth = build_mouth_axes_from_4pts(pts4)
    if R_mouth is not None:
        pts_local = (R_mouth.T @ pts_c.T).T
        pts_local2 = (R_mouth.T @ pts_c2.T).T
    else:
        pts_local = pts_c
        pts_local2 = pts_c2

    mouth_pcd_local = o3d.geometry.PointCloud()
    mouth_pcd_local.points = o3d.utility.Vector3dVector(pts_local)
    if mouth_pcd.has_colors():
        mouth_pcd_local.colors = mouth_pcd.colors

    mouth_pcd_camcolor_local = o3d.geometry.PointCloud()
    mouth_pcd_camcolor_local.points = o3d.utility.Vector3dVector(pts_local2)
    if mouth_pcd_camcolor is not None and mouth_pcd_camcolor.has_colors():
        mouth_pcd_camcolor_local.colors = mouth_pcd_camcolor.colors

    # --- 推論 ---
    pred_label, pred_value, detail = predict_mouth_label_from_pcd(mouth_pcd_local)

    # --- 保存（mouth PLY + mediapipe画像 + pred txt）---
    os.makedirs("PLY/pre/mouth", exist_ok=True)
    os.makedirs("PLY/pre/mediapipe_img", exist_ok=True)
    os.makedirs("PLY/pre/pred", exist_ok=True)

    mouth_filename = f"PLY/pre/mouth/mouth_{capture_id_str}_{timestamp}.ply"
    o3d.io.write_point_cloud(mouth_filename, mouth_pcd_local)
    print(f"[SAVE] mouth pcd: {mouth_filename}")

    annotated = selected.get("annotated_image", None)
    img_path = None
    if annotated is not None:
        img_path = f"PLY/pre/mediapipe_img/lip_cam{sel_cam}_{capture_id_str}_{timestamp}.png"
        cv2.imwrite(img_path, annotated)
        print(f"[SAVE] mediapipe img: {img_path}")

    pred_txt = f"PLY/pre/pred/pred_{capture_id_str}_{timestamp}.txt"
    with open(pred_txt, "w", encoding="utf-8") as pf:
        pf.write(f"capture_id: {capture_id_str}\n")
        pf.write(f"camera_index(lip source): {sel_cam}\n")
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

    return {
        "mouth_ply": mouth_filename,
        "mediapipe_img": img_path,
        "pred_label": pred_label,
        "pred_value": pred_value,
        "detail": detail,
        "selected_cam": sel_cam,
    }

# =========================================================
# Main
# =========================================================
SERIALS = [
    None,  # cam0
    None,  # cam1
    None,  # cam2
]

def start_pipelines():
    pipelines = []
    profiles = []
    for i in range(3):
        pipe = rs.pipeline()
        cfg = rs.config()
        if SERIALS[i]:
            cfg.enable_device(SERIALS[i])
        cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        prof = pipe.start(cfg)
        pipelines.append(pipe)
        profiles.append(prof)
    return pipelines, profiles

def stop_pipelines(pipelines):
    for p in pipelines:
        try:
            p.stop()
        except Exception:
            pass

def main():
    global SVM_PAYLOAD
    SVM_PAYLOAD = joblib.load(SVM_MODEL_PATH)

    pipelines, profiles = start_pipelines()
    print("[INFO] Running...  Stop with Q.  Capture with C.")

    last_pred_text = ""
    is_processing = False

    try:
        while True:
            # 表示用：cam0だけ取得
            fs0 = pipelines[0].wait_for_frames()
            aligned0 = rs.align(rs.stream.color).process(fs0)
            color0 = aligned0.get_color_frame()
            if not color0:
                continue

            frame_vis = np.asanyarray(color0.get_data()).copy()
            cv2.putText(frame_vis, "Press C: capture  |  Q: quit",
                        (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            if is_processing:
                cv2.putText(frame_vis, "PROCESSING... DO NOT MOVE",
                            (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            if last_pred_text:
                cv2.putText(frame_vis, last_pred_text,
                            (20, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

            if SHOW_CAM0_WINDOW:
                cv2.imshow("Cam0 (No AR)", frame_vis)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

            if key == ord('c') and (not is_processing):
                is_processing = True
                # pitch_label_degの代わりに「時刻ID」を使う
                capture_id_str = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

                # 画面にも一瞬出す
                overlay = frame_vis.copy()
                cv2.putText(overlay, "CAPTURED. PROCESSING... DO NOT MOVE",
                            (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                cv2.imshow("Cam0 (No AR)", overlay)
                cv2.waitKey(1)

                result = capture_and_process_3cams(pipelines, profiles, capture_id_str=capture_id_str)
                if result and result.get("pred_label") is not None:
                    if result.get("detail") is not None:
                        last_pred_text = f"PRED: {result['pred_label']}  {result['pred_value']:.1f}%"
                    else:
                        last_pred_text = f"PRED: {result['pred_label']}"

                is_processing = False

    finally:
        stop_pipelines(pipelines)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
