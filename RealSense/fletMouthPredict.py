# 統合GUI：
#  - 学習タブ：収録(A/I/U/E/O) + 学習開始（モデルを親フォルダ直下に保存）
#  - 推論タブ：親フォルダ指定→モデル読込→推論開始（c押下ごとに推論）

from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime
import traceback
import base64
import json

import flet as ft
import cv2
import numpy as np
import pyrealsense2 as rs
import open3d as o3d
import joblib
import trimesh
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
)

import captureMouth as core  # 収録側の処理を流用

LABELS = ["A", "I", "U", "E", "O"]
MODEL_FILENAME = "ply_svm_model.joblib"
META_FILENAME = "meta.json"
CM_FILENAME = "confusion_matrix.png"

# -------------------------
# 共通ユーティリティ
# -------------------------
def safe_subject_name(name: str) -> str:
    bad = ["\\", "/", ":", "*", "?", '"', "<", ">", "|"]
    for b in bad:
        name = name.replace(b, "_")
    return name.strip()

def now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def get_next_index(out_dir: Path, prefix: str) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    max_n = 0
    for p in out_dir.glob(f"{prefix}_*.ply"):
        last = p.stem.rsplit("_", 1)[-1]
        try:
            n = int(last)
            if n > max_n:
                max_n = n
        except ValueError:
            pass
    return max_n + 1

# -------------------------
# 点数カウント特徴量（学習/推論共通）
# -------------------------
def occupancy_grid_features_count(points: np.ndarray, grid: int) -> np.ndarray:
    pts = points.astype(np.float64, copy=True)
    pts -= pts.mean(axis=0, keepdims=True)

    max_abs = np.max(np.abs(pts))
    if max_abs > 0:
        pts /= max_abs

    pts = np.clip(pts, -1.0, 1.0)

    idx = ((pts + 1.0) * 0.5 * grid).astype(np.int64)
    idx = np.clip(idx, 0, grid - 1)

    counts = np.zeros((grid, grid, grid), dtype=np.float64)
    np.add.at(counts, (idx[:, 0], idx[:, 1], idx[:, 2]), 1.0)

    return counts.reshape(-1)

def load_points_from_ply(ply_path: Path) -> np.ndarray:
    geom = trimesh.load(str(ply_path), process=False)

    if hasattr(geom, "vertices") and geom.vertices is not None:
        pts = np.asarray(geom.vertices, dtype=np.float64)
    elif hasattr(geom, "points") and geom.points is not None:
        pts = np.asarray(geom.points, dtype=np.float64)
    else:
        raise ValueError(f"Unsupported PLY content: {ply_path}")

    pts = pts[:, :3]
    pts = pts[np.isfinite(pts).all(axis=1)]
    if len(pts) == 0:
        raise ValueError(f"No valid points in {ply_path}")
    return pts

def collect_dataset_fixed_order(data_root: Path, grid: int, label_order: list[str]):
    if not data_root.exists():
        raise FileNotFoundError(f"DATA_ROOT not found: {data_root}")

    X_list, y_list = [], []
    for lab in label_order:
        lab_dir = data_root / lab
        if not lab_dir.exists():
            raise FileNotFoundError(f"Label dir not found: {lab_dir}")

        for pf in sorted(lab_dir.glob("*.ply")):
            pts = load_points_from_ply(pf)
            feat = occupancy_grid_features_count(pts, grid=grid)
            X_list.append(feat)
            y_list.append(lab)

    if len(X_list) == 0:
        raise ValueError(f"No PLY files found under: {data_root}")

    X = np.vstack(X_list)
    y_str = np.array(y_list, dtype=object)
    return X, y_str

def build_pipeline(kernel: str, class_weight, probability: bool, seed: int) -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("svc", SVC(
            kernel=kernel,
            class_weight=class_weight,
            probability=probability,
            random_state=seed,
        )),
    ])

def save_confusion_matrix_png(path: Path, cm: np.ndarray, label_names: list[str], dpi: int = 200) -> None:
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_names)
    plt.rcParams["font.size"] = 16
    disp.plot(values_format="d", xticks_rotation=45)
    disp.figure_.tight_layout()
    disp.figure_.savefig(path, dpi=dpi)
    plt.close(disp.figure_)

# -------------------------
# 推論（モデルpayload互換）
# -------------------------
def predict_from_mouth_pcd(mouth_pcd: o3d.geometry.PointCloud, payload: dict):
    if payload is None:
        raise RuntimeError("model payload is None")
    if mouth_pcd is None or len(mouth_pcd.points) == 0:
        return None, None, None

    pts = np.asarray(mouth_pcd.points, dtype=np.float64)
    grid = int(payload.get("grid", 20))
    feat = occupancy_grid_features_count(pts, grid=grid).reshape(1, -1)

    model = payload["model"]
    label_order = payload["label_order"]

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(feat)[0]
        pred_idx = int(np.argmax(proba))
        pred_label = str(label_order[pred_idx])
        pred_percent = float(proba[pred_idx] * 100.0)
        percent_dict = {str(label_order[i]): float(proba[i] * 100.0) for i in range(len(label_order))}
        return pred_label, pred_percent, percent_dict

    pred_label = str(model.predict(feat)[0])
    return pred_label, None, None

# -------------------------
# 収録（あなたのfletMouthCapture.py相当：保存あり）
# -------------------------
def capture_and_process_3cams_to_dirs_save(
    pipelines,
    profiles,
    pitch_label_deg: float,
    tag_R,
    tag_t,
    raw_dir: Path,
    mouth_dir: Path,
    mpimg_dir: Path,
    subject_prefix: str,
):
    raw_dir.mkdir(parents=True, exist_ok=True)
    mouth_dir.mkdir(parents=True, exist_ok=True)
    mpimg_dir.mkdir(parents=True, exist_ok=True)

    base = raw_dir.parent.parent.parent # raw_dir = <base>/<subject>/raw_ply/<label>
    all_root = base / "ALL" / "mouth_ply"
    all_root.mkdir(parents=True, exist_ok=True)

    color_frames = [None] * len(pipelines)
    depth_frames = [None] * len(pipelines)
    aligns = [rs.align(rs.stream.color) for _ in pipelines]

    def grab_one(i):
        return pipelines[i].wait_for_frames()

    def make_T_from_Rt(R, t):
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = np.asarray(R, dtype=np.float64)
        T[:3, 3] = np.asarray(t, dtype=np.float64).reshape(3)
        return T

    def transform_xyz(xyz, T):
        xyz = np.asarray(xyz, dtype=np.float64).reshape(3)
        p = np.ones(4, dtype=np.float64)
        p[:3] = xyz
        q = T @ p
        return q[:3]

    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=len(pipelines)) as ex:
        for _ in range(core.NUM_FRAMES):
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

    timestamp = now_stamp()

    pcds = []
    raw_pcds = []
    for i in range(len(core.SERIALS)):
        pcd, pcd_raw = core.frames_to_pointcloud(
            color_frames[i], depth_frames[i], profiles[i],
            apply_flip=True, return_raw=True
        )
        pcds.append(pcd)
        raw_pcds.append(pcd_raw)

    for i, pcd_raw in enumerate(raw_pcds):
        raw_path = raw_dir / f"face_cam{i}_raw_{int(pitch_label_deg)}deg_{timestamp}.ply"
        o3d.io.write_point_cloud(str(raw_path), pcd_raw)

    base_pcd = pcds[0]
    T_1_to_0_icp = core.icp_to_cam0(pcds[1], base_pcd, core.T_1_to_0, source_cam_index=1)
    T_2_to_0_icp = core.icp_to_cam0(pcds[2], base_pcd, core.T_2_to_0, source_cam_index=2)

    pcd0_aligned = base_pcd
    pcd1_aligned = core.copy.deepcopy(pcds[1]); pcd1_aligned.transform(T_1_to_0_icp)
    pcd2_aligned = core.copy.deepcopy(pcds[2]); pcd2_aligned.transform(T_2_to_0_icp)

    merged_pcd = o3d.geometry.PointCloud()
    merged_pcd += pcd0_aligned
    merged_pcd += pcd1_aligned
    merged_pcd += pcd2_aligned

    merged_path = raw_dir / f"face_3cams_geom_merged_{int(pitch_label_deg)}deg_{timestamp}.ply"
    o3d.io.write_point_cloud(str(merged_path), merged_pcd)

    lip_results = []
    for cam_idx in range(len(core.SERIALS)):
        if cam_idx == 0:
            T_cam_to_0 = np.eye(4, dtype=np.float64)
        elif cam_idx == 1:
            T_cam_to_0 = T_1_to_0_icp
        else:
            T_cam_to_0 = T_2_to_0_icp

        res = core.detect_lip_3d_for_camera(
            color_frames[cam_idx],
            depth_frames[cam_idx],
            profiles[cam_idx],
            T_cam_to_0,
            cam_index=cam_idx
        )
        lip_results.append(res)

    if 0 <= pitch_label_deg <= 21.0:
        camera_priority = [0, 2]
    elif pitch_label_deg > 21.0:
        camera_priority = [2, 0]
    if -21.0 <= pitch_label_deg < 0:
        camera_priority = [0, 1]
    elif pitch_label_deg < -21.0:
        camera_priority = [1, 0]

    selected = None
    for idx in camera_priority:
        if idx < len(lip_results) and lip_results[idx].get("ok"):
            selected = lip_results[idx]
            break
    if selected is None or not selected.get("ok"):
        return False

    pts = selected["points_cam0"]

    T_cam0_to_tag_raw = make_T_from_Rt(tag_R, tag_t)
    T_cam0_to_tag = T_cam0_to_tag_raw @ core.T_FLIP

    pts_tag = {k: transform_xyz(v, T_cam0_to_tag) for k, v in pts.items()}
    lip_center_tag = (pts_tag["upper"] + pts_tag["lower"] + pts_tag["left"] + pts_tag["right"]) / 4.0

    face_landmarks = selected["face_landmarks"]
    cam_index = selected["camera_index"]

    annotated = selected.get("annotated_image", None)
    if annotated is not None:
        img_path = mpimg_dir / f"lip_cam{cam_index}_{timestamp}.png"
        cv2.imwrite(str(img_path), annotated)

    h, w, _ = np.asanyarray(color_frames[cam_index].get_data()).shape
    lip_poly = core.build_outer_lip_polygon(face_landmarks, w, h)

    color_intr = color_frames[cam_index].profile.as_video_stream_profile().get_intrinsics()
    if cam_index == 0:
        T_cam_to_cam0 = np.eye(4, dtype=np.float64)
    elif cam_index == 1:
        T_cam_to_cam0 = T_1_to_0_icp
    else:
        T_cam_to_cam0 = T_2_to_0_icp

    mouth_pcd = core.crop_pcd_by_lip_polygon_project(
        merged_pcd=merged_pcd,
        lip_poly_px=lip_poly,
        color_intrinsics=color_intr,
        T_cam_to_cam0=T_cam_to_cam0,
        depth_frame=depth_frames[cam_index],
        depth_tol_m=0.01,
        mask_dilate_px=0,
    )
    if mouth_pcd is None or len(mouth_pcd.points) == 0:
        return False

    mouth_pcd = core.keep_largest_cluster_dbscan(mouth_pcd, eps=0.006, min_points=30)
    if mouth_pcd is None or len(mouth_pcd.points) == 0:
        return False

    def transform_pcd_points(pcd, T):
        pts0 = np.asarray(pcd.points)
        pts_h = np.hstack([pts0, np.ones((len(pts0), 1), dtype=np.float64)])
        pts2 = (T @ pts_h.T).T[:, :3]
        pcd2 = o3d.geometry.PointCloud()
        pcd2.points = o3d.utility.Vector3dVector(pts2)
        if pcd.has_colors():
            pcd2.colors = pcd.colors
        return pcd2

    mouth_pcd_tag = transform_pcd_points(mouth_pcd, T_cam0_to_tag)

    pts_m = np.asarray(mouth_pcd_tag.points)
    pts_centered = pts_m - lip_center_tag.reshape(1, 3)

    x_axis = pts_tag["right"] - pts_tag["left"]
    x_axis = x_axis / (np.linalg.norm(x_axis) + 1e-9)

    y_axis = pts_tag["upper"] - pts_tag["lower"]
    y_axis = y_axis - np.dot(y_axis, x_axis) * x_axis
    y_axis = y_axis / (np.linalg.norm(y_axis) + 1e-9)

    z_axis = np.cross(x_axis, y_axis)
    z_axis = z_axis / (np.linalg.norm(z_axis) + 1e-9)

    R_mouth = np.stack([x_axis, y_axis, z_axis], axis=1)
    pts_local = (R_mouth.T @ pts_centered.T).T

    mouth_out = o3d.geometry.PointCloud()
    mouth_out.points = o3d.utility.Vector3dVector(pts_local)
    if mouth_pcd_tag.has_colors():
        mouth_out.colors = mouth_pcd_tag.colors

    idx = get_next_index(mouth_dir, subject_prefix)
    mouth_path = mouth_dir / f"{subject_prefix}_{idx}.ply"
    
    all_label_dir = all_root / mouth_dir.name
    all_label_dir.mkdir(parents=True, exist_ok=True)
    all_path = all_label_dir / f"{subject_prefix}_{idx}.ply"

    o3d.io.write_point_cloud(str(mouth_path), mouth_out)
    o3d.io.write_point_cloud(str(all_path), mouth_out)
    return True

# -------------------------
# 推論用収録（保存せず mouth_out を返す）
# -------------------------
def capture_and_process_3cams_return_mouth_local(
    pipelines,
    profiles,
    pitch_label_deg: float,
    tag_R,
    tag_t,
):
    color_frames = [None] * len(pipelines)
    depth_frames = [None] * len(pipelines)
    aligns = [rs.align(rs.stream.color) for _ in pipelines]

    def grab_one(i):
        return pipelines[i].wait_for_frames()

    def make_T_from_Rt(R, t):
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = np.asarray(R, dtype=np.float64)
        T[:3, 3] = np.asarray(t, dtype=np.float64).reshape(3)
        return T

    def transform_xyz(xyz, T):
        xyz = np.asarray(xyz, dtype=np.float64).reshape(3)
        p = np.ones(4, dtype=np.float64)
        p[:3] = xyz
        q = T @ p
        return q[:3]

    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=len(pipelines)) as ex:
        for _ in range(core.NUM_FRAMES):
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

    pcds = []
    for i in range(len(core.SERIALS)):
        pcd = core.frames_to_pointcloud(
            color_frames[i], depth_frames[i], profiles[i],
            apply_flip=True, return_raw=False
        )
        pcds.append(pcd)

    base_pcd = pcds[0]
    T_1_to_0_icp = core.icp_to_cam0(pcds[1], base_pcd, core.T_1_to_0, source_cam_index=1)
    T_2_to_0_icp = core.icp_to_cam0(pcds[2], base_pcd, core.T_2_to_0, source_cam_index=2)

    pcd0_aligned = base_pcd
    pcd1_aligned = core.copy.deepcopy(pcds[1]); pcd1_aligned.transform(T_1_to_0_icp)
    pcd2_aligned = core.copy.deepcopy(pcds[2]); pcd2_aligned.transform(T_2_to_0_icp)

    merged_pcd = o3d.geometry.PointCloud()
    merged_pcd += pcd0_aligned
    merged_pcd += pcd1_aligned
    merged_pcd += pcd2_aligned

    lip_results = []
    for cam_idx in range(len(core.SERIALS)):
        if cam_idx == 0:
            T_cam_to_0 = np.eye(4, dtype=np.float64)
        elif cam_idx == 1:
            T_cam_to_0 = T_1_to_0_icp
        else:
            T_cam_to_0 = T_2_to_0_icp

        res = core.detect_lip_3d_for_camera(
            color_frames[cam_idx],
            depth_frames[cam_idx],
            profiles[cam_idx],
            T_cam_to_0,
            cam_index=cam_idx
        )
        lip_results.append(res)

    if 0 <= pitch_label_deg <= 21.0:
        camera_priority = [0, 2]
    elif pitch_label_deg > 21.0:
        camera_priority = [2, 0]
    if -21.0 <= pitch_label_deg < 0:
        camera_priority = [0, 1]
    elif pitch_label_deg < -21.0:
        camera_priority = [1, 0]

    selected = None
    for idx in camera_priority:
        if idx < len(lip_results) and lip_results[idx].get("ok"):
            selected = lip_results[idx]
            break
    if selected is None or not selected.get("ok"):
        return None

    pts = selected["points_cam0"]

    T_cam0_to_tag_raw = make_T_from_Rt(tag_R, tag_t)
    T_cam0_to_tag = T_cam0_to_tag_raw @ core.T_FLIP

    pts_tag = {k: transform_xyz(v, T_cam0_to_tag) for k, v in pts.items()}
    lip_center_tag = (pts_tag["upper"] + pts_tag["lower"] + pts_tag["left"] + pts_tag["right"]) / 4.0

    face_landmarks = selected["face_landmarks"]
    cam_index = selected["camera_index"]

    h, w, _ = np.asanyarray(color_frames[cam_index].get_data()).shape
    lip_poly = core.build_outer_lip_polygon(face_landmarks, w, h)

    color_intr = color_frames[cam_index].profile.as_video_stream_profile().get_intrinsics()
    if cam_index == 0:
        T_cam_to_cam0 = np.eye(4, dtype=np.float64)
    elif cam_index == 1:
        T_cam_to_cam0 = T_1_to_0_icp
    else:
        T_cam_to_cam0 = T_2_to_0_icp

    mouth_pcd = core.crop_pcd_by_lip_polygon_project(
        merged_pcd=merged_pcd,
        lip_poly_px=lip_poly,
        color_intrinsics=color_intr,
        T_cam_to_cam0=T_cam_to_cam0,
        depth_frame=depth_frames[cam_index],
        depth_tol_m=0.01,
        mask_dilate_px=0,
    )
    if mouth_pcd is None or len(mouth_pcd.points) == 0:
        return None

    mouth_pcd = core.keep_largest_cluster_dbscan(mouth_pcd, eps=0.006, min_points=30)
    if mouth_pcd is None or len(mouth_pcd.points) == 0:
        return None

    def transform_pcd_points(pcd, T):
        pts0 = np.asarray(pcd.points)
        pts_h = np.hstack([pts0, np.ones((len(pts0), 1), dtype=np.float64)])
        pts2 = (T @ pts_h.T).T[:, :3]
        pcd2 = o3d.geometry.PointCloud()
        pcd2.points = o3d.utility.Vector3dVector(pts2)
        if pcd.has_colors():
            pcd2.colors = pcd.colors
        return pcd2

    mouth_pcd_tag = transform_pcd_points(mouth_pcd, T_cam0_to_tag)

    pts_m = np.asarray(mouth_pcd_tag.points)
    pts_centered = pts_m - lip_center_tag.reshape(1, 3)

    x_axis = pts_tag["right"] - pts_tag["left"]
    x_axis = x_axis / (np.linalg.norm(x_axis) + 1e-9)

    y_axis = pts_tag["upper"] - pts_tag["lower"]
    y_axis = y_axis - np.dot(y_axis, x_axis) * x_axis
    y_axis = y_axis / (np.linalg.norm(y_axis) + 1e-9)

    z_axis = np.cross(x_axis, y_axis)
    z_axis = z_axis / (np.linalg.norm(z_axis) + 1e-9)

    R_mouth = np.stack([x_axis, y_axis, z_axis], axis=1)
    pts_local = (R_mouth.T @ pts_centered.T).T

    mouth_out = o3d.geometry.PointCloud()
    mouth_out.points = o3d.utility.Vector3dVector(pts_local)
    if mouth_pcd_tag.has_colors():
        mouth_out.colors = mouth_pcd_tag.colors

    return mouth_out

# -------------------------
# GUI 状態
# -------------------------
@dataclass
class AppState:
    # 共通
    subject_dir: Path | None = None
    raw_dir: Path | None = None
    mouth_dir: Path | None = None
    mpimg_dir: Path | None = None

    # 収録（学習）
    current_label: str | None = None
    is_running_capture: bool = False
    stop_event: threading.Event | None = None
    worker_thread: threading.Thread | None = None
    capture_count : int = 0

    # 推論
    infer_parent_dir: Path | None = None
    model_payload: dict | None = None
    is_running_infer: bool = False
    stop_event_infer: threading.Event | None = None
    worker_thread_infer: threading.Thread | None = None
    last_pred_text: str = "PRED: --"

# -------------------------
# ワーカ（学習収録）
# -------------------------
def protocol_worker_capture(
    page: ft.Page,
    state: AppState,
    set_status_threadsafe,
    set_count_threadsafe,
    preview: ft.Image,
    capture_event: threading.Event,
    quit_event: threading.Event,
    train_root: ft.Container,
):
    pipelines = []
    profiles = []
    detector = core.create_detector()
    pitch_hist = core.deque(maxlen=10)

    try:
        set_status_threadsafe("RealSenseカメラを初期化中…")

        for serial in core.SERIALS:
            pipeline, profile = core.create_pipeline(serial)
            set_status_threadsafe(f"RealSenseカメラ {serial} 起動完了")
            pipelines.append(pipeline)
            profiles.append(profile)

        camera_params = core.get_color_intrinsics_from_profile(profiles[0])

        label = state.current_label
        if not label:
            raise RuntimeError("current_label が未設定です。")

        raw_dir_label = state.raw_dir / label
        mouth_dir_label = state.mouth_dir / label
        mpimg_dir_label = state.mpimg_dir / label
        raw_dir_label.mkdir(parents=True, exist_ok=True)
        mouth_dir_label.mkdir(parents=True, exist_ok=True)
        mpimg_dir_label.mkdir(parents=True, exist_ok=True)

        set_status_threadsafe("収録中：Flet画面にフォーカスして 'c' で撮影（ARマーカー必須）")

        is_processing = False

        while not state.stop_event.is_set() and not quit_event.is_set():
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
                tag_size=core.TAG_SIZE_M
            )

            matched_any = False
            frame_vis = color_image0.copy()

            R_tag = None
            t_tag = None
            pitch_deg_smooth = 0.0
            roll_deg = yaw_deg = 0.0

            for r in results:
                R_tag = r.pose_R
                t_tag = r.pose_t
                roll, pitch, yaw = core.rotation_matrix_to_euler(R_tag)
                pitch = -pitch
                roll_deg = float(np.degrees(roll))
                pitch_deg = float(np.degrees(pitch))
                yaw_deg = float(np.degrees(yaw))

                pitch_hist.append(pitch_deg)
                pitch_deg_smooth = sum(pitch_hist) / len(pitch_hist)

                matched_any = True
                break

            capture_ready = matched_any
            new_color = ft.Colors.RED_100 if not capture_ready else ft.Colors.WHITE
            page.run_thread(lambda c=new_color: setattr(train_root, "bgcolor", c))

            cv2.putText(
                frame_vis, f"CAPTURE: {'READY' if capture_ready else 'NG'}",
                (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                (0, 255, 0) if capture_ready else (0, 0, 255), 2
            )
            if matched_any:
                cv2.putText(
                    frame_vis, f"R:{roll_deg:+.1f}  P:{pitch_deg_smooth:+.1f}  Y:{yaw_deg:+.1f} [deg]",
                    (30, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2
                )

            if is_processing:
                cv2.putText(
                    frame_vis, "PROCESSING... DO NOT MOVE",
                    (30, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2
                )

            ok, buf = cv2.imencode(".jpg", frame_vis, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            if ok:
                b64 = base64.b64encode(buf).decode("ascii")
                page.run_thread(lambda b64=b64: setattr(preview, "src_base64", b64))
                page.run_thread(page.update)

            if capture_event.is_set():
                capture_event.clear()
                if capture_ready and (not is_processing) and (R_tag is not None) and (t_tag is not None):
                    is_processing = True
                    try:
                        saved = capture_and_process_3cams_to_dirs_save(
                            pipelines, profiles,
                            pitch_label_deg=pitch_deg_smooth,
                            tag_R=R_tag, tag_t=t_tag,
                            raw_dir=raw_dir_label,
                            mouth_dir=mouth_dir_label,
                            mpimg_dir=mpimg_dir_label,
                            subject_prefix=state.subject_dir.name,
                        )
                        if saved:
                            state.capture_count += 1
                            # count_view を更新（後述の set_count_threadsafe を使う）
                            set_count_threadsafe(state.capture_count)
                            set_status_threadsafe(f"保存しました（COUNT={state.capture_count}, pitch={pitch_deg_smooth:.2f}deg）")
                        else:
                            set_status_threadsafe("保存失敗（口検出/切り出し失敗など）")
                    except Exception as e:
                        set_status_threadsafe(f"保存処理でエラー: {e}")
                    finally:
                        is_processing = False
                else:
                    set_status_threadsafe("c入力：ARマーカー未検出のため撮影しません（NG）")

    except Exception:
        set_status_threadsafe("収録プロトコルで例外:\n" + traceback.format_exc())
    finally:
        for p in pipelines:
            try:
                p.stop()
            except Exception:
                pass

# -------------------------
# ワーカ（推論）
# -------------------------
def protocol_worker_infer(
    page: ft.Page,
    state: AppState,
    set_status_threadsafe,
    set_pred_threadsafe,
    preview: ft.Image,
    capture_event: threading.Event,
    quit_event: threading.Event,
):
    pipelines = []
    profiles = []
    detector = core.create_detector()
    pitch_hist = core.deque(maxlen=10)

    try:
        set_status_threadsafe("推論：RealSenseカメラを初期化中…")

        for serial in core.SERIALS:
            pipeline, profile = core.create_pipeline(serial)
            pipelines.append(pipeline)
            profiles.append(profile)

        camera_params = core.get_color_intrinsics_from_profile(profiles[0])

        if state.model_payload is None:
            raise RuntimeError("モデルが未ロードです。先に推論タブでモデルをロードしてください。")

        set_status_threadsafe("推論中：Flet画面にフォーカスして 'c' で推論（ARマーカー必須）")
        is_processing = False
        last_pred_overlay = "PRED: --"

        while not state.stop_event_infer.is_set() and not quit_event.is_set():
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
                tag_size=core.TAG_SIZE_M
            )

            matched_any = False
            frame_vis = color_image0.copy()

            R_tag = None
            t_tag = None
            pitch_deg_smooth = 0.0
            roll_deg = yaw_deg = 0.0

            for r in results:
                R_tag = r.pose_R
                t_tag = r.pose_t
                roll, pitch, yaw = core.rotation_matrix_to_euler(R_tag)
                pitch = -pitch
                roll_deg = float(np.degrees(roll))
                pitch_deg = float(np.degrees(pitch))
                yaw_deg = float(np.degrees(yaw))

                pitch_hist.append(pitch_deg)
                pitch_deg_smooth = sum(pitch_hist) / len(pitch_hist)

                matched_any = True
                break

            capture_ready = matched_any

            cv2.putText(
                frame_vis, f"CAPTURE: {'READY' if capture_ready else 'NG'}",
                (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                (0, 255, 0) if capture_ready else (0, 0, 255), 2
            )
            if matched_any:
                cv2.putText(
                    frame_vis, f"R:{roll_deg:+.1f}  P:{pitch_deg_smooth:+.1f}  Y:{yaw_deg:+.1f} [deg]",
                    (30, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2
                )

            # 推論結果オーバレイ
            cv2.putText(
                frame_vis, last_pred_overlay,
                (30, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2
            )

            if is_processing:
                cv2.putText(
                    frame_vis, "PROCESSING... DO NOT MOVE",
                    (30, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2
                )

            ok, buf = cv2.imencode(".jpg", frame_vis, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            if ok:
                b64 = base64.b64encode(buf).decode("ascii")
                page.run_thread(lambda b64=b64: setattr(preview, "src_base64", b64))
                page.run_thread(page.update)

            if capture_event.is_set():
                capture_event.clear()
                if capture_ready and (not is_processing) and (R_tag is not None) and (t_tag is not None):
                    is_processing = True
                    try:
                        mouth_local = capture_and_process_3cams_return_mouth_local(
                            pipelines, profiles,
                            pitch_label_deg=pitch_deg_smooth,
                            tag_R=R_tag, tag_t=t_tag,
                        )
                        if mouth_local is None:
                            last_pred_overlay = "PRED: -- (mouth detect failed)"
                            set_pred_threadsafe(last_pred_overlay)
                        else:
                            pred_label, pred_val, _ = predict_from_mouth_pcd(mouth_local, state.model_payload)
                            if pred_label is None:
                                last_pred_overlay = "PRED: --"
                            else:
                                if pred_val is None:
                                    last_pred_overlay = f"PRED: {pred_label}"
                                else:
                                    last_pred_overlay = f"PRED: {pred_label} ({pred_val:.1f}%)"
                            set_pred_threadsafe(last_pred_overlay)
                    except Exception as e:
                        last_pred_overlay = f"PRED: ERROR ({e})"
                        set_pred_threadsafe(last_pred_overlay)
                    finally:
                        is_processing = False
                else:
                    set_status_threadsafe("c入力：ARマーカー未検出のため推論しません（NG）")

    except Exception:
        set_status_threadsafe("推論プロトコルで例外:\n" + traceback.format_exc())
    finally:
        for p in pipelines:
            try:
                p.stop()
            except Exception:
                pass

# -------------------------
# 学習処理（親フォルダ/mouth_ply を DATA_ROOT として学習 → 親フォルダ直下に保存）
# -------------------------
def train_svm_and_save(subject_dir: Path, set_status_threadsafe):
    # faceTrain_SVM_dens.py 相当（保存先だけ親フォルダに変更）
    # 設定（必要ならここを変更）
    GRID = 25
    TEST_SIZE = 0.3
    SEED = 42
    LABEL_ORDER = ["A", "I", "U", "E", "O"]

    SVM_KERNEL = "rbf"
    C_GRID = [0.1, 1, 3, 5, 10, 30, 100]
    GAMMA_GRID = ["scale", "auto"]
    CLASS_WEIGHT = None
    PROBABILITY = True
    CV_SPLITS = 5
    CM_DPI = 200

    data_root = subject_dir / "mouth_ply"
    out_model = subject_dir / MODEL_FILENAME
    out_meta = subject_dir / META_FILENAME
    out_cm = subject_dir / CM_FILENAME

    set_status_threadsafe(f"学習開始：DATA_ROOT={data_root}")

    X, y_str = collect_dataset_fixed_order(data_root, GRID, LABEL_ORDER)
    label_to_id = {lab: i for i, lab in enumerate(LABEL_ORDER)}
    y = np.array([label_to_id[s] for s in y_str], dtype=np.int64)

    idx_all = np.arange(len(y))
    idx_tr, idx_te, y_tr, y_te = train_test_split(
        idx_all, y,
        test_size=TEST_SIZE,
        random_state=SEED,
        stratify=y
    )
    X_tr, X_te = X[idx_tr], X[idx_te]

    cv = StratifiedKFold(n_splits=CV_SPLITS, shuffle=True, random_state=SEED)
    param_grid = {"svc__C": C_GRID, "svc__gamma": GAMMA_GRID}

    grid = GridSearchCV(
        estimator=build_pipeline(SVM_KERNEL, CLASS_WEIGHT, PROBABILITY, SEED),
        param_grid=param_grid,
        scoring="accuracy",
        cv=cv,
        n_jobs=-1,
        refit=True,
        return_train_score=True,
    )
    grid.fit(X_tr, y_tr)

    best_model = grid.best_estimator_
    y_pred = best_model.predict(X_te)

    labels_fixed = list(range(len(LABEL_ORDER)))
    acc = float(accuracy_score(y_te, y_pred))
    cm = confusion_matrix(y_te, y_pred, labels=labels_fixed)
    report = classification_report(
        y_te, y_pred,
        labels=labels_fixed,
        target_names=LABEL_ORDER,
        digits=4
    )

    save_confusion_matrix_png(out_cm, cm, LABEL_ORDER, dpi=CM_DPI)

    payload = {
        "model": best_model,
        "label_order": LABEL_ORDER,
        "grid": GRID,
        "best_params": grid.best_params_,
        "best_cv_score": float(grid.best_score_),
        "test_accuracy": acc,
    }
    joblib.dump(payload, out_model)

    meta = {
        "label_order": LABEL_ORDER,
        "grid": GRID,
        "best_params": grid.best_params_,
        "best_cv_score": float(grid.best_score_),
        "test_accuracy": acc,
    }
    out_meta.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    set_status_threadsafe(
        "学習完了\n"
        f"- model: {out_model}\n"
        f"- meta: {out_meta}\n"
        f"- cm: {out_cm}\n"
        f"- test_accuracy: {acc:.4f}\n"
        f"- best_params: {grid.best_params_}\n"
        f"- report:\n{report}"
    )

# -------------------------
# Flet UI
# -------------------------
def main(page: ft.Page):
    page.title = "口点群GUI（学習/推論統合）"
    page.window_width = 1080
    page.window_height = 720
    page.horizontal_alignment = ft.CrossAxisAlignment.CENTER

    state = AppState()

    DUMMY_PNG_B64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMB/ax0f9kAAAAASUVORK5CYII="
    preview = ft.Image(src_base64=DUMMY_PNG_B64, fit=ft.ImageFit.CONTAIN,)

    status = ft.Text(value="READY", selectable=True)
    count_view = ft.Text(value="COUNT: 0", size=24, weight=ft.FontWeight.BOLD)
    paths_view = ft.Text(value="", selectable=True)
    pred_view = ft.Text(value="PRED: --", selectable=True, size=30)

    
    # ---- 学習タブ：親フォルダ作成 + ラベル収録 + 学習開始 ----
    subject_name = ft.TextField(label="親フォルダ名（被験者名など）")
    # ---- 推論タブ：親フォルダ指定→モデルロード→推論開始（c押下ごと） ----
    infer_parent = ft.TextField(label="推論：親フォルダ名（例：被験者フォルダ）")

    all_buttons = []

    def reg_button(btn: ft.ElevatedButton):
        all_buttons.append(btn)
        return btn

    def apply_responsive_layout(w: float, h: float):
    # 画面幅に追従しつつ、極端に広い画面では上限を設ける
        content_w = min(int(w * 0.95), 1400)

        # 映像：横余白を減らして大きく
        preview.width = content_w
        preview.height = int(h * 0.50)
        preview.fit = ft.ImageFit.CONTAIN

        # 入力欄：映像幅に追従（映像幅の60%）
        tfw = max(320, int(content_w * 0.60))
        subject_name.width = tfw
        infer_parent.width = tfw

        # ボタン：画面幅の10%
        bw = max(140, int(w * 0.10))
        for b in all_buttons:
            b.width = bw

        page.update()

    def on_resize(e: ft.PageResizeEvent):
        apply_responsive_layout(e.width, e.height)

    page.on_resize = on_resize

    # key events（現在動いている worker が参照する capture_event を立てる）
    capture_event = threading.Event()
    quit_event = threading.Event()

    def on_key(e: ft.KeyboardEvent):
        if e.key.lower() == "c":
            capture_event.set()
        elif e.key.lower() == "q":
            quit_event.set()

    page.on_keyboard_event = on_key

    def set_count(n: int):
        count_view.value = f"COUNT: {n}"
        page.update()

    def set_count_threadsafe(n: int):
        page.run_thread(lambda n=n: set_count(n))

    def set_status(msg: str):
        status.value = msg
        page.update()

    def set_pred(msg: str):
        pred_view.value = msg
        page.update()

    def set_status_threadsafe(msg: str):
        page.run_thread(lambda: set_status(msg))

    def set_pred_threadsafe(msg: str):
        page.run_thread(lambda: set_pred(msg))

    def set_paths():
        if state.subject_dir is None:
            paths_view.value = ""
        else:
            paths_view.value = (
                f"親フォルダ: {state.subject_dir}\n"
                f"raw_ply: {state.raw_dir}\n"
                f"mouth_ply: {state.mouth_dir}\n"
                f"mediapipe_img: {state.mpimg_dir}\n"
                f"収録実行中: {state.is_running_capture}\n"
                f"推論実行中: {state.is_running_infer}\n"
            )
        page.update()

    def on_create_folder(_):
        name = safe_subject_name(subject_name.value or "")
        if not name:
            set_status("親フォルダ名が空です。")
            return

        base = Path.cwd()
        subject_dir = base / name
        raw_dir = subject_dir / "raw_ply"
        mouth_dir = subject_dir / "mouth_ply"
        mpimg_dir = subject_dir / "mediapipe_img"

        subject_dir.mkdir(parents=True, exist_ok=True)
        raw_dir.mkdir(parents=True, exist_ok=True)
        mouth_dir.mkdir(parents=True, exist_ok=True)
        mpimg_dir.mkdir(parents=True, exist_ok=True)

        state.subject_dir = subject_dir
        state.raw_dir = raw_dir
        state.mouth_dir = mouth_dir
        state.mpimg_dir = mpimg_dir

        set_status(f"作成しました: {subject_dir}")
        set_paths()

    def on_start_capture_for_label(label: str):
        if state.subject_dir is None:
            set_status("先に親フォルダを作成してください。")
            return
        if state.is_running_capture:
            set_status("すでに収録プロトコル実行中です。")
            return

        capture_event.clear()
        quit_event.clear()
        state.capture_count = 0
        set_count(0)

        state.current_label = label
        state.stop_event = threading.Event()
        state.is_running_capture = True

        t = threading.Thread(
            target=protocol_worker_capture,
            args=(page, state, set_status_threadsafe, set_count_threadsafe, preview, capture_event, quit_event, train_root),
            daemon=True
        )
        state.worker_thread = t
        t.start()

        set_status(f"[学習-収録] {label} 開始：'c'で撮影（ARマーカー必須） / 停止はStop")
        set_paths()

    def on_stop_capture(_):
        if not state.is_running_capture or state.stop_event is None:
            set_status("収録プロトコルは実行中ではありません。")
            return
        state.stop_event.set()
        state.is_running_capture = False
        state.capture_count = 0
        count_view.value = "COUNT: 0"
        page.update()
        set_status("停止要求を出しました。")
        set_paths()

    def on_train_start(_):
        if state.subject_dir is None:
            set_status("先に親フォルダを作成してください。")
            return

        def worker():
            try:
                train_svm_and_save(state.subject_dir, set_status_threadsafe)
            except Exception:
                set_status_threadsafe("学習で例外:\n" + traceback.format_exc())

        threading.Thread(target=worker, daemon=True).start()

    def on_load_model(_):
        name = safe_subject_name(infer_parent.value or "")
        if not name:
            set_status("推論：親フォルダ名が空です。")
            return

        base = Path.cwd()
        parent_dir = base / name
        model_path = parent_dir / MODEL_FILENAME
        if not model_path.exists():
            set_status(f"推論：モデルが見つかりません: {model_path}")
            return

        payload = joblib.load(model_path)
        state.infer_parent_dir = parent_dir
        state.model_payload = payload
        set_status(f"推論：モデルをロードしました: {model_path}\nlabel_order={payload.get('label_order')} grid={payload.get('grid')}")
        set_pred("PRED: --")

    def on_start_infer(_):
        if state.model_payload is None:
            set_status("推論：先にモデルをロードしてください。")
            return
        if state.is_running_infer:
            set_status("推論：すでに推論実行中です。")
            return

        capture_event.clear()
        quit_event.clear()

        state.stop_event_infer = threading.Event()
        state.is_running_infer = True

        t = threading.Thread(
            target=protocol_worker_infer,
            args=(page, state, set_status_threadsafe, set_pred_threadsafe, preview, capture_event, quit_event),
            daemon=True
        )
        state.worker_thread_infer = t
        t.start()

        set_status("推論開始：'c'で推論（ARマーカー必須） / 停止は推論停止")
        set_paths()

    def on_stop_infer(_):
        if not state.is_running_infer or state.stop_event_infer is None:
            set_status("推論：実行中ではありません。")
            return
        state.stop_event_infer.set()
        state.is_running_infer = False
        set_status("推論：停止要求を出しました。")
        set_paths()

    def build_train_content():
        return ft.Column(
            [
                ft.Row([ft.ElevatedButton("戻る", on_click=lambda _: show_home())]),
                ft.Text("学習（収録→学習→モデル保存）", size=18),
                subject_name,
                ft.Row([
                    reg_button(ft.ElevatedButton(text="親フォルダ作成", on_click=on_create_folder)),
                    reg_button(ft.ElevatedButton(text="収録停止", on_click=on_stop_capture)),
                    reg_button(ft.ElevatedButton(text="学習開始（モデル保存）", on_click=on_train_start)),
                ], alignment=ft.MainAxisAlignment.CENTER),
                ft.Row([
                    reg_button(ft.ElevatedButton(text=f"{lbl} 収録開始", on_click=lambda e, l=lbl: on_start_capture_for_label(l)))
                    for lbl in LABELS
                ], alignment=ft.MainAxisAlignment.CENTER),
                ft.Divider(),
                ft.Text("プレビュー（共通）"),
                preview,
                ft.Divider(),
                count_view,
                status,
                paths_view,
                ft.Text("※Flet画面にフォーカスして 'c'（ARマーカー検出時のみ） / 'q'で停止要求", size=12),
            ],
            spacing=10,
            horizontal_alignment=ft.CrossAxisAlignment.CENTER
        )
    train_root = ft.Container(
        bgcolor=ft.Colors.WHITE,
        content=build_train_content(),
        padding=10
    )

    def build_train_page():
        return train_root

    def build_infer_page():
        return ft.Column(
            [
                ft.Row(
                    [
                        ft.ElevatedButton("戻る", on_click=lambda _: show_home()),
                    ]
                ),
                ft.Text("推論（モデルロード→リアルタイム推論）", size=18),
                infer_parent,
                ft.Row([
                    reg_button(ft.ElevatedButton(text="モデルロード", on_click=on_load_model)),
                    reg_button(ft.ElevatedButton(text="推論開始", on_click=on_start_infer)),
                    reg_button(ft.ElevatedButton(text="推論停止", on_click=on_stop_infer)),
                ], alignment=ft.MainAxisAlignment.CENTER),
                ft.Divider(),
                ft.Text("プレビュー（共通）"),
                preview,
                ft.Divider(),
                pred_view,
                status,
                ft.Text("※Flet画面にフォーカスして 'c'（ARマーカー検出時のみ） / 'q'で停止要求", size=12),
            ],
            spacing=10,
            horizontal_alignment=ft.CrossAxisAlignment.CENTER
        )

    def build_home_page():
        return ft.Column(
            [
                ft.Text("モード選択", size=20),
                ft.Row(
                    [
                        reg_button(ft.ElevatedButton("学習へ", on_click=lambda _: show_train())),
                        reg_button(ft.ElevatedButton("推論へ", on_click=lambda _: show_infer())),
                    ],
                    alignment=ft.MainAxisAlignment.CENTER
                ),
                ft.Divider(),
                status,
            ],
            spacing=10
        )

    def show_home():
        # 実行中なら止める（戻る動作）
        try:
            on_stop_capture(None)
        except Exception:
            pass
        try:
            on_stop_infer(None)
        except Exception:
            pass

        page.controls.clear()
        page.add(build_home_page())
        page.update()
        apply_responsive_layout(page.width, page.height)

    def show_train():
        page.controls.clear()
        page.add(build_train_page())
        page.update()
        apply_responsive_layout(page.width, page.height)

    def show_infer():
        page.controls.clear()
        page.add(build_infer_page())
        page.update()
        apply_responsive_layout(page.width, page.height)

    # 起動時はホーム表示
    show_home()

if __name__ == "__main__":
    ft.app(target=main)
