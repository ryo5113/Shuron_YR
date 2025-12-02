import os, time, math, csv, argparse, sys
from collections import deque
from datetime import datetime

import numpy as np
import cv2
import mediapipe as mp
import pyrealsense2 as rs

# ==== 追加（YOLO） ====
from ultralytics import YOLO
import torch

# ===================== 引数 =====================
def parse_args():
    ap = argparse.ArgumentParser(
        description="RealSense + MediaPipe(FaceMesh) + YOLO Tongue Segmentation (overlay RGB | Depth)"
    )
    ap.add_argument("--model", required=True, help="YOLO segment weights (e.g., best.pt)")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf",  type=float, default=0.25)
    ap.add_argument("--iou",   type=float, default=0.50)
    ap.add_argument("--max_det", type=int, default=1)
    ap.add_argument("--device", default=None, help="None->auto / 'cpu' / '0' (GPU index)")
    ap.add_argument("--save_video", action="store_true", help="record combined RGB+Depth view")
    return ap.parse_args()

# ===================== 設定（RStest5.py 由来） =====================
# RealSense
W, H, FPS = 640, 480, 30

# 口開閉（MAR）ヒステリシス
MAR_OPEN_TH, MAR_CLOSE_TH = 0.32, 0.28

# 表示切替（全顔→唇のみ）
VALID_THRES = 440
SUSTAIN_FRAMES_ON  = 60
SUSTAIN_FRAMES_OFF = 60

# スナップ保存
SNAP_DIR = "mouth_snaps"
SNAP_COOLDOWN_SEC = 1.5
CROP_MARGIN = 0.50
os.makedirs(SNAP_DIR, exist_ok=True)

# ==== 録画設定 ====
SAVE_VIDEO   = True  # 実際の有効/無効は引数 --save_video で切り替え
VIDEO_PATH   = "capture.mp4"
VIDEO_FOURCC = "mp4v"
WARMUP_SEC   = 1.0
rec_t0 = None
frames_for_fps = 0

# ==== セッション保存先 ====
SESSION_TS = time.strftime("%Y%m%d_%H%M%S")
BASE_DIR   = f"session_{SESSION_TS}"
SESSION_T0 = time.perf_counter()

DIR_IMAGES = os.path.join(BASE_DIR, "images")
DIR_VIDEO  = os.path.join(BASE_DIR, "video")
DIR_LOGS   = os.path.join(BASE_DIR, "logs")

SNAP_DIR   = DIR_IMAGES
CSV_PATH   = os.path.join(DIR_LOGS, "lip_landmarks_norm.csv")
VIDEO_PATH = os.path.join(DIR_VIDEO, "capture.mp4")

# 距離表示など
SHOW_DISTANCE_RUNTIME = True
SAVE_DISTANCE_TEXT    = False
DIST_IN_FILENAME      = True
DEPTH_WIN = 5
USE_DEPTH_MEDIAN = True

# CSV ログ
LOG_LIPS_TO_CSV = True

# 依存・定数
mp_face_mesh = mp.solutions.face_mesh
mp_draw = mp.solutions.drawing_utils

# 唇だけの接続＆点集合
LIPS_CONNS = mp_face_mesh.FACEMESH_LIPS
LIP_ID_SET = sorted({i for (i, j) in LIPS_CONNS} | {j for (i, j) in LIPS_CONNS})

# 全顔の線
FULL_CONNS = mp_face_mesh.FACEMESH_TESSELATION

# 代表ランドマークID
LM_LIP_LEFT, LM_LIP_RIGHT = 61, 291
LM_INNER_UP, LM_INNER_LO  = 13, 14

# 初回CSVヘッダ
os.makedirs(DIR_LOGS, exist_ok=True)
if LOG_LIPS_TO_CSV and not os.path.exists(CSV_PATH):
    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "timestamp_sec","frame_idx","landmark_id",
            "x_norm","y_norm","z_rel","MAR","mouth_open"
        ])

# === 共有ステータス（conversation.py から参照/更新） ===
_mouth_is_open_shared = False
_recording_flag_shared = False

# ===================== ユーティリティ（RStest5.py 由来） =====================
def now_stamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]

def compute_mar(landmarks, w, h):
    lx, ly = landmarks[LM_LIP_LEFT].x * w,  landmarks[LM_LIP_LEFT].y * h
    rx, ry = landmarks[LM_LIP_RIGHT].x * w, landmarks[LM_LIP_RIGHT].y * h
    ux, uy = landmarks[LM_INNER_UP].x * w,  landmarks[LM_INNER_UP].y * h
    dx, dy = landmarks[LM_INNER_LO].x * w,  landmarks[LM_INNER_LO].y * h
    width  = math.hypot(rx - lx, ry - ly) + 1e-6
    height = math.hypot(dx - ux, dy - uy)
    center = (int((lx+rx)/2), int((ly+ry)/2))
    return float(height/width), center

def mouth_bbox_px(landmarks, w, h, margin_ratio=0.5):
    xs = np.array([landmarks[i].x for i in LIP_ID_SET]) * w
    ys = np.array([landmarks[i].y for i in LIP_ID_SET]) * h
    x0, y0, x1, y1 = xs.min(), ys.min(), xs.max(), ys.max()
    bw, bh = (x1 - x0), (y1 - y0)
    cx, cy = (x0 + x1)/2, (y0 + y1)/2
    half = int(max(bw, bh) * (1 + margin_ratio) * 0.5)
    x0i = max(0, int(cx - half)); y0i = max(0, int(cy - half))
    x1i = min(w, int(cx + half)); y1i = min(h, int(cy + half))
    return x0i, y0i, x1i, y1i

def draw_lips_only(img_bgr, landmarks, w, h, point_color=(0,255,0), line_color=(0,255,0)):
    for (i, j) in LIPS_CONNS:
        p1 = (int(landmarks[i].x * w), int(landmarks[i].y * h))
        p2 = (int(landmarks[j].x * w), int(landmarks[j].y * h))
        cv2.line(img_bgr, p1, p2, line_color, 1, cv2.LINE_AA)
    for i in LIP_ID_SET:
        p = (int(landmarks[i].x * w), int(landmarks[i].y * h))
        cv2.circle(img_bgr, p, 1, point_color, -1, cv2.LINE_AA)

def depth_at_px(depth_frame, x, y, win=3, use_median=True):
    w = depth_frame.get_width()
    h = depth_frame.get_height()
    r = win // 2
    vals = []
    for j in range(max(0, y - r), min(h, y + r + 1)):
        for i in range(max(0, x - r), min(w, x + r + 1)):
            d = depth_frame.get_distance(i, j)  # meters
            if d > 0:
                vals.append(d * 1000.0)
    if not vals:
        return float("nan")
    return float(np.median(vals) if use_median else np.mean(vals))

def save_mouth_snap(color_bgr, landmarks, dist_mm):
    h, w = color_bgr.shape[:2]
    x0, y0, x1, y1 = mouth_bbox_px(landmarks, w, h, CROP_MARGIN)
    crop = color_bgr[y0:y1, x0:x1].copy()

    ann = crop.copy()
    for (i, j) in LIPS_CONNS:
        p1 = (int(landmarks[i].x * w) - x0, int(landmarks[i].y * h) - y0)
        p2 = (int(landmarks[j].x * w) - x0, int(landmarks[j].y * h) - y0)
        cv2.line(ann, p1, p2, (0,255,0), 1, cv2.LINE_AA)
    for i in LIP_ID_SET:
        p = (int(landmarks[i].x * w) - x0, int(landmarks[i].y * h) - y0)
        cv2.circle(ann, p, 1, (0,255,0), -1, cv2.LINE_AA)

    if SAVE_DISTANCE_TEXT:
        txt = f"{dist_mm:.0f} mm" if not math.isnan(dist_mm) else "NaN mm"
        cv2.putText(ann, txt, (8, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2, cv2.LINE_AA)

    ts = now_stamp()
    if DIST_IN_FILENAME and not math.isnan(dist_mm):
        tag = f"_{int(dist_mm)}mm"
    elif DIST_IN_FILENAME and math.isnan(dist_mm):
        tag = "_NaNmm"
    else:
        tag = ""
    base = f"mouth_{ts}_{tag}"
    crop_path = os.path.join(SNAP_DIR, f"{base}_crop.png")
    ann_path  = os.path.join(SNAP_DIR, f"{base}_ann.png")
    cv2.imwrite(crop_path, crop)
    cv2.imwrite(ann_path,  ann)

def count_valid_points(lms, eps=0.0):
    c = 0
    for lm in lms:
        if -eps <= lm.x <= 1.0+eps and -eps <= lm.y <= 1.0+eps:
            c += 1
    return c

# ========== YOLO ユーティリティ（beroSeg.py 由来） ==========
def pick_largest_mask(masks):
    areas = masks.reshape(masks.shape[0], -1).sum(axis=1)
    idx = int(np.argmax(areas))
    return masks[idx]

def overlay_mask(bgr, mask_bin, color=(180,200,255), alpha=0.55, draw_contour=True):
    h, w = bgr.shape[:2]
    m = (mask_bin*255).astype(np.uint8)
    m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
    out = bgr.copy()
    # m>0 の画素をブレンド
    out[m>0] = out[m>0]*(1-alpha) + np.array(color, np.float32)*alpha
    out = out.astype(np.uint8)
    if draw_contour:
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, cnts, -1, (255,255,255), 2, cv2.LINE_AA)
    return out, m

# ===========conversation.py から共有ステータス取得用関数===========
def get_mouth_is_open() -> bool: #_mouth_is_open_sharedと同じbool値を返す
    return bool(_mouth_is_open_shared)

def set_recording_flag(v: bool): #引数と同じbool値を_recording_flag_sharedにセットする
    global _recording_flag_shared
    _recording_flag_shared = bool(v)

def start_camera(model, imgsz=640, conf=0.35, iou=0.5, max_det=1, device=None, save_video=False): #conversation.py から呼ばれる
    device = device if device is not None else ("0" if torch.cuda.is_available() else "cpu")
    print("is_available:", torch.cuda.is_available())
    print("torch.version.cuda:", torch.version.cuda)
    print("device_count:", torch.cuda.device_count())
    if torch.cuda.is_available():
        print("device_name[0]:", torch.cuda.get_device_name(0))
    print(f"[YOLO] device={device}, cuda={torch.cuda.is_available()}")
    model = YOLO(model)
    task = getattr(model.model, "task", None)
    if task != "segment":
        print(f"[ERR] model task is '{task}', expected 'segment'. Wrong weights?")
        sys.exit(1)

    # RealSense 初期化
    config = rs.config()
    config.enable_stream(rs.stream.color, W, H, rs.format.bgr8, FPS)
    config.enable_stream(rs.stream.depth, W, H, rs.format.z16, FPS)
    pipeline = rs.pipeline()
    profile  = pipeline.start(config)
    align_to_color = rs.align(rs.stream.color)

    # Depth 可視化
    colorizer = rs.colorizer()
    colorizer.set_option(rs.option.color_scheme, 0)
    colorizer.set_option(rs.option.histogram_equalization_enabled, 0)
    colorizer.set_option(rs.option.min_distance, 0.3)
    colorizer.set_option(rs.option.max_distance, 1.0)

    # MediaPipe FaceMesh
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    # 状態
    mar_hist = deque(maxlen=8)
    mouth_is_open = False
    prev_state = False
    last_snap_t = 0.0
    show_lips_only = False
    streak_on = streak_off = 0
    frame_idx = 0

    # 保存先
    for d in (DIR_IMAGES, DIR_VIDEO, DIR_LOGS):
        os.makedirs(d, exist_ok=True)

    # 録画フラグ
    enable_record = bool(save_video)
    writer = None
    rec_t0 = None
    frames_for_fps = 0

    try:
        while True:
            frames = pipeline.wait_for_frames()
            frames = align_to_color.process(frames)
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()
            if not color_frame or not depth_frame:
                continue

            color = np.asanyarray(color_frame.get_data())  # BGR
            depth_vis = np.asanyarray(colorizer.colorize(depth_frame).get_data())
            h, w = color.shape[:2]

            # --------- MediaPipe FaceMesh -----------
            rgb_for_mp = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb_for_mp)
            rgb_out = color.copy()

            mid = (w//2, h//2)  # fallback
            mar_smooth = float("nan")

            if results.multi_face_landmarks:
                fl = results.multi_face_landmarks[0]

                # 描画（全顔 or 唇のみ）
                if show_lips_only:
                    draw_lips_only(rgb_out, fl.landmark, w, h)
                else:
                    mp_draw.draw_landmarks(
                        image=rgb_out,
                        landmark_list=fl,
                        connections=FULL_CONNS,
                        landmark_drawing_spec=mp_draw.DrawingSpec(color=(0,255,0), thickness=1, circle_radius=1),
                        connection_drawing_spec=mp_draw.DrawingSpec(color=(200,200,200), thickness=1)
                    )

                # MAR と中心点
                mar, mid = compute_mar(fl.landmark, w, h)
                mar_hist.append(mar)
                mar_smooth = float(np.mean(mar_hist))

                # 開閉判定
                if mouth_is_open:
                    if mar_smooth < MAR_CLOSE_TH:
                        mouth_is_open = False
                else:
                    if mar_smooth > MAR_OPEN_TH:
                        mouth_is_open = True
                state = "OPEN" if mouth_is_open else "CLOSED"
                _mouth_is_open_shared = bool(mouth_is_open)

                # 距離
                dist_mm = depth_at_px(depth_frame, mid[0], mid[1], DEPTH_WIN, USE_DEPTH_MEDIAN)
                if SHOW_DISTANCE_RUNTIME:
                    txt = f"MAR={mar_smooth:.3f} [{state}]  |  D={(dist_mm if not math.isnan(dist_mm) else float('nan')):.0f} mm"
                else:
                    txt = f"MAR={mar_smooth:.3f} [{state}]"
                cv2.putText(rgb_out, txt, (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                            (0,200,255) if mouth_is_open else (255,255,255), 2, cv2.LINE_AA)

                # スナップ保存（遷移 or OPEN中クールダウン）
                t = time.time()
                edge = (mouth_is_open != prev_state)
                cooldown = (t - last_snap_t) >= SNAP_COOLDOWN_SEC
                if edge or (mouth_is_open and cooldown):
                    try:
                        save_mouth_snap(color, fl.landmark, dist_mm)
                    except Exception as e:
                        print("save_mouth_snap error:", e)
                    last_snap_t = t
                prev_state = mouth_is_open

                # CSV 追記
                if LOG_LIPS_TO_CSV:
                    ts = time.time()
                    rows = []
                    for i in LIP_ID_SET:
                        lm = fl.landmark[i]
                        rows.append([
                            f"{ts:.6f}", frame_idx, i,
                            f"{lm.x:.6f}", f"{lm.y:.6f}", f"{lm.z:.6f}",
                            f"{mar_smooth:.6f}", int(mouth_is_open)
                        ])
                    with open(CSV_PATH, "a", newline="", encoding="utf-8") as f:
                        csv.writer(f).writerows(rows)

            # --------- YOLO 舌セグメンテーション（RGBに重畳） ----------
            # rgb_out（FaceMesh描画済みの BGR）を入力にする
            r = model.predict(
                source=rgb_out, imgsz=imgsz, conf=conf, iou=iou,
                device=device, verbose=False, max_det=max_det
            )[0]

            vis = r.plot(conf=True, boxes=True, labels=True, masks=True)  # YOLOの可視化（ボックス・マスクなど）
            # マスクを明示的にブレンド（輪郭付き）
            if r.masks is not None and len(r.masks.data) > 0:
                masks = r.masks.data.cpu().numpy().astype(np.uint8)  # (N,H',W')
                m = pick_largest_mask(masks)
                vis, _ = overlay_mask(vis, m, alpha=0.55, draw_contour=True)

            # --------- 2画面合成（左: YOLO&FaceMesh重畳RGB / 右: Depth可視化） ----------
            rgb_v   = cv2.resize(vis,       (W, H))
            depth_v = cv2.resize(depth_vis, (W, H))
            combo   = np.hstack([rgb_v, depth_v])

            # 録画
            if enable_record:
                now = time.perf_counter()
                if writer is None:
                    if rec_t0 is None:
                        rec_t0 = now
                        frames_for_fps = 0
                    frames_for_fps += 1

                    if (now - rec_t0) >= max(0.2, float(WARMUP_SEC)):
                        eff_fps = max(1.0, frames_for_fps / (now - rec_t0))
                        h_out, w_out = combo.shape[:2]
                        fourcc = cv2.VideoWriter_fourcc(*VIDEO_FOURCC)
                        os.makedirs(DIR_VIDEO, exist_ok=True)
                        writer = cv2.VideoWriter(VIDEO_PATH, fourcc, eff_fps, (w_out, h_out))
                        print(f"[Video] init writer @ eff_fps ≈ {eff_fps:.2f}")
                else:
                    writer.write(combo)

            # 録画中表示(conversation.py からの共有フラグ参照)
            if _recording_flag_shared:
                cv2.putText(combo, "REC", (combo.shape[1]-100, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,0,255), 3, cv2.LINE_AA)

            cv2.imshow("RGB(YOLO+FaceMesh) | Depth (ESC to quit, SPACE: lips-only toggle)", combo)
            k = cv2.waitKey(1) & 0xFF
            if k == 27:  # ESC
                break
            elif k == 32:  # SPACE
                show_lips_only = not show_lips_only

            frame_idx += 1

    finally:
        cv2.destroyAllWindows()
        try:
            pipeline.stop()
        except Exception:
            pass
        try:
            face_mesh.close()
        except Exception:
            pass
        try:
            if writer is not None:
                writer.release()
        except Exception:
            pass

# ===================== メイン =====================
def main():
    args = parse_args()

    device = args.device if args.device is not None else ("0" if torch.cuda.is_available() else "cpu")
    model = YOLO(args.model)
    task = getattr(model.model, "task", None)
    if task != "segment":
        print(f"[ERR] model task is '{task}', expected 'segment'. Wrong weights?")
        sys.exit(1)

    # RealSense 初期化
    config = rs.config()
    config.enable_stream(rs.stream.color, W, H, rs.format.bgr8, FPS)
    config.enable_stream(rs.stream.depth, W, H, rs.format.z16, FPS)
    pipeline = rs.pipeline()
    profile  = pipeline.start(config)
    align_to_color = rs.align(rs.stream.color)

    # Depth 可視化
    colorizer = rs.colorizer()
    colorizer.set_option(rs.option.color_scheme, 0)
    colorizer.set_option(rs.option.histogram_equalization_enabled, 0)
    colorizer.set_option(rs.option.min_distance, 0.3)
    colorizer.set_option(rs.option.max_distance, 1.0)

    # MediaPipe FaceMesh
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    # 状態
    mar_hist = deque(maxlen=8)
    mouth_is_open = False
    prev_state = False
    last_snap_t = 0.0
    show_lips_only = False
    streak_on = streak_off = 0
    frame_idx = 0

    # 保存先
    for d in (DIR_IMAGES, DIR_VIDEO, DIR_LOGS):
        os.makedirs(d, exist_ok=True)

    # 録画フラグ
    enable_record = bool(args.save_video)
    writer = None
    rec_t0 = None
    frames_for_fps = 0

    try:
        while True:
            frames = pipeline.wait_for_frames()
            frames = align_to_color.process(frames)
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()
            if not color_frame or not depth_frame:
                continue

            color = np.asanyarray(color_frame.get_data())  # BGR
            depth_vis = np.asanyarray(colorizer.colorize(depth_frame).get_data())
            h, w = color.shape[:2]

            # --------- MediaPipe FaceMesh -----------
            rgb_for_mp = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb_for_mp)
            rgb_out = color.copy()

            mid = (w//2, h//2)  # fallback
            mar_smooth = float("nan")

            if results.multi_face_landmarks:
                fl = results.multi_face_landmarks[0]

                # 描画（全顔 or 唇のみ）
                if show_lips_only:
                    draw_lips_only(rgb_out, fl.landmark, w, h)
                else:
                    mp_draw.draw_landmarks(
                        image=rgb_out,
                        landmark_list=fl,
                        connections=FULL_CONNS,
                        landmark_drawing_spec=mp_draw.DrawingSpec(color=(0,255,0), thickness=1, circle_radius=1),
                        connection_drawing_spec=mp_draw.DrawingSpec(color=(200,200,200), thickness=1)
                    )

                # MAR と中心点
                mar, mid = compute_mar(fl.landmark, w, h)
                mar_hist.append(mar)
                mar_smooth = float(np.mean(mar_hist))

                # 開閉判定
                if mouth_is_open:
                    if mar_smooth < MAR_CLOSE_TH:
                        mouth_is_open = False
                else:
                    if mar_smooth > MAR_OPEN_TH:
                        mouth_is_open = True
                state = "OPEN" if mouth_is_open else "CLOSED"
                _mouth_is_open_shared = bool(mouth_is_open)

                # 距離
                dist_mm = depth_at_px(depth_frame, mid[0], mid[1], DEPTH_WIN, USE_DEPTH_MEDIAN)
                if SHOW_DISTANCE_RUNTIME:
                    txt = f"MAR={mar_smooth:.3f} [{state}]  |  D={(dist_mm if not math.isnan(dist_mm) else float('nan')):.0f} mm"
                else:
                    txt = f"MAR={mar_smooth:.3f} [{state}]"
                cv2.putText(rgb_out, txt, (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                            (0,200,255) if mouth_is_open else (255,255,255), 2, cv2.LINE_AA)

                # スナップ保存（遷移 or OPEN中クールダウン）
                t = time.time()
                edge = (mouth_is_open != prev_state)
                cooldown = (t - last_snap_t) >= SNAP_COOLDOWN_SEC
                if edge or (mouth_is_open and cooldown):
                    try:
                        save_mouth_snap(color, fl.landmark, dist_mm)
                    except Exception as e:
                        print("save_mouth_snap error:", e)
                    last_snap_t = t
                prev_state = mouth_is_open

                # CSV 追記
                if LOG_LIPS_TO_CSV:
                    ts = time.time()
                    rows = []
                    for i in LIP_ID_SET:
                        lm = fl.landmark[i]
                        rows.append([
                            f"{ts:.6f}", frame_idx, i,
                            f"{lm.x:.6f}", f"{lm.y:.6f}", f"{lm.z:.6f}",
                            f"{mar_smooth:.6f}", int(mouth_is_open)
                        ])
                    with open(CSV_PATH, "a", newline="", encoding="utf-8") as f:
                        csv.writer(f).writerows(rows)

            # --------- YOLO 舌セグメンテーション（RGBに重畳） ----------
            # rgb_out（FaceMesh描画済みの BGR）を入力にする
            r = model.predict(
                source=rgb_out, imgsz=args.imgsz, conf=args.conf, iou=args.iou,
                device=device, verbose=False, max_det=args.max_det
            )[0]

            vis = r.plot(conf=True, boxes=True, labels=True, masks=True)  # YOLOの可視化（ボックス・マスクなど）
            # マスクを明示的にブレンド（輪郭付き）
            if r.masks is not None and len(r.masks.data) > 0:
                masks = r.masks.data.cpu().numpy().astype(np.uint8)  # (N,H',W')
                m = pick_largest_mask(masks)
                vis, _ = overlay_mask(vis, m, alpha=0.55, draw_contour=True)

            # --------- 2画面合成（左: YOLO&FaceMesh重畳RGB / 右: Depth可視化） ----------
            rgb_v   = cv2.resize(vis,       (W, H))
            depth_v = cv2.resize(depth_vis, (W, H))
            combo   = np.hstack([rgb_v, depth_v])

            # 録画
            if enable_record:
                now = time.perf_counter()
                if writer is None:
                    if rec_t0 is None:
                        rec_t0 = now
                        frames_for_fps = 0
                    frames_for_fps += 1

                    if (now - rec_t0) >= max(0.2, float(WARMUP_SEC)):
                        eff_fps = max(1.0, frames_for_fps / (now - rec_t0))
                        h_out, w_out = combo.shape[:2]
                        fourcc = cv2.VideoWriter_fourcc(*VIDEO_FOURCC)
                        os.makedirs(DIR_VIDEO, exist_ok=True)
                        writer = cv2.VideoWriter(VIDEO_PATH, fourcc, eff_fps, (w_out, h_out))
                        print(f"[Video] init writer @ eff_fps ≈ {eff_fps:.2f}")
                else:
                    writer.write(combo)

            # 録画中表示(conversation.py からの共有フラグ参照)
            if _recording_flag_shared:
                cv2.putText(combo, "REC", (combo.shape[1]-100, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,0,255), 3, cv2.LINE_AA)

            cv2.imshow("RGB(YOLO+FaceMesh) | Depth (ESC to quit, SPACE: lips-only toggle)", combo)
            k = cv2.waitKey(1) & 0xFF
            if k == 27:  # ESC
                break
            elif k == 32:  # SPACE
                show_lips_only = not show_lips_only

            frame_idx += 1

    finally:
        cv2.destroyAllWindows()
        try:
            pipeline.stop()
        except Exception:
            pass
        try:
            face_mesh.close()
        except Exception:
            pass
        try:
            if writer is not None:
                writer.release()
        except Exception:
            pass

if __name__ == "__main__":
    main()

# 実行例
# python RSSeg.py --model .\runs\segment\tongue_seg_cpu\weights\best.pt --imgsz 640 --conf 0.35
