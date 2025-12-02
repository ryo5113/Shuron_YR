import os, time, math, csv
from collections import deque
from datetime import datetime

import numpy as np
import cv2
import mediapipe as mp
import pyrealsense2 as rs

import sounddevice as sd
from scipy.io import wavfile
import subprocess

# ===================== 設定 =====================
# RealSense
W, H, FPS = 640, 480, 30
VIDEO_FPS = 21.0

# 口開閉（MAR）ヒステリシス
MAR_OPEN_TH, MAR_CLOSE_TH = 0.32, 0.28

# 表示切替（全顔→唇のみ）
VALID_THRES = 440         # 0<=x<=1,0<=y<=1 の点がこの数以上で「安定」
SUSTAIN_FRAMES_ON  = 60    # 連続で上回ったら唇のみ
SUSTAIN_FRAMES_OFF = 60    # 連続で下回ったら全顔

# スナップ保存
SNAP_DIR = "mouth_snaps"
SNAP_COOLDOWN_SEC = 1.5
CROP_MARGIN = 0.50
os.makedirs(SNAP_DIR, exist_ok=True)

# ==== 録画設定 ====
SAVE_VIDEO   = True
VIDEO_PATH   = "capture.mp4"
VIDEO_FOURCC = "mp4v"   # うまくいかない環境では "avc1" なども試せます
WARMUP_SEC   = 3.0           # 実効FPSを測る時間
video_writer = None
rec_t0 = None
frames_for_fps = 0

# ==== セッションごとの保存ディレクトリ構成 ====
SESSION_TS = time.strftime("%Y%m%d_%H%M%S")            
BASE_DIR   = f"session_{SESSION_TS}"
SESSION_T0 = time.perf_counter()  # この瞬間を t=0 にする

DIR_IMAGES = os.path.join(BASE_DIR, "images")          # スナップ（切り出し・注釈付き）
DIR_VIDEO  = os.path.join(BASE_DIR, "video")           # mp4
DIR_LOGS   = os.path.join(BASE_DIR, "logs")            # csvやその他ログ

# 既存の保存系フラグがある前提で、それらのパスをここで上書き
SNAP_DIR = DIR_IMAGES                                   # 既存: 口元スナップの保存先
CSV_PATH = os.path.join(DIR_LOGS, "lip_landmarks_norm.csv")
VIDEO_PATH = os.path.join(DIR_VIDEO, "capture.mp4")     # 既存のVIDEO_PATHを置き換え

AUDIO_SR    = 48000
AUDIO_PATH  = os.path.join(DIR_VIDEO, f"audio_{SESSION_TS}.wav")
MERGED_PATH = os.path.join(DIR_VIDEO, f"merged_{SESSION_TS}.mp4")

# 距離表示（ランタイム/保存/ファイル名） すべて既定で非表示
SHOW_DISTANCE_RUNTIME = True
SAVE_DISTANCE_TEXT    = False
DIST_IN_FILENAME      = True
DEPTH_WIN = 5
USE_DEPTH_MEDIAN = True

# ---- 唇ランドマークCSV ログ ----
LOG_LIPS_TO_CSV = True
CSV_BUFFER = []

# 依存・定数
mp_face_mesh = mp.solutions.face_mesh
mp_draw = mp.solutions.drawing_utils

# 唇だけの接続＆点集合
LIPS_CONNS = mp_face_mesh.FACEMESH_LIPS
LIP_ID_SET = sorted({i for (i, j) in LIPS_CONNS} | {j for (i, j) in LIPS_CONNS})

# 全顔の線（点も表示するなら later で DrawingSpec 指定）
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

# ===================== ユーティリティ =====================
def now_stamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]

def compute_mar(landmarks, w, h):
    lx, ly = landmarks[LM_LIP_LEFT].x * w,  landmarks[LM_LIP_LEFT].y * h
    rx, ry = landmarks[LM_LIP_RIGHT].x * w, landmarks[LM_LIP_RIGHT].y * h
    ux, uy = landmarks[LM_INNER_UP].x * w,  landmarks[LM_INNER_UP].y * h
    dx, dy = landmarks[LM_INNER_LO].x * w,  landmarks[LM_INNER_LO].y * h
    width  = math.hypot(rx - lx, ry - ly) + 1e-6
    height = math.hypot(dx - ux, dy - uy)
    center = (int((lx+rx)/2), int((ly+ry)/2))  # 両口角の中点
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

def put_fps(img, fps):
    cv2.putText(img, f"FPS: {fps:.1f}", (10, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2, cv2.LINE_AA)

def count_valid_points(lms, eps=0.0):
    c = 0
    for lm in lms:
        if -eps <= lm.x <= 1.0+eps and -eps <= lm.y <= 1.0+eps:
            c += 1
    return c

audio_chunks = [] # 録音データを貯めるバッファ
record_audio = False   # 録音開始を制御するフラグ

def audio_callback(indata, frames, time_info, status):
    """sounddevice のコールバックで呼ばれる。録音データを貯める。"""
    if status:
        print(status)
    if record_audio:
        audio_chunks.append(indata.copy())

def merge_with_ffmpeg():
    """RealSense で保存した VIDEO_PATH と AUDIO_PATH を ffmpeg で結合"""
    cmd = [
        "ffmpeg",
        "-y",
        "-i", VIDEO_PATH,
        "-i", AUDIO_PATH,
        "-c:v", "copy",
        "-c:a", "aac",
        "-shortest",
        MERGED_PATH,
    ]
    subprocess.run(cmd, check=True)
    print("結合完了:", MERGED_PATH)

# ===================== 初期化 =====================
# RealSense
config = rs.config()
config.enable_stream(rs.stream.color, W, H, rs.format.bgr8, FPS)
config.enable_stream(rs.stream.depth, W, H, rs.format.z16, FPS)
pipeline = rs.pipeline()
profile  = pipeline.start(config)
align_to_color = rs.align(rs.stream.color)

# Depthの可視化（見やすさ向上）
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
first_frame_time = None

video_writer = None  # 最初のcombo生成後にサイズ確定して初期化

for d in (DIR_IMAGES, DIR_VIDEO, DIR_LOGS):
    os.makedirs(d, exist_ok=True)

# ===================== ループ =====================
try:
    # （以下、実効FPS測定用コードはコメントアウト）
    # t0 = time.time(); fCount = 0; fps = 0.0
    # MAX_FPS_RETRY = 5         # 実効FPSを測り直す最大回数
    # eff_fps = FPS             # 初期値として設定FPSを入れておく

    # if SAVE_VIDEO:
    #     combo_for_size = None   # 後でサイズ取得用

    #     for attempt in range(MAX_FPS_RETRY):
    #         rec_t0 = time.perf_counter()
    #         frames_for_fps = 0
    #         combo_for_size = None

    #         while True:
    #             frames = pipeline.wait_for_frames()
    #             frames = align_to_color.process(frames)
    #             color_frame = frames.get_color_frame()
    #             depth_frame = frames.get_depth_frame()
    #             if not color_frame or not depth_frame:
    #                 continue

    #             # ここでは最低限 combo が作れればOK（本番ループと同じ処理でも構いません）
    #             color = np.asanyarray(color_frame.get_data())
    #             depth_vis = np.asanyarray(colorizer.colorize(depth_frame).get_data())
    #             rgb_v   = cv2.resize(color, (W, H))
    #             depth_v = cv2.resize(depth_vis, (W, H))
    #             combo_for_size = np.hstack([rgb_v, depth_v])

    #             rgb_tmp = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
    #             _ = face_mesh.process(rgb_tmp)

    #             frames_for_fps += 1
    #             now = time.perf_counter()
    #             if (now - rec_t0) >= WARMUP_SEC:
    #                 eff_fps = frames_for_fps / (now - rec_t0)
    #                 break
    #         print(f"[Video] measured eff_fps ≈ {eff_fps:.2f} (attempt {attempt+1})")

    #         if eff_fps < FPS * 0.95 and eff_fps > FPS * 0.4: #30または低すぎるFPSではなかったらそれを採用
    #             break 
    #         else: # 30だったらもう一回測る
    #             continue

    #     # WARMUP が終わったタイミングで VideoWriter を初期化
    #     if combo_for_size is not None:
    #         h_out, w_out = combo_for_size.shape[:2]
    #         fourcc = cv2.VideoWriter_fourcc(*VIDEO_FOURCC)
    #         os.makedirs(DIR_VIDEO, exist_ok=True)
    #         video_writer = cv2.VideoWriter(VIDEO_PATH, fourcc, eff_fps, (w_out, h_out))
    #         print(f"[Video] init writer @ eff_fps ≈ {eff_fps:.2f}")
    #         record_audio = True  # 録音も有効化

    with sd.InputStream(samplerate=AUDIO_SR,
                        channels=1,
                        dtype="int16",
                        callback=audio_callback):
        print("録画・録音開始（ESCキーで終了）")

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
            combo_for_size = np.hstack([color, depth_vis])  # 最後のフレームを保存

            # FaceMesh
            rgb = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb)

            rgb_out = color.copy()
            if results.multi_face_landmarks:
                fl = results.multi_face_landmarks[0]

                # 有効点数で描画モードを更新
                #valid = count_valid_points(fl.landmark)
                #if valid >= VALID_THRES:
                #    streak_on += 1; streak_off = 0
                #    if not show_lips_only and streak_on >= SUSTAIN_FRAMES_ON:
                #        show_lips_only = True
                #else:
                #    streak_off += 1; streak_on = 0
                #    if show_lips_only and streak_off >= SUSTAIN_FRAMES_OFF:
                #        show_lips_only = False

                # 描画
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
                    if mar_smooth < MAR_CLOSE_TH: mouth_is_open = False
                else:
                    if mar_smooth > MAR_OPEN_TH: mouth_is_open = True
                state = "OPEN" if mouth_is_open else "CLOSED"

                # 距離（ランタイム注記はフラグで制御）
                dist_mm = depth_at_px(depth_frame, mid[0], mid[1], DEPTH_WIN, USE_DEPTH_MEDIAN)
                if SHOW_DISTANCE_RUNTIME:
                    txt = f"MAR={mar_smooth:.3f} [{state}]  |  D={(dist_mm if not math.isnan(dist_mm) else float('nan')):.0f} mm"
                else:
                    txt = f"MAR={mar_smooth:.3f} [{state}]"
                cv2.putText(rgb_out, txt, (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                            (0,200,255) if mouth_is_open else (255,255,255), 2, cv2.LINE_AA)

                # 中点は常に可視化したい場合は次を有効
                # cv2.circle(rgb_out, mid, 4, (0,255,255), -1, cv2.LINE_AA)

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

                # ---- CSV: 唇ランドマーク（正規化）を追記 ----
                if LOG_LIPS_TO_CSV:
                    time_now = time.perf_counter()
                    # まだ基準時刻が決まっていなければ、最初のフレームの時刻を保存
                    if first_frame_time is None:
                        first_frame_time = time_now
                    # 「0フレーム目の時間 = 0.0 秒」とした相対時間
                    time_rel = time_now - first_frame_time
                    for i in LIP_ID_SET:
                        lm = fl.landmark[i]  # 正規化座標（0-1）
                        CSV_BUFFER.append([
                            f"{time_rel:.6f}", frame_idx, i,
                            f"{lm.x:.6f}", f"{lm.y:.6f}", f"{lm.z:.6f}",
                            f"{mar_smooth:.6f}", int(mouth_is_open)
                        ])

            # Depth側のオーバレイ（距離テキストはフラグで）
            depth_out = depth_vis
            # if SHOW_DISTANCE_RUNTIME and results.multi_face_landmarks:
            #     cv2.circle(depth_out, mid, 4, (0,255,255), -1, cv2.LINE_AA)
            #     cv2.putText(depth_out, f"D={dist_mm:.0f} mm" if not math.isnan(dist_mm) else "D=NaN",
            #                 (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2, cv2.LINE_AA)

            # 2画面合成
            rgb_v   = cv2.resize(rgb_out,   (W, H))
            depth_v = cv2.resize(depth_out, (W, H))
            combo   = np.hstack([rgb_v, depth_v])

            # # --- 録画 (実効FPS版）---
            # if SAVE_VIDEO and video_writer is not None:
            #     video_writer.write(combo)
            # --- 録画（固定FPS版）---
            if SAVE_VIDEO:
                if video_writer is None:
                    # 最初のフレームで writer を初期化
                    h_out, w_out = combo.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*VIDEO_FOURCC)
                    os.makedirs(DIR_VIDEO, exist_ok=True)
                    video_writer = cv2.VideoWriter(
                        VIDEO_PATH, fourcc, VIDEO_FPS, (w_out, h_out)
                    )
                    print(f"[Video] init writer @ fixed VIDEO_FPS = {VIDEO_FPS}")
                    # このタイミングから音声も保存開始
                    record_audio = True
                video_writer.write(combo)

            cv2.imshow("RGB | Depth (FaceMesh: full or lips-only + CSV logging)", combo)
            k = cv2.waitKey(1) & 0xFF
            if cv2.waitKey(1) & 0xFF == 27:
                if video_writer is not None:
                    video_writer.release()
                break
            elif k == 32: # SPACE -> 描画モードトグル
                show_lips_only = not show_lips_only

            frame_idx += 1

    # --- ここからループ終了後 (with を抜けたあと) に音声を保存 ---
    if audio_chunks:
        audio_data = np.concatenate(audio_chunks, axis=0)
        wavfile.write(AUDIO_PATH, AUDIO_SR, audio_data)
        print("音声を保存しました:", AUDIO_PATH)

        # 映像と音声が両方あれば結合
        if SAVE_VIDEO and os.path.exists(VIDEO_PATH):
            merge_with_ffmpeg()
        else:
            print("VIDEO_PATH が存在しないため結合をスキップしました。")
    else:
        print("audio_chunks が空です。音声が録音されていません。")
    
    # --- CSV をまとめて書き出し ---
    if LOG_LIPS_TO_CSV and CSV_BUFFER:
        with open(CSV_PATH, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerows(CSV_BUFFER)
        print(f"CSV を書き出しました: {CSV_PATH}  行数={len(CSV_BUFFER)}")


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
    # try:
    #     if video_writer is not None:
    #         video_writer.release()
    # except Exception:
    #     pass
