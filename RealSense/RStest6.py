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
W, H, FPS = 640, 480, 30  # カメラ設定値（最大）

# 口開閉（MAR）ヒステリシス
MAR_OPEN_TH, MAR_CLOSE_TH = 0.32, 0.28

# 表示切替（全顔→唇のみ）
VALID_THRES = 440
SUSTAIN_FRAMES_ON  = 60
SUSTAIN_FRAMES_OFF = 60

# スナップ保存関連
SNAP_COOLDOWN_SEC = 1.5
CROP_MARGIN = 0.50

# ==== セッションフォルダ構成 ====
SESSION_TS = time.strftime("%Y%m%d_%H%M%S")
BASE_DIR   = f"session_{SESSION_TS}"
DIR_IMAGES = os.path.join(BASE_DIR, "images")
DIR_VIDEO  = os.path.join(BASE_DIR, "video")
DIR_LOGS   = os.path.join(BASE_DIR, "logs")
for d in (BASE_DIR, DIR_IMAGES, DIR_VIDEO, DIR_LOGS):
    os.makedirs(d, exist_ok=True)

# スナップ保存先
SNAP_DIR = DIR_IMAGES

# ==== 録画設定 ====
SAVE_VIDEO   = True
VIDEO_FOURCC = "mp4v"
VIDEO_PATH   = os.path.join(DIR_VIDEO, "capture.mp4")

# ==== 音声録音 ====
AUDIO_SR    = 48000
AUDIO_PATH  = os.path.join(DIR_VIDEO, f"audio_{SESSION_TS}.wav")
MERGED_PATH = os.path.join(DIR_VIDEO, f"merged_{SESSION_TS}.mp4")
# 有音判定（簡易VAD）
AUDIO_VAD_TH = 0.1   # RMS 正規化値の閾値（サンプル）

audio_level_rms = 0.0   # 直近チャンクの RMS 値
audio_is_active = False # 有音フラグ（CSVに書き出す）

# 距離表示
SHOW_DISTANCE_RUNTIME = True
SAVE_DISTANCE_TEXT    = False
DIST_IN_FILENAME      = True
DEPTH_WIN             = 5
USE_DEPTH_MEDIAN      = True

# ---- 唇ランドマークCSV ログ ----
LOG_LIPS_TO_CSV = True
CSV_PATH        = os.path.join(DIR_LOGS, "lip_landmarks_norm.csv")
CSV_BUFFER      = []       # 本番用バッファ
CALIB_CSV_BUFFER = []      # キャリブ用バッファ

# 初回CSVヘッダ
if LOG_LIPS_TO_CSV and not os.path.exists(CSV_PATH):
    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "timestamp_sec","frame_idx","landmark_id",
            "x_norm","y_norm","z_RS","MAR","mouth_open","voice_active","sound_level_rms"
        ])

# ==== MediaPipe 関連 ====
mp_face_mesh = mp.solutions.face_mesh
mp_draw      = mp.solutions.drawing_utils

LIPS_CONNS = mp_face_mesh.FACEMESH_LIPS
LIP_ID_SET = sorted({i for (i, j) in LIPS_CONNS} | {j for (i, j) in LIPS_CONNS})

FULL_CONNS = mp_face_mesh.FACEMESH_TESSELATION

LM_LIP_LEFT, LM_LIP_RIGHT = 61, 291
LM_INNER_UP, LM_INNER_LO  = 13, 14

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

def draw_lips_only(img_bgr, landmarks, w, h,
                   point_color=(0,255,0), line_color=(0,255,0)):
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
            d = depth_frame.get_distance(i, j)
            if d > 0:
                vals.append(d * 1000.0)
    if not vals:
        return float("nan")
    arr = np.array(vals)
    return float(np.median(arr) if use_median else np.mean(arr))

def save_mouth_snap(color_bgr, landmarks, dist_mm, prefix="mouth"):
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
    base = f"{prefix}_{ts}{tag}"

    crop_path = os.path.join(SNAP_DIR, f"{base}_crop.png")
    ann_path  = os.path.join(SNAP_DIR, f"{base}_ann.png")

    cv2.imwrite(crop_path, crop)
    cv2.imwrite(ann_path,  ann)

# ===================== 音声まわり =====================
audio_chunks = []        # 本番用
record_audio = False     # 本番で VideoWriter 初期化後に True

def audio_callback(indata, frames, time_info, status):
    global audio_level_rms, audio_is_active
    if status:
        print("[Audio] status:", status)
    
    # int16 → float に正規化して RMS を計算
    data_f = indata.astype(np.float32) / 32768.0
    rms = float(np.sqrt(np.mean(data_f**2)))
    audio_level_rms = rms

    # 閾値で有音/無音を判定
    audio_is_active = (rms > AUDIO_VAD_TH)

    if record_audio:
        audio_chunks.append(indata.copy())

def merge_with_ffmpeg():
    if not (os.path.exists(VIDEO_PATH) and os.path.exists(AUDIO_PATH)):
        print("[FFmpeg] 入力ファイル不足で結合スキップ")
        return
    cmd = [
        "ffmpeg", "-y",
        "-i", VIDEO_PATH,
        "-i", AUDIO_PATH,
        "-c:v", "copy",
        "-c:a", "aac",
        "-shortest",
        MERGED_PATH,
    ]
    try:
        subprocess.run(cmd, check=True)
        print("[FFmpeg] 結合完了:", MERGED_PATH)
    except Exception as e:
        print("[FFmpeg] 結合失敗:", e)

# ===================== RealSense 初期化 =====================
config = rs.config()
config.enable_stream(rs.stream.color, W, H, rs.format.bgr8, FPS)
config.enable_stream(rs.stream.depth, W, H, rs.format.z16, FPS)
pipeline = rs.pipeline()
profile  = pipeline.start(config)
align_to_color = rs.align(rs.stream.color)

colorizer = rs.colorizer()
colorizer.set_option(rs.option.color_scheme, 0)
colorizer.set_option(rs.option.histogram_equalization_enabled, 0)
colorizer.set_option(rs.option.min_distance, 0.3)
colorizer.set_option(rs.option.max_distance, 1.0)

# ===================== MediaPipe FaceMesh =====================
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# ===================== 状態変数（本番用） =====================
mar_hist       = deque(maxlen=8)
mouth_is_open  = False
prev_state     = False
last_snap_t    = 0.0
show_lips_only = False
frame_idx      = 0
first_frame_time = None
video_writer   = None

# ===================== キャリブレーション関数 =====================
def calibrate_fps_manual():
    """
    本番とほぼ同じ処理（RealSense + FaceMesh + MAR + 距離 + テキスト描画 +
    スナップ + CSVバッファ + VideoWriter + 音声録音）を行い、
    ESC で手動終了した区間の実効FPSを測る。
    """

    print("\n[Calib] キャリブレーション開始（ESCで終了）")

    # キャリブ用のファイルパス
    calib_video_path = os.path.join(DIR_VIDEO, f"calib_capture_{SESSION_TS}.mp4")
    calib_audio_path = os.path.join(DIR_VIDEO, f"calib_audio_{SESSION_TS}.wav")
    calib_csv_path   = os.path.join(DIR_LOGS,  f"calib_lip_landmarks_{SESSION_TS}.csv")

    calib_audio_chunks = []
    calib_record_audio = False
    calib_video_writer = None

    calib_mar_hist    = deque(maxlen=8)
    calib_mouth_open  = False
    calib_prev_state  = False
    calib_last_snap_t = 0.0
    calib_frame_idx   = 0
    calib_first_time  = None

    global CALIB_CSV_BUFFER
    CALIB_CSV_BUFFER = []

    frame_count = 0
    t0 = None

    def calib_audio_callback(indata, frames, time_info, status):
        if status:
            print("[Calib-Audio] status:", status)
        if calib_record_audio:
            calib_audio_chunks.append(indata.copy())

    with sd.InputStream(samplerate=AUDIO_SR,
                        channels=1,
                        dtype="int16",
                        callback=calib_audio_callback):
        while True:
            frames = pipeline.wait_for_frames()
            frames = align_to_color.process(frames)
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()
            if not color_frame or not depth_frame:
                continue

            color = np.asanyarray(color_frame.get_data())
            depth_vis = np.asanyarray(colorizer.colorize(depth_frame).get_data())
            h, w = color.shape[:2]

            rgb = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb)

            rgb_out = color.copy()
            dist_mm = float("nan")

            if results.multi_face_landmarks:
                fl = results.multi_face_landmarks[0]

                # 全顔描画
                mp_draw.draw_landmarks(
                    image=rgb_out,
                    landmark_list=fl,
                    connections=FULL_CONNS,
                    landmark_drawing_spec=mp_draw.DrawingSpec(
                        color=(0,255,0), thickness=1, circle_radius=1
                    ),
                    connection_drawing_spec=mp_draw.DrawingSpec(
                        color=(200,200,200), thickness=1
                    )
                )

                # MARと口の開閉
                mar, mid = compute_mar(fl.landmark, w, h)
                calib_mar_hist.append(mar)
                mar_smooth = float(np.mean(calib_mar_hist))
                if calib_mouth_open:
                    if mar_smooth < MAR_CLOSE_TH:
                        calib_mouth_open = False
                else:
                    if mar_smooth > MAR_OPEN_TH:
                        calib_mouth_open = True
                state = "OPEN" if calib_mouth_open else "CLOSED"

                # 距離
                dist_mm = depth_at_px(depth_frame, mid[0], mid[1],
                                      DEPTH_WIN, USE_DEPTH_MEDIAN)

                # テキスト描画
                if SHOW_DISTANCE_RUNTIME:
                    txt = f"[CALIB] MAR={mar_smooth:.3f} [{state}]  |  D={(dist_mm if not math.isnan(dist_mm) else float('nan')):.0f} mm"
                else:
                    txt = f"[CALIB] MAR={mar_smooth:.3f} [{state}]"
                cv2.putText(rgb_out, txt, (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                            (0,200,255) if calib_mouth_open else (255,255,255),
                            2, cv2.LINE_AA)

                # スナップ保存（prefixをcalib_にして区別）
                t = time.time()
                edge = (calib_mouth_open != calib_prev_state)
                cooldown = (t - calib_last_snap_t) >= SNAP_COOLDOWN_SEC
                if edge or (calib_mouth_open and cooldown):
                    try:
                        save_mouth_snap(color, fl.landmark, dist_mm,
                                        prefix="calib_mouth")
                    except Exception as e:
                        print("[Calib] save_mouth_snap error:", e)
                    calib_last_snap_t = t
                calib_prev_state = calib_mouth_open

                # CSVバッファ（キャリブ用）
                if LOG_LIPS_TO_CSV:
                    t_now = time.perf_counter()
                    if calib_first_time is None:
                        calib_first_time = t_now
                    t_rel = t_now - calib_first_time
                    for i in LIP_ID_SET:
                        lm = fl.landmark[i]
                        CALIB_CSV_BUFFER.append([
                            f"{t_rel:.2f}", calib_frame_idx, i,
                            f"{lm.x:.2f}", f"{lm.y:.2f}", f"{dist_mm:.2f}",
                            f"{mar_smooth:.2f}", int(calib_mouth_open), int(audio_is_active), f"{audio_level_rms:.2f}"
                        ])

            # 2画面合成
            rgb_v   = cv2.resize(rgb_out,   (W, H))
            depth_v = cv2.resize(depth_vis, (W, H))
            combo   = np.hstack([rgb_v, depth_v])

            # VideoWriter（キャリブ用）：fpsはとりあえずカメラ設定値FPSでOK（負荷目的）
            if SAVE_VIDEO:
                if calib_video_writer is None:
                    h_out, w_out = combo.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*VIDEO_FOURCC)
                    calib_video_writer = cv2.VideoWriter(
                        calib_video_path, fourcc, FPS, (w_out, h_out)
                    )
                    print(f"[Calib] VideoWriter init: {calib_video_path}, fps={FPS}")
                    # 映像と同時に音声も本格的に貯め始める
                    calib_record_audio = True
                calib_video_writer.write(combo)

            cv2.imshow("Calib (same as main)", combo)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESCでキャリブレーション終了
                print("[Calib] ESC で終了")
                break

            if t0 is None:
                t0 = time.perf_counter()
            frame_count     += 1
            calib_frame_idx += 1

    cv2.destroyWindow("Calib (same as main)")

    # 実効FPS計算
    if t0 is None or frame_count == 0:
        print("[Calib] フレームが取得できなかったため、設定FPSを使用します")
        eff_fps = float(FPS)
    else:
        elapsed = time.perf_counter() - t0
        eff_fps = frame_count / elapsed
        print(f"[Calib] frames={frame_count}, elapsed={elapsed:.3f}s, eff_fps≈{eff_fps:.2f}")

    # キャリブ用動画をクローズ
    if calib_video_writer is not None:
        calib_video_writer.release()

    # キャリブ音声を保存（必要なければこのブロックごと消してもよい）
    if calib_audio_chunks:
        calib_audio_data = np.concatenate(calib_audio_chunks, axis=0)
        wavfile.write(calib_audio_path, AUDIO_SR, calib_audio_data)
        print("[Calib] 音声を保存しました:", calib_audio_path)

    # キャリブCSVを書き出し
    if LOG_LIPS_TO_CSV and CALIB_CSV_BUFFER:
        with open(calib_csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([
                "timestamp_sec","frame_idx","landmark_id",
                "x_norm","y_norm","z(RS)","MAR","mouth_open","voice_active","sound_level_rms"
            ])
            w.writerows(CALIB_CSV_BUFFER)
        print("[Calib] CSVを書き出しました:", calib_csv_path,
              "行数=", len(CALIB_CSV_BUFFER))

    print("[Calib] キャリブレーション完了\n")
    return eff_fps

# ===================== メインループ =====================
try:
    # 1) キャリブレーション（手動終了）
    eff_fps = calibrate_fps_manual()
    print(f"[Info] 本番で使用するFPS（実効FPS）: {eff_fps:.2f}")
    print("[Info] これから本番計測を開始します。録画・録音開始（ESCキーで終了）\n")

    with sd.InputStream(samplerate=AUDIO_SR,
                        channels=1,
                        dtype="int16",
                        callback=audio_callback):
        print("=== 本番開始: RealSense + FaceMesh + 音声録音 ===")

        while True:
            frames = pipeline.wait_for_frames()
            frames = align_to_color.process(frames)
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()
            if not color_frame or not depth_frame:
                continue

            color = np.asanyarray(color_frame.get_data())
            depth_vis = np.asanyarray(colorizer.colorize(depth_frame).get_data())
            h, w = color.shape[:2]

            rgb = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb)

            rgb_out = color.copy()
            dist_mm = float("nan")

            if results.multi_face_landmarks:
                fl = results.multi_face_landmarks[0]

                # 描画モード
                if show_lips_only:
                    draw_lips_only(rgb_out, fl.landmark, w, h)
                else:
                    mp_draw.draw_landmarks(
                        image=rgb_out,
                        landmark_list=fl,
                        connections=FULL_CONNS,
                        landmark_drawing_spec=mp_draw.DrawingSpec(
                            color=(0,255,0), thickness=1, circle_radius=1
                        ),
                        connection_drawing_spec=mp_draw.DrawingSpec(
                            color=(200,200,200), thickness=1
                        )
                    )

                mar, mid = compute_mar(fl.landmark, w, h)
                mar_hist.append(mar)
                mar_smooth = float(np.mean(mar_hist))

                if mouth_is_open:
                    if mar_smooth < MAR_CLOSE_TH:
                        mouth_is_open = False
                else:
                    if mar_smooth > MAR_OPEN_TH:
                        mouth_is_open = True
                state = "OPEN" if mouth_is_open else "CLOSED"

                dist_mm = depth_at_px(depth_frame, mid[0], mid[1],
                                      DEPTH_WIN, USE_DEPTH_MEDIAN)

                if SHOW_DISTANCE_RUNTIME:
                    txt = f"MAR={mar_smooth:.3f} [{state}]  |  D={(dist_mm if not math.isnan(dist_mm) else float('nan')):.0f} mm"
                else:
                    txt = f"MAR={mar_smooth:.3f} [{state}]"
                cv2.putText(rgb_out, txt, (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                            (0,200,255) if mouth_is_open else (255,255,255),
                            2, cv2.LINE_AA)

                # スナップ保存（本番）
                t = time.time()
                edge = (mouth_is_open != prev_state)
                cooldown = (t - last_snap_t) >= SNAP_COOLDOWN_SEC
                if edge or (mouth_is_open and cooldown):
                    try:
                        save_mouth_snap(color, fl.landmark, dist_mm,
                                        prefix="mouth")
                    except Exception as e:
                        print("[Main] save_mouth_snap error:", e)
                    last_snap_t = t
                prev_state = mouth_is_open

                # CSVバッファ（本番）
                if LOG_LIPS_TO_CSV:
                    t_now = time.perf_counter()
                    if first_frame_time is None:
                        first_frame_time = t_now
                    t_rel = t_now - first_frame_time
                    for i in LIP_ID_SET:
                        lm = fl.landmark[i]
                        CSV_BUFFER.append([
                            f"{t_rel:.2f}", frame_idx, i,
                            f"{lm.x:.2f}", f"{lm.y:.2f}", f"{dist_mm:.2f}",
                            f"{mar_smooth:.2f}", int(mouth_is_open), int(audio_is_active), f"{audio_level_rms:.2f}"
                        ])

            # 2画面合成
            rgb_v   = cv2.resize(rgb_out,   (W, H))
            depth_v = cv2.resize(depth_vis, (W, H))
            combo   = np.hstack([rgb_v, depth_v])

            # VideoWriter（本番）: eff_fps を使用
            if SAVE_VIDEO:
                if video_writer is None:
                    h_out, w_out = combo.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*VIDEO_FOURCC)
                    video_writer = cv2.VideoWriter(
                        VIDEO_PATH, fourcc, eff_fps, (w_out, h_out)
                    )
                    print(f"[Video] init writer @ eff_fps = {eff_fps:.2f}")
                    # 映像開始とともに音声もバッファに貯める
                    record_audio = True
                video_writer.write(combo)

            cv2.imshow("RGB | Depth (main)", combo)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESCで本番終了
                print("[Main] ESC で終了")
                break
            elif key == 32:  # Spaceで全顔/唇のみ切り替え
                show_lips_only = not show_lips_only

            frame_idx += 1

    if video_writer is not None:
        video_writer.release()
        video_writer = None

    # ---- 本番終了後：音声保存＋結合 ----
    if audio_chunks:
        audio_data = np.concatenate(audio_chunks, axis=0)
        wavfile.write(AUDIO_PATH, AUDIO_SR, audio_data)
        print("[Main] 音声を保存しました:", AUDIO_PATH)
        if SAVE_VIDEO and os.path.exists(VIDEO_PATH):
            merge_with_ffmpeg()
        else:
            print("[Main] VIDEO_PATH がないため結合スキップ")
    else:
        print("[Main] audio_chunks が空です。音声なし。")

    # ---- CSV をまとめて保存 ----
    if LOG_LIPS_TO_CSV and CSV_BUFFER:
        with open(CSV_PATH, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerows(CSV_BUFFER)
        print("[Main] CSVを書き出しました:", CSV_PATH,
              "行数=", len(CSV_BUFFER))

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
        if video_writer is not None:
            video_writer.release()
    except Exception:
        pass
