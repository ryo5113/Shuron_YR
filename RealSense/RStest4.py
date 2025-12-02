import pyrealsense2 as rs
import cv2
import numpy as np
import mediapipe as mp
import time

# ===== RealSense setup =====
W, H, FPS = 640, 360, 30
config = rs.config()
config.enable_stream(rs.stream.color, W, H, rs.format.bgr8, FPS)
config.enable_stream(rs.stream.depth, W, H, rs.format.z16, FPS)

pipeline = rs.pipeline()
profile  = pipeline.start(config)

# 深度をカラーにアライン（画角・ピクセルを一致）
align_to_color = rs.align(rs.stream.color)

# 深度の可視化
colorizer = rs.colorizer()
colorizer.set_option(rs.option.color_scheme, 0)  # 見やすい配色に（任意）
colorizer.set_option(rs.option.histogram_equalization_enabled, 0)  # 線形にする(任意)
colorizer.set_option(rs.option.min_distance, 0.1)  # m 単位で近側クリップ
colorizer.set_option(rs.option.max_distance, 1.0)  # m 単位で遠側クリップ

# ===== MediaPipe FaceMesh =====
mp_face_mesh = mp.solutions.face_mesh
mp_drawing   = mp.solutions.drawing_utils
mp_styles    = mp.solutions.drawing_styles

face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

def draw_fps(img, fps):
    cv2.putText(img, f"FPS: {fps:.1f}", (10, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2, cv2.LINE_AA)

def sample_depth_mm(depth_frame, x, y):
    # アライン後のカラー座標(px,py)に対応する深度(ミリ)を取得
    if 0 <= x < depth_frame.get_width() and 0 <= y < depth_frame.get_height():
        d = depth_frame.get_distance(x, y)  # meters
        return d * 1000.0 if d > 0 else np.nan
    return np.nan

try:
    t0 = time.time(); fcount = 0; fps = 0.0
    while True:
        # --- 同期フレームの取得（カラー基準でアライン） ---
        frames = pipeline.wait_for_frames()
        frames = align_to_color.process(frames)
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()
        if not color_frame or not depth_frame:
            continue

        color = np.asanyarray(color_frame.get_data())                 # BGR
        depth_vis = np.asanyarray(colorizer.colorize(depth_frame).get_data())
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_vis, alpha=0.03), cv2.COLORMAP_JET)

        # --- FaceMesh推論（MediaPipeはRGB想定） ---
        rgb = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        overlay = color.copy()
        if results.multi_face_landmarks:
            for fl in results.multi_face_landmarks:
                mp_drawing.draw_landmarks(
                    image=overlay,
                    landmark_list=fl,
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_styles.get_default_face_mesh_tesselation_style()
                )
                # 例: 鼻先(ランドマークID=1)で深度を注釈
                ih, iw = overlay.shape[:2]
                lx, ly = fl.landmark[1].x, fl.landmark[1].y
                px, py = int(lx * iw), int(ly * ih)
                z_mm = sample_depth_mm(depth_frame, px, py)
                if not np.isnan(z_mm):
                    cv2.circle(overlay, (px, py), 3, (0, 255, 255), -1)
                    cv2.putText(overlay, f"{z_mm:.0f} mm", (px+6, py-6),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 2, cv2.LINE_AA)

        # --- 並列表示（サイズを揃える） ---
        overlay   = cv2.resize(overlay, (W, H))
        depth_vis = cv2.resize(depth_vis, (W, H))
        combo = np.hstack([overlay, depth_vis])

        # --- FPS描画 ---
        fcount += 1
        if fcount >= 10:
            now = time.time()
            fps = fcount / (now - t0)
            t0 = now; fcount = 0
        draw_fps(combo, fps)

        cv2.imshow("RealSense FaceMesh | depth aligned to color", combo)
        if cv2.waitKey(1) & 0xFF == 27:  # ESC
            break
finally:
    cv2.destroyAllWindows()
    pipeline.stop()
    face_mesh.close()
