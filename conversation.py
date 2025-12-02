# unified_realtime_conversation.py （D映像+距離表示 対応版）

import os
import io
import cv2
import csv
import time
import math
import json
import queue
import threading
import datetime
import numpy as np
import requests
import sounddevice as sd
import simpleaudio as sa
import RSSeg

# ====== 設定（必要に応じて変更） ======
LMSTUDIO_URL = "http://localhost:1234/v1/chat/completions"
LMSTUDIO_MODEL = "gemma-3-4b"
VOICEVOX_HOST = "http://127.0.0.1:50021"
VOICEVOX_SPEAKER_ID = 3

WHISPER_MODEL_NAME = "large"
WHISPER_LANGUAGE = "ja"
AUDIO_SAMPLE_RATE = 16000

FRAME_MS = 20
RMS_SILENCE_THRESHOLD = 0.05
SILENCE_HANG_MS = 1000
MIN_UTTERANCE_MS = 400

MAR_OPEN_TH = 0.55
MAR_CLOSE_TH = 0.45
MOUTH_CLOSED_HOLD_MS = 300

SESSION_TS = time.strftime("%Y%m%d_%H%M%S")
OUT_DIR_IMAGES = "./face_frames"
CSV_PATH = f"conversation_log_{SESSION_TS}.csv"
YOLO_MODEL_PATH = ".\\runs\\segment\\tongue_seg_cpu\\weights\\best.pt"

# ====== 依存 ======
import pyrealsense2 as rs
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_styles   = mp.solutions.drawing_styles
mp_face_mesh= mp.solutions.face_mesh
import whisper

# ====== 共有状態 ======
audio_q = queue.Queue()
utterance_q = queue.Queue()
control_lock = threading.Lock()

mouth_is_closed = True
last_mouth_change_ts = 0.0
last_mouth_closed_ts = 0.0
last_mouth_open_ts = 0.0

# ====== 初期化 ======
os.makedirs(OUT_DIR_IMAGES, exist_ok=True)
if not os.path.exists(CSV_PATH):
    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(["role", "text"])

print("[Init] Loading Whisper model (CPU)...")
whisper_model = whisper.load_model(WHISPER_MODEL_NAME, device="cpu")

def now_str():
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")

def write_csv(role, text):
    with open(CSV_PATH, "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow([role, text])

def compute_rms(x: np.ndarray) -> float:
    if x.size == 0: return 0.0
    return float(np.sqrt(np.mean(np.square(x), dtype=np.float64)))

def play_wav_bytes(wav_bytes: bytes):
    import wave
    wf = wave.open(io.BytesIO(wav_bytes), 'rb')
    data = wf.readframes(wf.getnframes())
    sa.WaveObject(data, wf.getnchannels(), wf.getsampwidth(), wf.getframerate()).play().wait_done()

def voicevox_tts(text: str, speaker_id: int = VOICEVOX_SPEAKER_ID) -> bytes:
    q = requests.post(f"{VOICEVOX_HOST}/audio_query", params={"text": text, "speaker": speaker_id}, timeout=30)
    q.raise_for_status()
    s = requests.post(f"{VOICEVOX_HOST}/synthesis", params={"speaker": speaker_id}, json=q.json(), timeout=60)
    s.raise_for_status()
    return s.content

def call_lmstudio(messages):
    payload = {"model": LMSTUDIO_MODEL, "messages": messages, "temperature": 0.7, "stream": False}
    r = requests.post(LMSTUDIO_URL, headers={"Content-Type": "application/json"}, data=json.dumps(payload), timeout=120)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]

# ====== MAR 用ランドマークID ======
MOUTH_LEFT, MOUTH_RIGHT, MOUTH_UP, MOUTH_DOWN = 61, 291, 13, 14

def _dist(a, b): return math.hypot(a[0]-b[0], a[1]-b[1])

def calc_mar(landmarks, image_w, image_h):
    lx, ly = landmarks[MOUTH_LEFT].x * image_w,  landmarks[MOUTH_LEFT].y * image_h
    rx, ry = landmarks[MOUTH_RIGHT].x * image_w, landmarks[MOUTH_RIGHT].y * image_h
    ux, uy = landmarks[MOUTH_UP].x * image_w,    landmarks[MOUTH_UP].y * image_h
    dx, dy = landmarks[MOUTH_DOWN].x * image_w,  landmarks[MOUTH_DOWN].y * image_h
    horiz = _dist((lx,ly), (rx,ry))
    vert  = _dist((ux,uy), (dx,dy))
    mar   = 0.0 if horiz <= 1e-6 else vert / horiz
    center = (int((lx+rx)/2), int((ly+ry)/2))  # 唇中心（距離取得用）
    return mar, center

# ====== 距離取得（RSSegと同等のウィンドウ平均/中央値） ======
def depth_at_px(depth_frame, x, y, win=5, use_median=True):
    w = depth_frame.get_width()
    h = depth_frame.get_height()
    r = win // 2
    vals = []
    for j in range(max(0, y - r), min(h, y + r + 1)):
        for i in range(max(0, x - r), min(w, x + r + 1)):
            d = depth_frame.get_distance(i, j)  # meters
            if d > 0: vals.append(d * 1000.0)
    if not vals: return float("nan")
    return float(np.median(vals) if use_median else np.mean(vals))

# ====== スレッド: RealSense + FaceMesh（RGB と D を横並び表示） ======
def start_camera_thread():
    kwargs = dict(
        model=YOLO_MODEL_PATH,
        imgsz=640, conf=0.35, iou=0.50, max_det=1,
        device="cpu", save_video=True
    )
    th = threading.Thread(target=RSSeg.start_camera, kwargs=kwargs, daemon=True)
    th.start()
    return th
# def video_thread():
#     global mouth_is_closed, last_mouth_change_ts, last_mouth_closed_ts, last_mouth_open_ts

#     pipeline = rs.pipeline()
#     config = rs.config()
#     # RGB + Depth を有効化（RSSeg準拠） :contentReference[oaicite:4]{index=4}
#     config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
#     config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
#     profile  = pipeline.start(config)
#     align_to_color = rs.align(rs.stream.color)  # :contentReference[oaicite:5]{index=5}

#     # Depth 可視化（colorizer） :contentReference[oaicite:6]{index=6}
#     colorizer = rs.colorizer()
#     colorizer.set_option(rs.option.color_scheme, 0)
#     colorizer.set_option(rs.option.histogram_equalization_enabled, 0)
#     colorizer.set_option(rs.option.min_distance, 0.3)
#     colorizer.set_option(rs.option.max_distance, 1.0)

#     face_mesh = mp_face_mesh.FaceMesh(
#         static_image_mode=False, max_num_faces=1, refine_landmarks=True,
#         min_detection_confidence=0.5, min_tracking_confidence=0.5)

#     try:
#         while True:
#             frames = pipeline.wait_for_frames()
#             frames = align_to_color.process(frames)
#             color_frame = frames.get_color_frame()
#             depth_frame = frames.get_depth_frame()
#             if not color_frame or not depth_frame:
#                 continue

#             color = np.asanyarray(color_frame.get_data())   # BGR
#             depth_vis = np.asanyarray(colorizer.colorize(depth_frame).get_data())  # 可視化D画像 :contentReference[oaicite:7]{index=7}
#             h, w = color.shape[:2]

#             # ---- FaceMesh ＋ MAR/唇中心 ----
#             rgb_for_mp = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
#             results = face_mesh.process(rgb_for_mp)
#             rgb_out = color.copy()

#             mid = (w // 2, h // 2)
#             if results.multi_face_landmarks:
#                 fl = results.multi_face_landmarks[0]
#                 mp_drawing.draw_landmarks(
#                     image=rgb_out, landmark_list=fl,
#                     connections=mp_face_mesh.FACEMESH_TESSELATION,
#                     landmark_drawing_spec=None,
#                     connection_drawing_spec=mp_styles.get_default_face_mesh_tesselation_style())
#                 mp_drawing.draw_landmarks(
#                     image=rgb_out, landmark_list=fl,
#                     connections=mp_face_mesh.FACEMESH_LIPS,
#                     landmark_drawing_spec=None,
#                     connection_drawing_spec=mp_styles.get_default_face_mesh_connections_style())

#                 mar, mid = calc_mar(fl.landmark, w, h)

#                 # 口開閉ヒステリシス（閉→開/開→閉）・閉じ保持時間
#                 now = time.time()
#                 with control_lock:
#                     if mouth_is_closed:
#                         if mar >= MAR_OPEN_TH:
#                             mouth_is_closed = False
#                             last_mouth_open_ts = now
#                             last_mouth_change_ts = now
#                     else:
#                         if mar <= MAR_CLOSE_TH and (now - last_mouth_change_ts) * 1000.0 >= MOUTH_CLOSED_HOLD_MS:
#                             mouth_is_closed = True
#                             last_mouth_closed_ts = now
#                             last_mouth_change_ts = now

#                 # 唇中心近傍の距離[mm]（RSSeg同様の関数で取得） :contentReference[oaicite:8]{index=8}
#                 dist_mm = depth_at_px(depth_frame, mid[0], mid[1], win=5, use_median=True)
#                 state = "CLOSED" if mouth_is_closed else "OPEN"
#                 text = f"MAR={mar:.3f} [{state}]  |  D={(dist_mm if not math.isnan(dist_mm) else float('nan')):.0f} mm"
#                 cv2.putText(rgb_out, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
#                             (0,200,255) if not mouth_is_closed else (255,255,255), 2, cv2.LINE_AA)  # :contentReference[oaicite:9]{index=9}

#             # ---- ランドマーク込みフレーム保存（RGB側） ----
#             cv2.imwrite(os.path.join(OUT_DIR_IMAGES, f"{now_str()}.jpg"), rgb_out)

#             # ---- 2画面合成：左=RGB(可視化)、右=Depth（colorizer） ---- :contentReference[oaicite:10]{index=10}
#             rgb_v   = cv2.resize(rgb_out,   (640, 480))
#             depth_v = cv2.resize(depth_vis, (640, 480))
#             combo   = np.hstack([rgb_v, depth_v])

#             cv2.imshow("RGB(FaceMesh) | Depth  (ESC to quit)", combo)
#             if cv2.waitKey(1) & 0xFF == 27:
#                 break

#     finally:
#         pipeline.stop()
#         face_mesh.close()
#         cv2.destroyAllWindows()

# ====== 音声: 無音検出＋口閉じで発話確定 → キュー投入 ======
def audio_thread():
    frame_len = int(AUDIO_SAMPLE_RATE * FRAME_MS / 1000)
    buf = bytearray()
    is_speaking = False
    last_voice_ts = 0.0
    start_ts = 0.0

    def callback(indata, frames, time_info, status):
        nonlocal buf, is_speaking, last_voice_ts, start_ts
        data = indata.copy().flatten()
        rms = compute_rms(data)
        now = time.time()

        if rms >= RMS_SILENCE_THRESHOLD:
            if not is_speaking:
                is_speaking = True
                start_ts = now
                RSSeg.set_recording_flag(True)  # 録音開始指示
            last_voice_ts = now

        pcm16 = (np.clip(data, -1.0, 1.0) * 32767.0).astype(np.int16).tobytes()
        buf.extend(pcm16)

        silence_ms = (now - last_voice_ts) * 1000.0 if last_voice_ts > 0 else 0.0
        closed_now = (not RSSeg.get_mouth_is_open())
        if is_speaking and closed_now and (silence_ms >= SILENCE_HANG_MS):
            dur_ms = (now - start_ts) * 1000.0
            if dur_ms >= MIN_UTTERANCE_MS and len(buf) > 0:
                utterance_q.put(bytes(buf))
            buf = bytearray(); is_speaking = False; last_voice_ts = 0.0; start_ts = 0.0
            RSSeg.set_recording_flag(False)  # 録音停止指示

    with sd.InputStream(channels=1, samplerate=AUDIO_SAMPLE_RATE, dtype='float32',
                        blocksize=frame_len, callback=callback):
        while True:
            time.sleep(0.05)

# ====== 発話処理（Whisper→LM→TTS） ======
def nl_pipeline_thread():
    while True:
        pcm_bytes = utterance_q.get()
        if not pcm_bytes: continue
        audio_np = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0

        try:
            opts = {}
            if WHISPER_LANGUAGE is not None: opts["language"] = WHISPER_LANGUAGE
            result = whisper_model.transcribe(audio_np, fp16=False, **opts)  # CPU
            user_text = result["text"].strip()
        except Exception as e:
            user_text = ""
            print("[STT Error]", e)
        if not user_text: continue

        print("[User]", user_text)
        write_csv("user", user_text)

        try:
            reply = call_lmstudio([
                {"role": "system", "content": "あなたは簡潔で丁寧に日本語で答えるアシスタントです。"},
                {"role": "user", "content": user_text}
            ])
        except Exception as e:
            reply = f"ローカルLLMとの通信でエラー: {e}"

        print("[Assistant]", reply)
        write_csv("assistant", reply)

        try:
            wav_bytes = voicevox_tts(reply, VOICEVOX_SPEAKER_ID)
            play_wav_bytes(wav_bytes)
        except Exception as e:
            print("[TTS Error]", e)

def main():
    tv = start_camera_thread() # RealSense + FaceMesh スレッド開始
    ta = threading.Thread(target=audio_thread, daemon=True)
    tn = threading.Thread(target=nl_pipeline_thread, daemon=True)
    ta.start(); tn.start()
    print("Ready. 映像ウィンドウをアクティブにして ESC で終了")
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        pass

if __name__ == "__main__":
    main()
