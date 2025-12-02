import flet as ft
import cv2
import threading
import time
import base64
import io
from datetime import datetime
from PIL import Image

VIDEO_FOURCC = "mp4v"
WARMUP_SEC   = 1.0   # FPS 測定に使うウォームアップ時間（秒）

def main(page: ft.Page):
    page.title = "Camera Rec Test"

    # 状態管理用の変数（main 内で共有）
    capturing = False
    cap = None
    writer = None

    # FPS 計測用
    rec_t0 = None
    frames_for_fps = 0

    # 映像表示用 Image コントロール
    img = ft.Image(width=640, height=480)
    status_text = ft.Text("待機中")

    def capture_loop():
        nonlocal capturing, cap, writer, rec_t0, frames_for_fps

        while capturing and cap is not None:
            ret, frame = cap.read()
            if not ret:
                break

            now = time.perf_counter()

            # ---- 録画処理（実効 FPS を計測してから VideoWriter を作る）----
            if writer is None:
                # まだ writer が無い → FPS 計測フェーズ
                if rec_t0 is None:
                    rec_t0 = now
                    frames_for_fps = 0
                frames_for_fps += 1

                # 一定時間たったら eff_fps を計算して writer 初期化
                if (now - rec_t0) >= max(0.2, float(WARMUP_SEC)):
                    eff_fps = max(1.0, frames_for_fps / (now - rec_t0))
                    h_out, w_out = frame.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*VIDEO_FOURCC)

                    filename = datetime.now().strftime(
                        "record_%Y%m%d_%H%M%S.mp4"
                    )
                    writer = cv2.VideoWriter(
                        filename,
                        fourcc,
                        eff_fps,
                        (w_out, h_out),
                    )
                    print(f"[Video] init writer @ eff_fps ≈ {eff_fps:.1f}")
                    status_text.value = f"録画中: {filename} (FPS≒{eff_fps:.1f})"
                    status_text.update()
            else:
                # writer がすでにある → 普通に書き込む
                writer.write(frame)
            # 表示用に RGB → PNG バイト → base64 に変換
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(frame_rgb)
            buf = io.BytesIO()
            pil_img.save(buf, format="PNG")
            b64_str = base64.b64encode(buf.getvalue()).decode("utf-8")

            img.src_base64 = b64_str
            img.update()

            # 軽くスリープ（CPU負荷抑制）
            time.sleep(0.03)

        # ループを抜けたら後始末
        if cap is not None:
            cap.release()
            cap = None
        if writer is not None:
            writer.release()
            writer = None

    def start_capture(e):
        nonlocal capturing, cap, writer, rec_t0, frames_for_fps

        if capturing:
            return  # すでに撮影中なら何もしない

        # カメラオープン
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            status_text.value = "カメラを開けませんでした"
            status_text.update()
            return

        capturing = True
        status_text.value = f"録画準備中..."
        status_text.update()

        # 別スレッドでキャプチャ開始
        t = threading.Thread(target=capture_loop, daemon=True)
        t.start()

    def stop_capture(e):
        nonlocal capturing
        if not capturing:
            return
        capturing = False
        status_text.value = "停止しました"
        status_text.update()

    # ボタン配置
    start_btn = ft.ElevatedButton("録画開始", on_click=start_capture)
    stop_btn = ft.ElevatedButton("録画停止", on_click=stop_capture)

    page.add(
        status_text,
        img,
        ft.Row([start_btn, stop_btn]),
    )

if __name__ == "__main__":
    ft.app(target=main)
