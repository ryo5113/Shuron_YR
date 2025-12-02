import cv2
import sounddevice as sd
import numpy as np
from scipy.io import wavfile
import subprocess
import os
import time

# ===== 設定値 =====
CAMERA_INDEX = 0          # 使用するカメラのインデックス
AUDIO_SR = 48000          # 音声サンプリングレート

SESSION_TS = time.strftime("%Y%m%d_%H%M%S")
VIDEO_PATH = f"output_video_{SESSION_TS}.mp4"     # 映像のみ
AUDIO_PATH = f"output_audio_{SESSION_TS}.wav"     # 音声のみ
MERGED_PATH = f"output_merged_{SESSION_TS}.mp4"   # 映像＋音声（最終ファイル）

# 音声データを貯めるバッファ
audio_chunks = []


def audio_callback(indata, frames, time, status):
    """sounddevice のコールバックで呼ばれる。録音データをためるだけ。"""
    if status:
        print(status)
    # int16 でコピーして保存
    audio_chunks.append(indata.copy())


def record_video_and_audio():
    global audio_chunks
    audio_chunks = []

    # --- カメラ初期化 ---
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("カメラを開けませんでした")
        return False

    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 0:
        fps = 30.0  # 取得できない場合のフォールバック

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(VIDEO_PATH, fourcc, fps, (width, height))

    # --- 音声録音開始（ストリーム＋コールバック） ---
    with sd.InputStream(samplerate=AUDIO_SR,
                        channels=1,
                        dtype="int16",
                        callback=audio_callback):
        print("録画・録音開始（ESCキーで終了）")

        while True:
            ret, frame = cap.read()
            if not ret:
                print("フレーム取得に失敗しました")
                break

            writer.write(frame)

            # プレビューとキー入力チェック
            cv2.imshow("preview", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                print("ESCキーが押されたので終了します")
                break

    # 後処理
    writer.release()
    cap.release()
    cv2.destroyAllWindows()

    # --- 音声バッファを1本の配列にまとめて保存 ---
    if len(audio_chunks) == 0:
        print("音声データがありませんでした")
        return False

    audio_data = np.concatenate(audio_chunks, axis=0)
    wavfile.write(AUDIO_PATH, AUDIO_SR, audio_data)
    print("音声を保存しました:", AUDIO_PATH)

    return True


def merge_with_ffmpeg():
    # ffmpeg で映像＋音声を結合
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


if __name__ == "__main__":
    ok = record_video_and_audio()

    if ok and os.path.exists(VIDEO_PATH) and os.path.exists(AUDIO_PATH):
        merge_with_ffmpeg()
    else:
        print("録画または録音が正常に完了しなかったため、結合をスキップしました。")
