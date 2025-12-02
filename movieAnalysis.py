import os
from pathlib import Path
import csv
import subprocess

import cv2
import numpy as np
import mediapipe as mp

# ①② で利用する既存スクリプト
import soundAnalysis   # /mnt/data/soundAnalysis.py
import soundWhisper    # /mnt/data/soundWhisper.py

# ========= 設定 =========
# 元動画（映像＋音声入り）
VIDEO_PATH = "output_merged_20251114_225145.mp4"

# ③ MediaPipe 口唇ランドマーク出力（映像のみ）
LIPS_SILENT_VIDEO_PATH = "output_lips_20251114_225145.mp4"
LIPS_CSV_PATH          = "lips_norm_20251114_225145.csv"

# ④ 元音声を載せ直した MediaPipe 処理付き動画
LIPS_WITH_AUDIO_PATH   = "output_lips_with_audio_20251114_225145.mp4"

# Whisper の出力テキスト
WHISPER_TXT_PATH = "whisper_result_20251114_225145.txt"
# ========================


def run_sound_analysis(video_path: str) -> Path:
    """
    ① soundAnalysis.py を用いて周波数解析＋ノイズ除去を実行。
       TARGET_VIDEO / OUTPUT_DIR を上書きして main() を呼ぶ。
       戻り値として、denoise 後の音声ファイル path を返す。
    """
    base = Path(video_path)

    # 解析対象動画と出力フォルダを上書き
    soundAnalysis.TARGET_VIDEO = str(video_path)
    soundAnalysis.OUTPUT_DIR   = f"out_{base.stem}"

    # 既存の main() 実行
    soundAnalysis.main()

    # denoise 後の WAV は soundAnalysis.OUTPUT_DIR 内の audio_denoised.wav として出力される前提
    audio_path = Path(soundAnalysis.OUTPUT_DIR) / "audio_denoised.wav"
    return audio_path


def run_whisper(audio_path: Path, txt_path: str) -> None:
    """
    ② soundWhisper.py を用いて、音声の文字起こしを実行。
       AUDIO_PATH / OUTPUT_PATH を上書きして main() を呼ぶ。
    """
    soundWhisper.AUDIO_PATH  = str(audio_path)
    soundWhisper.OUTPUT_PATH = txt_path

    soundWhisper.main()


def run_mediapipe_lips(video_path: str,
                       csv_path: str,
                       out_video_path: str) -> None:
    """
    ③ MediaPipe を用いて、動画から口元のランドマークを取得し、
       - 口唇ランドマークを描画した動画（映像のみ）
       - 正規化（0〜1）の口唇ランドマーク座標 CSV
       を出力する。
    """
    mp_face_mesh   = mp.solutions.face_mesh
    mp_drawing     = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"動画を開けませんでした: {video_path}")
        return

    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 0:
        fps = 30.0

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_video_path, fourcc, fps, (width, height))

    # 口唇ランドマークのインデックス集合
    LIPS_CONNS = mp_face_mesh.FACEMESH_LIPS
    lip_id_set = sorted({i for (i, j) in LIPS_CONNS} | {j for (i, j) in LIPS_CONNS})

    # CSV の準備
    csv_file = open(csv_path, "w", newline="", encoding="utf-8")
    csv_writer = csv.writer(csv_file)

    # ヘッダ
    header = ["frame", "time_sec"]
    for lid in lip_id_set:
        header += [f"lip_{lid}_x", f"lip_{lid}_y", f"lip_{lid}_z"]
    csv_writer.writerow(header)

    frame_idx = 0

    with mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as face_mesh:

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(frame_rgb)

            time_sec = frame_idx / fps if fps > 0 else 0.0
            row = [frame_idx, time_sec]

            if results.multi_face_landmarks:
                face_landmarks = results.multi_face_landmarks[0]
                lm_dict = {
                    i: face_landmarks.landmark[i]
                    for i in range(len(face_landmarks.landmark))
                }

                for lid in lip_id_set:
                    lm = lm_dict.get(lid, None)
                    if lm is None:
                        row += ["", "", ""]
                    else:
                        row += [lm.x, lm.y, lm.z]

                # 口唇のみ描画（スタイル自体はデフォルトのテッセレーションスタイルを流用）
                mp_drawing.draw_landmarks(
                    image=frame,
                    landmark_list=face_landmarks,
                    connections=LIPS_CONNS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles
                        .get_default_face_mesh_tesselation_style(),
                )
            else:
                # 顔が検出されなかったフレームは空欄
                for _ in lip_id_set:
                    row += ["", "", ""]

            csv_writer.writerow(row)
            writer.write(frame)

            frame_idx += 1

    cap.release()
    writer.release()
    csv_file.close()

    print(f"口唇ランドマークCSV: {csv_path}")
    print(f"口唇ランドマーク動画（無音）: {out_video_path}")


def merge_lips_video_with_original_audio(lips_video_path: str,
                                         original_video_path: str,
                                         out_path: str) -> None:
    """
    ④ MediaPipe 処理後の「映像のみ」動画に、
       元動画に含まれる音声トラックを載せ直す（ffmpeg）。
    """
    cmd = [
        "ffmpeg",
        "-y",
        "-i", lips_video_path,        # 0: MediaPipe 処理済みの映像
        "-i", original_video_path,    # 1: 元動画（ここから音声を取る）
        "-c:v", "copy",
        "-c:a", "aac",
        "-map", "0:v:0",              # 映像は 0 番入力の 0 番ビデオ
        "-map", "1:a:0",              # 音声は 1 番入力の 0 番オーディオ
        "-shortest",
        out_path,
    ]
    subprocess.run(cmd, check=True)
    print(f"MediaPipe処理＋元音声付き動画: {out_path}")


def main():
    video_path = VIDEO_PATH

    # ① 周波数解析＋ノイズ除去
    audio_path = run_sound_analysis(video_path)
    print(f"[1] 周波数解析＆denoise音声: {audio_path}")

    # ② Whisper 文字起こし（denoise 後の音声を使用）
    run_whisper(audio_path, WHISPER_TXT_PATH)
    print(f"[2] Whisper文字起こし結果: {WHISPER_TXT_PATH}")

    # ③ MediaPipe 口唇ランドマーク（映像のみの動画と CSV）
    run_mediapipe_lips(
        video_path,
        LIPS_CSV_PATH,
        LIPS_SILENT_VIDEO_PATH
    )
    print("[3] MediaPipe 口唇解析完了")

    # ④ MediaPipe 処理済み映像 ＋ 元動画の音声 を結合
    if os.path.exists(LIPS_SILENT_VIDEO_PATH) and os.path.exists(video_path):
        merge_lips_video_with_original_audio(
            LIPS_SILENT_VIDEO_PATH,
            video_path,
            LIPS_WITH_AUDIO_PATH
        )
    else:
        print("元動画または MediaPipe 処理動画が存在しないため、音声付き結合をスキップしました。")


if __name__ == "__main__":
    main()
