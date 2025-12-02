import os
from pathlib import Path
import flet as ft
import soundAnalysis  # 同一フォルダ内の soundAnalysis.py を import
import soundWhisper
import movieAnalysis  # MediaPipe + 音声結合用

def main(page: ft.Page):
    page.title = "動画解析 + Whisper文字おこし + MediaPipe"

    selected_video_path: str = ""
    analysis_output_dir: str | None = None   # out_denoised_xxx
    denoised_wav_path: str | None = None    # out_denoised_xxx/audio_denoised.wav
    analysis_done: bool = False

    # MediaPipe 関連の出力パス
    lips_output_dir: str | None = None
    lips_csv_path: str | None = None
    lips_silent_video_path: str | None = None
    lips_with_audio_path: str | None = None

    status_text = ft.Text("動画ファイルを選択してください。")
    whisper_result = ft.Text("", selectable=True)

    # 振動スペクトル画像表示用 Image コントロール
    spectrum_image = ft.Image(width=600, height=300)

    # MediaPipe 適用後動画表示用
    mediapipe_video = ft.Video()

    # ---------- FilePicker（動画選択） ----------
    def on_video_picked(e: ft.FilePickerResultEvent):
        nonlocal selected_video_path, analysis_output_dir, denoised_wav_path, analysis_done
        nonlocal lips_output_dir, lips_csv_path, lips_silent_video_path, lips_with_audio_path

        whisper_result.value = ""
        whisper_result.update()

        # MediaPipe 関連もリセット
        lips_output_dir = None
        lips_csv_path = None
        lips_silent_video_path = None
        lips_with_audio_path = None
        mediapipe_video.src = None
        mediapipe_video.update()

        if e.files:
            selected_video_path = e.files[0].path
            status_text.value = f"選択中の動画: {selected_video_path}"
            # 動画を変えたので、解析フラグをリセット
            analysis_output_dir = None
            denoised_wav_path = None
            analysis_done = False
        else:
            selected_video_path = ""
            status_text.value = "動画ファイルが選択されていません。"

        status_text.update()

    file_picker = ft.FilePicker(on_result=on_video_picked)
    page.overlay.append(file_picker)

    # ---------- 解析ボタン ----------
    def run_analysis(e):
        nonlocal selected_video_path, analysis_output_dir, denoised_wav_path, analysis_done

        if not selected_video_path:
            status_text.value = "先に動画ファイルを選択してください。"
            status_text.update()
            return

        # out_denoised_元ファイル名 というフォルダ名にする
        base = os.path.splitext(os.path.basename(selected_video_path))[0]
        analysis_output_dir = f"out_denoised_{base}"
        denoised_wav_path = str(Path(analysis_output_dir) / "audio_denoised.wav")

        # soundAnalysis の設定を上書き
        soundAnalysis.TARGET_VIDEO = selected_video_path
        soundAnalysis.OUTPUT_DIR = analysis_output_dir

        status_text.value = "解析中です…（少し時間がかかります）"
        status_text.update()

        # 周波数解析＋ノイズ除去などを実行
        soundAnalysis.main()

        # audio_denoised.wav ができているか確認
        if not os.path.isfile(denoised_wav_path):
            status_text.value = f"解析は終了しましたが、音声ファイルが見つかりません: {denoised_wav_path}"
            analysis_done = False
        else:
            status_text.value = (
                "解析が完了しました。\n"
                f"出力フォルダ: {os.path.abspath(analysis_output_dir)}\n"
                f"Whisper用音声: {denoised_wav_path}"
            )
            analysis_done = True

            # 振動スペクトル画像を表示
            img_path = str(Path(analysis_output_dir) / "amplitude_spectrum_denoised.png")
            if os.path.isfile(img_path):
                spectrum_image.src = img_path
                spectrum_image.update()
            else:
                spectrum_image.src = None
                spectrum_image.update()

        status_text.update()

    # ---------- 文字おこしボタン ----------
    def run_whisper(e):
        nonlocal analysis_done, denoised_wav_path, analysis_output_dir

        if not analysis_done or not denoised_wav_path or not os.path.isfile(denoised_wav_path):
            status_text.value = "先に「解析」を実行して、音声ファイルを生成してください。"
            status_text.update()
            return

        # 出力テキストファイル（解析フォルダ内に置く）
        base = os.path.splitext(os.path.basename(denoised_wav_path))[0]
        out_txt = str(Path(analysis_output_dir) / f"{base}_whisper.txt")

        # soundWhisper の設定を上書き
        soundWhisper.AUDIO_PATH = denoised_wav_path
        soundWhisper.OUTPUT_PATH = out_txt

        status_text.value = "Whisperで文字おこし中です…（時間がかかる場合があります）"
        status_text.update()

        soundWhisper.main()

        # 結果の読み込み
        try:
            with open(out_txt, "r", encoding="utf-8") as f:
                text = f.read()
        except Exception as ex:
            whisper_result.value = f"結果ファイルの読み込みに失敗しました: {ex}"
        else:
            whisper_result.value = text or "(空の結果)"

        whisper_result.update()
        status_text.value = f"文字おこしが完了しました。出力: {os.path.abspath(out_txt)}"
        status_text.update()

    # ---------- MediaPipe適用ボタン ----------
    def run_mediapipe(e):
        nonlocal selected_video_path, lips_output_dir, lips_csv_path, lips_silent_video_path, lips_with_audio_path

        if not selected_video_path:
            status_text.value = "先に動画ファイルを選択してください。"
            status_text.update()
            return

        base = os.path.splitext(os.path.basename(selected_video_path))[0]

        # 出力フォルダ（解析フォルダがあればそれを流用、なければ新規作成）
        if analysis_output_dir:
            lips_output_dir = analysis_output_dir
        else:
            lips_output_dir = f"out_lips_{base}"
        Path(lips_output_dir).mkdir(parents=True, exist_ok=True)

        lips_csv_path = str(Path(lips_output_dir) / f"{base}_lips_norm.csv")
        lips_silent_video_path = str(Path(lips_output_dir) / f"{base}_lips_silent.mp4")
        lips_with_audio_path = str(Path(lips_output_dir) / f"{base}_lips_with_audio.mp4")

        status_text.value = "MediaPipe適用中です…（口唇ランドマーク抽出）"
        status_text.update()

        # ③ MediaPipe 口唇ランドマーク（映像のみ動画と CSV）
        movieAnalysis.run_mediapipe_lips(
            selected_video_path,
            lips_csv_path,
            lips_silent_video_path,
        )

        status_text.value = "MediaPipe適用完了。音声を結合しています…"
        status_text.update()

        # ④ MediaPipe 処理済み映像 ＋ 元動画の音声 を結合
        if os.path.exists(lips_silent_video_path) and os.path.exists(selected_video_path):
            movieAnalysis.merge_lips_video_with_original_audio(
                lips_silent_video_path,
                selected_video_path,
                lips_with_audio_path,
            )

            # ページ内に動画を表示
            mediapipe_video.src = lips_with_audio_path
            mediapipe_video.update()

            status_text.value = (
                "MediaPipe適用 + 音声結合が完了しました。\n"
                f"出力動画: {os.path.abspath(lips_with_audio_path)}\n"
                f"ランドマークCSV: {os.path.abspath(lips_csv_path)}"
            )
        else:
            status_text.value = (
                "MediaPipe処理動画または元動画が存在しないため、音声付き結合をスキップしました。"
            )

        status_text.update()

    # ---------- UI 部品 ----------
    pick_button = ft.ElevatedButton(
        "動画を選択",
        on_click=lambda e: file_picker.pick_files(allow_multiple=False),
    )
    analyze_button = ft.ElevatedButton("解析", on_click=run_analysis)
    whisper_button = ft.ElevatedButton("文字おこし", on_click=run_whisper)
    mediapipe_button = ft.ElevatedButton("MediaPipe適用", on_click=run_mediapipe)

    page.add(
        ft.Column(
            [
                status_text,
                ft.Row([pick_button, analyze_button, whisper_button, mediapipe_button]),
                ft.Text("----- 振動スペクトル -----"),
                spectrum_image,
                ft.Text("----- Whisper文字おこし結果 -----"),
                ft.Container(
                    whisper_result,
                    height=250,
                    padding=10,
                ),
                ft.Text("----- MediaPipe適用動画 -----"),
                mediapipe_video,
            ],
            expand=True,
        )
    )

if __name__ == "__main__":
    ft.app(target=main)
