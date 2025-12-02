import flet as ft

def main(page: ft.Page):
    page.title = "Video Test"

    # 再生したい動画ファイルのパスを指定
    media = ft.VideoMedia("RSResult/session_20251104_172407/video/capture.mp4")

    video = ft.Video(
        playlist=[media],
        playlist_mode=ft.PlaylistMode.NONE,  # 単一動画再生
        aspect_ratio=16 / 9,
        autoplay=False,      # 起動時は止まった状態
        show_controls=True,  # 再生/一時停止ボタン＋シークバーを表示
    )

    play_btn = ft.ElevatedButton("再生", on_click=lambda e: video.play())
    pause_btn = ft.ElevatedButton("一時停止", on_click=lambda e: video.pause())

    page.add(
        ft.Column(
            controls=[
                # 上：動画（余った高さを全部使う）
                ft.Container(
                    content=video,
                    padding=10,
                    expand=True,   # ← これで「Columnの残り高さ」を占有
                    alignment=ft.alignment.center,
                ),
                # 下：ボタン行（必要な高さだけ確保）
                ft.Container(
                    content=ft.Row(
                        [play_btn, pause_btn],
                        alignment=ft.MainAxisAlignment.CENTER,
                    ),
                    padding=10,
                ),
            ],
            expand=True,  # ページ全体を使う
        )
    )

ft.app(target=main)
