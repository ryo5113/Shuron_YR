import flet as ft

import soundPredict2
import fletMouthPredict_with_plybutton

def main(page: ft.Page):
    page.title = "統合アプリ"
    page.window_width = 1080
    page.window_height = 720
    page.horizontal_alignment = ft.CrossAxisAlignment.CENTER
    page.vertical_alignment = ft.MainAxisAlignment.CENTER
    all_buttons = []

    def apply_responsive_layout(w: float, h: float):
        # 例：画面幅に応じてボタン幅/高さ/文字サイズを調整
        bw = max(220, int(w * 0.22))   # ボタン幅
        bh = max(50, int(h * 0.08))    # ボタン高さ
        fs = max(14, int(min(w, h) * 0.02))  # 文字サイズ

        for b in all_buttons:
            b.width = bw
            b.height = bh
            # ボタン内文字のサイズ調整（Text を content にしている場合）
            if isinstance(b.content, ft.Text):
                b.content.size = fs

        page.update()

    def show_root_home():
        # 統合ホーム画面
        page.controls.clear()
        btn_sound = ft.ElevatedButton(
            content=ft.Text("発音評価はこちら"),
            on_click=lambda _: soundPredict2.main(page, root_home=show_root_home),
        )
        btn_mouth = ft.ElevatedButton(
            content=ft.Text("口形状評価はこちら"),
            on_click=lambda _: fletMouthPredict_with_plybutton.main(page, root_home=show_root_home),
        )
        all_buttons.clear()
        all_buttons.extend([btn_sound, btn_mouth])

        page.add(
            ft.Column(
                expand=True,
                alignment=ft.MainAxisAlignment.CENTER,                # 縦方向中央
                horizontal_alignment=ft.CrossAxisAlignment.CENTER,    # 横方向中央
                spacing=20,
                controls=[
                    ft.Text("モード選択", size=24, weight=ft.FontWeight.BOLD),
                    ft.Row([btn_sound, btn_mouth], alignment=ft.MainAxisAlignment.CENTER),
                ],
            )
        )

        apply_responsive_layout(page.width, page.height)

    show_root_home()

if __name__ == "__main__":
    ft.app(target=main)
