import flet as ft

def main(page: ft.Page):
    page.title = "TestApp"
    text = ""
     #HOME画面を作成
    def home_view():
        global text
        text = ft.TextField(label="")
        return ft.View(
            "/",
            [
                text,
                ft.ElevatedButton("ページ1へ", on_click=lambda e: page.go("/page1")),
            ],
        )
    

     #ページ1画面を作成
    def page1_view():
        global text
        
        return ft.View(
            "/page1",
            [
                ft.Text(text.value, size=100),

                ft.ElevatedButton("HOMEへ", on_click=lambda e: page.go("/")),
            ],
        )

     #ページ遷移用の関数
    def route_change(e):
        page.views.clear()
        if page.route == "/":
            page.views.append(home_view())
        elif page.route == "/page1":
            page.views.append(page1_view())
        page.update()

    page.on_route_change = route_change
    page.go("/")   #初期ページはHOME

ft.app(target=main)
