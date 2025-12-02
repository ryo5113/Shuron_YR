from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A1, A2, A3, A4,landscape, portrait
from reportlab.lib.units import cm, mm

FILE_NAME = './box_grid_calibration.pdf'
# 縦
VERTICAL_SIZE = 14
# 横
HORIZONTAL_SIZE = 9
# 開始位置
START_X = 10.0*mm
START_Y = 10.0*mm
# 正方形のサイズ
RECT_SIZE = 20.0*mm

if __name__ == '__main__':
    # A4縦向き
    pdf_canvas = canvas.Canvas(FILE_NAME, pagesize=portrait(A4))
    # A4横向き
    # pdf_canvas = canvas.Canvas(FILE_NAME, pagesize=landscape(A4))
    pdf_canvas.saveState()

    cnt_flag = True

    X, Y = START_X, START_Y
    # 縦描画
    for i in range(VERTICAL_SIZE):
        # 横描画
        for j in range(HORIZONTAL_SIZE):
            # 白と黒を交互に描画
            pdf_canvas.setFillColorRGB(255, 255, 255) if cnt_flag else pdf_canvas.setFillColorRGB(0, 0, 0)
            pdf_canvas.rect(X, Y, RECT_SIZE, RECT_SIZE, stroke=0, fill=1)
            # X位置をずらす
            X += RECT_SIZE
            # フラグ反転
            cnt_flag = not cnt_flag

        # 偶数の場合は白黒が交互にならないのでフラグを一度反転
        if HORIZONTAL_SIZE % 2 == 0:
            cnt_flag = not cnt_flag

        # X座標開始点に戻す
        X = START_X
        # Y位置をずらす
        Y += RECT_SIZE

    pdf_canvas.restoreState()
    pdf_canvas.save()

