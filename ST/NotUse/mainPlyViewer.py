# -*- coding: utf-8 -*-
import sys
import os
import numpy as np

from PyQt6 import QtWidgets
from PyQt6.QtGui import QAction
from PyQt6.QtCore import QTimer

import pyqtgraph as pg
import pyqtgraph.opengl as gl
from plyfile import PlyData


# ============================================================
# PLY Select Window
# ============================================================
class PlySelectWindow(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PLY選択")
        self.resize(400, 120)

        layout = QtWidgets.QVBoxLayout(self)
        btn = QtWidgets.QPushButton("PLYを選択して3D表示")
        btn.clicked.connect(self.open)
        layout.addWidget(btn)

    def open(self):
        fname, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "PLY選択", "", "PLY Files (*.ply)"
        )
        if fname:
            self.viewer = PointCloudViewer(fname)
            self.viewer.show()
            self.close()


# ============================================================
# Main Viewer
# ============================================================
class PointCloudViewer(QtWidgets.QMainWindow):

    # ===== 表示スケール（mm）=====
    AXIS_LEN  = 50       # ±50 mm = ±5 cm
    MARK_STEP = 10       # 10 mm = 1 cm
    MARK_SIZE = 6

    # ★ 区別しやすい固定色（色弱配慮）
    COLOR_P1 = (1.0, 0.2, 0.2, 1.0)   # 赤系
    COLOR_P2 = (0.2, 0.6, 1.0, 1.0)   # 青系

    def __init__(self, plyfile):
        super().__init__()
        self.setWindowTitle("PointCloud Viewer (PLY)")
        self.resize(1600, 900)

        self.file1_path = None
        self.file2_path = None

        self.p1_org = None
        self.c1 = None
        self.p2_org = None
        self.c2 = None

        self.use_vertex_color = True
        self.show_p1 = True
        self.show_p2 = True

        # ---------------- Central ----------------
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        main = QtWidgets.QHBoxLayout(central)

        # ---------------- Control Panel ----------------
        ctrl = QtWidgets.QWidget()
        ctrl.setFixedWidth(280)
        cl = QtWidgets.QVBoxLayout(ctrl)

        cl.addWidget(QtWidgets.QLabel("<b>Global Transform</b>"))
        self.g_tx = self.sb(); self.g_ty = self.sb(); self.g_tz = self.sb()
        self.g_rx = self.sb(); self.g_ry = self.sb(); self.g_rz = self.sb()
        for n,s in [
            ("Tx [mm]",self.g_tx),("Ty [mm]",self.g_ty),("Tz [mm]",self.g_tz),
            ("Rx [deg]",self.g_rx),("Ry [deg]",self.g_ry),("Rz [deg]",self.g_rz)
        ]:
            cl.addLayout(self.row(n,s))

        cl.addWidget(QtWidgets.QLabel("<b>2nd Transform</b>"))
        self.s_tx = self.sb(); self.s_ty = self.sb(); self.s_tz = self.sb()
        self.s_rx = self.sb(); self.s_ry = self.sb(); self.s_rz = self.sb()
        for n,s in [
            ("dTx [mm]",self.s_tx),("dTy [mm]",self.s_ty),("dTz [mm]",self.s_tz),
            ("dRx [deg]",self.s_rx),("dRy [deg]",self.s_ry),("dRz [deg]",self.s_rz)
        ]:
            cl.addLayout(self.row(n,s))

        # --- 表示制御ボタン群 ---
        self.btn_color = QtWidgets.QPushButton("色表示：PLYの色情報")
        self.btn_color.clicked.connect(self.toggle_color_mode)
        cl.addWidget(self.btn_color)

        self.btn_p1 = QtWidgets.QPushButton("モデル1：表示中")
        self.btn_p1.clicked.connect(self.toggle_p1)
        cl.addWidget(self.btn_p1)

        self.btn_p2 = QtWidgets.QPushButton("モデル2：表示中")
        self.btn_p2.clicked.connect(self.toggle_p2)
        cl.addWidget(self.btn_p2)

        cl.addStretch()
        main.addWidget(ctrl)

        # ---------------- 3D View ----------------
        self.view = gl.GLViewWidget()
        self.view.opts["distance"] = 180
        self.view.setCameraPosition(azimuth=45, elevation=25)
        main.addWidget(self.view, 1)

        self.add_axes()
        self.add_axis_markers()

        self.pc1 = gl.GLScatterPlotItem()
        self.pc2 = gl.GLScatterPlotItem()
        self.view.addItem(self.pc1)
        self.view.addItem(self.pc2)

        menubar = self.menuBar()
        filemenu = menubar.addMenu("File")
        filemenu.addAction(QAction("Load Second PLY", self, triggered=self.load_second))

        for sb in [
            self.g_tx,self.g_ty,self.g_tz,
            self.g_rx,self.g_ry,self.g_rz,
            self.s_tx,self.s_ty,self.s_tz,
            self.s_rx,self.s_ry,self.s_rz
        ]:
            sb.editingFinished.connect(self.apply_transform)

        self.load_first(plyfile)

    # ========================================================
    # Helper
    def sb(self):
        s = QtWidgets.QDoubleSpinBox()
        s.setRange(-500, 500)
        s.setDecimals(3)
        return s

    def row(self, name, sb):
        l = QtWidgets.QHBoxLayout()
        l.addWidget(QtWidgets.QLabel(name))
        l.addWidget(sb)
        return l

    # ========================================================
    # Axes（完全維持）
    def add_axes(self):
        def add(a,b,c):
            self.view.addItem(gl.GLLinePlotItem(
                pos=np.array([a,b]), color=c, width=3
            ))
        L = self.AXIS_LEN
        add([0,0,0],[ L,0,0],(1,0,0,1))
        add([0,0,0],[-L,0,0],(1,0.5,0,1))
        add([0,0,0],[0, L,0],(0,0.6,0,1))
        add([0,0,0],[0,-L,0],(0.6,1,0,1))
        add([0,0,0],[0,0, L],(0,0,1,1))
        add([0,0,0],[0,0,-L],(0,1,1,1))

    def add_axis_markers(self):
        pts = []
        for v in range(-self.AXIS_LEN, self.AXIS_LEN + 1, self.MARK_STEP):
            pts += [[v,0,0],[0,v,0],[0,0,v]]
        self.view.addItem(gl.GLScatterPlotItem(
            pos=np.array(pts), size=self.MARK_SIZE, color=(1,1,1,0.8)
        ))

    # ========================================================
    # PLY
    def load_ply(self, fname):
        ply = PlyData.read(fname)
        v = ply["vertex"]
        pts = np.column_stack((v["x"], v["y"], v["z"])) * 1000.0
        if {"red","green","blue"}.issubset(v.data.dtype.names):
            col = np.column_stack((v["red"], v["green"], v["blue"])) / 255.0
        else:
            col = None
        return pts, col

    def load_first(self, fname):
        self.file1_path = fname
        self.p1_org, self.c1 = self.load_ply(fname)
        self.apply_transform()

    def load_second(self):
        fname,_ = QtWidgets.QFileDialog.getOpenFileName(
            self,"Second PLY","","PLY Files (*.ply)"
        )
        if fname:
            self.file2_path = fname
            self.p2_org, self.c2 = self.load_ply(fname)
            self.apply_transform()

    # ========================================================
    # Transform
    def rot(self, rx, ry, rz):
        rx, ry, rz = np.deg2rad([rx, ry, rz])
        cx, sx = np.cos(rx), np.sin(rx)
        cy, sy = np.cos(ry), np.sin(ry)
        cz, sz = np.cos(rz), np.sin(rz)
        Rx = np.array([[1,0,0],[0,cx,-sx],[0,sx,cx]])
        Ry = np.array([[cy,0,sy],[0,1,0],[-sy,0,cy]])
        Rz = np.array([[cz,-sz,0],[sz,cz,0],[0,0,1]])
        return Rz @ Ry @ Rx

    def apply_transform(self):
        if self.p1_org is None:
            return

        Gt = np.array([self.g_tx.value(), self.g_ty.value(), self.g_tz.value()])
        GR = self.rot(self.g_rx.value(), self.g_ry.value(), self.g_rz.value())

        # ---- Model 1 ----
        if self.show_p1:
            p1 = (GR @ self.p1_org.T).T + Gt
            if self.use_vertex_color and self.c1 is not None:
                self.pc1.setData(pos=p1, color=self.c1, size=8)
            else:
                self.pc1.setData(pos=p1, color=self.COLOR_P1, size=8)
        else:
            self.pc1.setData(pos=np.empty((0,3)))

        # ---- Model 2 ----
        if self.p2_org is not None:
            if self.show_p2:
                St = np.array([self.s_tx.value(), self.s_ty.value(), self.s_tz.value()])
                SR = self.rot(self.s_rx.value(), self.s_ry.value(), self.s_rz.value())
                p2 = (GR @ ((SR @ self.p2_org.T).T + St).T).T + Gt
                if self.use_vertex_color and self.c2 is not None:
                    self.pc2.setData(pos=p2, color=self.c2, size=8)
                else:
                    self.pc2.setData(pos=p2, color=self.COLOR_P2, size=8)
            else:
                self.pc2.setData(pos=np.empty((0,3)))

    # ========================================================
    # UI Actions
    def toggle_color_mode(self):
        self.use_vertex_color = not self.use_vertex_color
        self.btn_color.setText(
            "色表示：PLYの色情報" if self.use_vertex_color else "色表示：固定色（区別強）"
        )
        self.apply_transform()

    def toggle_p1(self):
        self.show_p1 = not self.show_p1
        self.btn_p1.setText("モデル1：表示中" if self.show_p1 else "モデル1：非表示")
        self.apply_transform()

    def toggle_p2(self):
        self.show_p2 = not self.show_p2
        self.btn_p2.setText("モデル2：表示中" if self.show_p2 else "モデル2：非表示")
        self.apply_transform()


# ============================================================
def main():
    pg.setConfigOptions(antialias=False)
    app = QtWidgets.QApplication(sys.argv)
    w = PlySelectWindow()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
