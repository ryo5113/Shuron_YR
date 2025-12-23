import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# =========================
# 手動設定（ここだけ編集）
# =========================
PLY_PATH = "PLY/ml/mouth/mouth_-20deg_20251223_200401.ply"

# PLYごとの角度（あなたの元スクリプト踏襲）
ANGLE_DEG = -20.0

# 回転中心（カメラから顔中心までの距離 [m]）
PIVOT_Z = 0.6

# 学習データ保存先（ラベルフォルダの“親”）
# 例: ML2D/2type/U, ML2D/2type/NotU を作りたい → OUTPUT_DATASET_ROOT="ML2D/2type"
OUTPUT_DATASET_ROOT = "PLY/ML2D/2type"
LABEL_NAME = "U"   # "U" or "NotU"

# 画像保存の見た目（元 PLYgraph.py を踏襲）
POINT_SIZE = 0.5
DPI = 200

# 軸範囲（あなたの PLYgraph.py の設定を踏襲して必要なら調整）
XY_XLIM = (-0.08, 0.07)
XY_YLIM = (-0.15, 0.0)
XZ_XLIM = (-0.08, 0.07)
XZ_YLIM = (0.45, 0.6)
ZY_XLIM = (0.45, 0.6)
ZY_YLIM = (-0.1, 0.0)
# =========================


def rotate_points_about_pivot_y(points: np.ndarray, angle_deg: float, pivot_z: float) -> np.ndarray:
    theta = np.deg2rad(angle_deg)
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)

    # Y軸まわりの回転行列（右手系）
    R_y = np.array([
        [ cos_t, 0.0, sin_t],
        [ 0.0,  1.0, 0.0 ],
        [-sin_t, 0.0, cos_t]
    ], dtype=np.float64)

    pivot = np.array([0.0, 0.0, pivot_z], dtype=np.float64)

    # pivot中心回転：pivotへ平行移動→回転→戻す
    points_centered = points - pivot
    points_rot = (R_y @ points_centered.T).T
    return points_rot + pivot


def save_view_scatter(x, y, colors, xlim, ylim, title, out_path: Path):
    fig = plt.figure(figsize=(6, 6), dpi=DPI)
    ax = fig.add_subplot(111)

    ax.scatter(x, y, c=colors, s=POINT_SIZE)
    ax.set_xlim(list(xlim))
    ax.set_ylim(list(ylim))
    ax.set_title(title)
    ax.grid(alpha=0.2)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(out_path), dpi=DPI)
    plt.close(fig)


def main():
    ply_path = Path(PLY_PATH)

    # PLY から点群を読み込み
    pcd = o3d.io.read_point_cloud(str(ply_path))
    print(pcd)
    o3d.visualization.draw_geometries([pcd])
    points = np.asarray(pcd.points)   # (N,3)
    colors = np.asarray(pcd.colors)   # (N,3) 0〜1 の RGB

    if points.size == 0:
        print("点群が空です。PLY の内容を確認してください。")
        return

    # 角度補正（pivot中心のY回転）
    points = rotate_points_about_pivot_y(points, ANGLE_DEG, PIVOT_Z)

    # 出力先
    out_dir = Path(OUTPUT_DATASET_ROOT) / LABEL_NAME
    stem = ply_path.stem

    # 1) XY（元スクリプト踏襲：X反転）
    out_xy = out_dir / f"{stem}__XY.png"
    save_view_scatter(
        x=-points[:, 0],
        y=points[:, 1],
        colors=colors,
        xlim=XY_XLIM,
        ylim=XY_YLIM,
        title="mouth shape(XY)",
        out_path=out_xy
    )

    # 2) XZ（元スクリプト踏襲：X反転）
    out_xz = out_dir / f"{stem}__XZ.png"
    save_view_scatter(
        x=-points[:, 0],
        y=points[:, 2],
        colors=colors,
        xlim=XZ_XLIM,
        ylim=XZ_YLIM,
        title="mouth shape(XZ)",
        out_path=out_xz
    )

    # 3) ZY
    out_zy = out_dir / f"{stem}__ZY.png"
    save_view_scatter(
        x=points[:, 2],
        y=points[:, 1],
        colors=colors,
        xlim=ZY_XLIM,
        ylim=ZY_YLIM,
        title="mouth shape(ZY)",
        out_path=out_zy
    )

    print("[SAVE]", out_xy)
    print("[SAVE]", out_xz)
    print("[SAVE]", out_zy)


if __name__ == "__main__":
    main()
