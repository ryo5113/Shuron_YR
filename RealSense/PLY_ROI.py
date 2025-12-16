import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt

# 読み込む PLY ファイル名
PLY_PATH = "PLY/ply/face_3cams_geom_merged_-20deg_20251216_171202.ply"

# カメラから顔中心までの距離 [m]（頭をその場回転させているという前提）
PIVOT_Z = 0.6

# ========= 追加: ROI（口周辺など）の範囲 [m] =========
# ここは実データを見ながらユーザーが調整してください
ROI_X_MIN, ROI_X_MAX = 0.025, 0.075   # X方向の範囲(負の値に注意)
ROI_Y_MIN, ROI_Y_MAX = -0.08, -0.04   # Y方向の範囲
ROI_Z_MIN, ROI_Z_MAX =  0.48, 0.53   # Z方向の範囲
# =====================================================

def crop_roi(points, colors):
    """
    3DバウンディングボックスでROIを切り出す関数
    points: (N,3)
    colors: (N,3)
    戻り値: (roi_points, roi_colors)
    """
    x, y, z = points[:, 0], points[:, 1], points[:, 2]

    mask = (
        (x >= ROI_X_MIN) & (x <= ROI_X_MAX) &
        (y >= ROI_Y_MIN) & (y <= ROI_Y_MAX) &
        (z >= ROI_Z_MIN) & (z <= ROI_Z_MAX)
    )

    roi_points = points[mask]
    roi_colors = colors[mask]
    return roi_points, roi_colors

def main():
    # PLY から点群を読み込み
    pcd = o3d.io.read_point_cloud(PLY_PATH)
    print(pcd)
    o3d.visualization.draw_geometries([pcd])

    # NumPy 配列に変換
    points = np.asarray(pcd.points)   # 形状: (N, 3)
    colors = np.asarray(pcd.colors)   # 形状: (N, 3) 0〜1 の RGB

    if points.size == 0:
        print("点群が空です。PLY の内容を確認してください。")
        return
    
    # ------------ 撮影角度を PLYごとに設定する ------------
    # 0度撮影: angle_deg = 0.0
    # 20度撮影: angle_deg = 20.0 など
    angle_deg = -20.0
    # ---------------------------------------------------

    theta = np.deg2rad(angle_deg)
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)

    # Y軸まわりの回転行列（右手系）
    R_y = np.array([
        [ cos_t, 0.0, sin_t],
        [ 0.0,  1.0, 0.0 ],
        [-sin_t, 0.0, cos_t]
    ], dtype=np.float64)

    # 回転中心（カメラから PIVOT_Z m 前方）
    pivot = np.array([0.0, 0.0, PIVOT_Z], dtype=np.float64)

    # pivot を中心とした回転
    points_centered = points - pivot
    points_rot = (R_y @ points_centered.T).T
    points = points_rot + pivot

    # ========= 追加: ROIで口周辺などを切り出す =========
    roi_points, roi_colors = crop_roi(points, colors)

    print(f"ROI 内の点数: {roi_points.shape[0]} 点")

    if roi_points.size == 0:
        print("ROI 内に点がありません。ROI の範囲を見直してください。")
        return

    # 外れ点に強いレンジ（例：5%～95%）
    p_lo, p_hi = 5, 95

    # ROI内での「見た目X」に合わせたいなら、ここで -x に統一（描画が -x なので)
    x_vis = -roi_points[:, 0]
    y = roi_points[:, 1]
    z = roi_points[:, 2]

    x_lo, x_hi = np.percentile(x_vis, [p_lo, p_hi])
    y_lo, y_hi = np.percentile(y,     [p_lo, p_hi])
    z_lo, z_hi = np.percentile(z,     [p_lo, p_hi])

    print("=== ROI 内での座標レンジ（percentile） ===")
    print(f"Xvis(-X): {x_lo:.4f} 〜 {x_hi:.4f} [m] (幅 = {x_hi - x_lo:.4f} m)")
    print(f"Y:        {y_lo:.4f} 〜 {y_hi:.4f} [m] (高さ = {y_hi - y_lo:.4f} m)")
    print(f"Z:        {z_lo:.4f} 〜 {z_hi:.4f} [m] (奥行き = {z_hi - z_lo:.4f} m)")
    print("========================================")

    # 以降の描画は「ROIのみ」を描画する例
    points_plot = roi_points
    colors_plot = roi_colors
    pad = 0.01

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    plt.rcParams["font.size"] = 20

    # XY 平面
    axes[0].scatter(-points_plot[:, 0], points_plot[:, 1], c=colors_plot, s=0.5)
    axes[0].set_xlim([x_lo - pad, x_hi + pad])
    axes[0].set_ylim([y_lo - pad, y_hi + pad])
    axes[0].set_xlabel('X [m]', fontsize=20)
    axes[0].set_ylabel('Y [m]', fontsize=20)
    axes[0].set_title('mouth shape(XY)')
    axes[0].tick_params(axis='both', labelsize=20)
    axes[0].grid(alpha=0.2)

    # XZ 平面
    axes[1].scatter(-points_plot[:, 0], points_plot[:, 2], c=colors_plot, s=0.5)
    axes[1].set_xlim([x_lo - pad, x_hi + pad])
    axes[1].set_ylim([z_lo - pad, z_hi + pad])
    axes[1].set_xlabel('X [m]', fontsize=20)
    axes[1].set_ylabel('Z [m]', fontsize=20)
    axes[1].set_title('mouth shape(XZ)')
    axes[1].tick_params(axis='both', labelsize=20)
    axes[1].grid(alpha=0.2)

    # ZY 平面
    axes[2].scatter(points_plot[:, 2], points_plot[:, 1], c=colors_plot, s=0.5)
    axes[2].set_xlim([z_lo - pad, z_hi + pad])
    axes[2].set_ylim([y_lo - pad, y_hi + pad])
    axes[2].set_xlabel('Z [m]', fontsize=20)
    axes[2].set_ylabel('Y [m]', fontsize=20)
    axes[2].set_title('mouth shape(ZY)')
    axes[2].tick_params(axis='both', labelsize=20)
    axes[2].grid(alpha=0.2)

    plt.tight_layout()
    plt.savefig("PLY/ply/U_0deg_3cam_roi.png")
    plt.show()

if __name__ == "__main__":
    main()
