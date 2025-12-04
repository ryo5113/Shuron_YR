import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt

# 読み込む PLY ファイル名
PLY_PATH = "PLY/face_3cams_geom_merged_20251203_191006_20deg.ply"

# カメラから顔中心までの距離 [m]（頭をその場回転させているという前提）
PIVOT_Z = 0.5

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
    
    # ------------ ここを PLYごとに設定する ------------
    # 0度撮影: angle_deg = 0.0
    # 20度撮影: angle_deg = 20.0 
    angle_deg = 20.0
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

    # # 点群を回転(回転のみを考慮)
    # points = (R_y @ points.T).T

    # 回転中心（カメラから 0.5 m 前方にあると仮定）
    pivot = np.array([0.0, 0.0, PIVOT_Z], dtype=np.float64)

    # 「pivot を原点に移動 → 回転 → 元の位置に戻す」で
    # pivot を中心とした回転にする
    points_centered = points - pivot
    points_rot = (R_y @ points_centered.T).T
    points = points_rot + pivot

    fig, axes = plt.subplots(1, 3, figsize=(16, 4))

    plt.rcParams["font.size"] = 20

    # 1段目: XY 平面
    axes[0].scatter(-points[:, 0], points[:, 1], c=colors, s=0.5)
    axes[0].set_xlabel('X [m]', fontsize=20)
    axes[0].set_ylabel('Y [m]', fontsize=20)
    axes[0].set_title('mouth shape(XY)')
    axes[0].tick_params(axis='both', labelsize=20)
    axes[0].grid(alpha=0.2)

    # 2段目: XZ 平面
    axes[1].scatter(-points[:, 0], points[:, 2], c=colors, s=0.5)
    axes[1].set_xlabel('X [m]', fontsize=20)
    axes[1].set_ylabel('Z [m]', fontsize=20)
    axes[1].set_title('mouth shape(XZ)')
    axes[1].tick_params(axis='both', labelsize=20)
    axes[1].grid(alpha=0.2)

    # 3段目: ZY 平面
    axes[2].scatter(points[:, 2], points[:, 1], c=colors, s=0.5)
    axes[2].set_xlabel('Z [m]', fontsize=20)
    axes[2].set_ylabel('Y [m]', fontsize=20)
    axes[2].set_title('mouth shape(ZY)')
    axes[2].tick_params(axis='both', labelsize=20)
    axes[2].grid(alpha=0.2)

    plt.tight_layout()
    plt.savefig("PLY/U_20deg_yoko_v3.png")
    plt.show()

if __name__ == "__main__":
    main()
