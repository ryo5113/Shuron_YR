import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt

# 読み込む PLY ファイル名
PLY_PATH = "PLY/face_3cams_geom_merged_20251203_191142_60deg.ply"


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

    fig, axes = plt.subplots(3, 1, figsize=(6, 12))

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
    plt.show()


if __name__ == "__main__":
    main()
