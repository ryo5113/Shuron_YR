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

    # NumPy 配列に変換（回転前）
    points = np.asarray(pcd.points)   # 形状: (N, 3)
    colors = np.asarray(pcd.colors)   # 形状: (N, 3) 0〜1 の RGB

    if points.size == 0:
        print("点群が空です。PLY の内容を確認してください。")
        return

    # ------------ ここを PLYごとに設定する ------------
    # 0度撮影: angle_deg = 0.0
    # 20度撮影を「0度に戻す」: angle_deg = 20.0 など
    angle_deg = 20.0
    # ---------------------------------------------------

    theta = np.deg2rad(angle_deg)
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)

    # Y軸まわりの回転行列（右手系）
    R_y = np.array([
        [cos_t, 0.0, sin_t],
        [0.0,  1.0, 0.0 ],
        [-sin_t, 0.0, cos_t]
    ], dtype=np.float64)

    # 回転中心（カメラから 0.5 m 前方にあると仮定）
    pivot = np.array([0.0, 0.0, PIVOT_Z], dtype=np.float64)

    # pivot を中心とした回転の 4x4 変換行列を作成
    # x' = R (x - pivot) + pivot = R x + (pivot - R pivot)
    t = pivot - R_y @ pivot
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R_y
    T[:3, 3] = t

    # Open3D の点群に回転＋並進を適用
    pcd.transform(T)

    # 変換後の座標を再取得
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)

    # ===================== 全点群のグラフ描画（従来処理） =====================
    plt.rcParams["font.size"] = 20
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))

    # 1枚目: XY 平面
    axes[0].scatter(-points[:, 0], points[:, 1], c=colors, s=0.5)
    axes[0].set_xlabel('X [m]', fontsize=20)
    axes[0].set_ylabel('Y [m]', fontsize=20)
    axes[0].set_title('mouth shape (XY)')
    axes[0].tick_params(axis='both', labelsize=20)
    axes[0].grid(alpha=0.2)

    # 2枚目: XZ 平面
    axes[1].scatter(-points[:, 0], points[:, 2], c=colors, s=0.5)
    axes[1].set_xlabel('X [m]', fontsize=20)
    axes[1].set_ylabel('Z [m]', fontsize=20)
    axes[1].set_title('mouth shape (XZ)')
    axes[1].tick_params(axis='both', labelsize=20)
    axes[1].grid(alpha=0.2)

    # 3枚目: ZY 平面
    axes[2].scatter(points[:, 2], points[:, 1], c=colors, s=0.5)
    axes[2].set_xlabel('Z [m]', fontsize=20)
    axes[2].set_ylabel('Y [m]', fontsize=20)
    axes[2].set_title('mouth shape (ZY)')
    axes[2].tick_params(axis='both', labelsize=20)
    axes[2].grid(alpha=0.2)

    plt.tight_layout()
    plt.savefig("PLY/U_20deg_yoko_all.png")
    plt.show()

    # ===================== バウンディングボックスで唇領域を抽出 =====================
    print("唇領域を選択するために、Open3Dウィンドウでバウンディングボックスを指定してください（Escで終了）。")
    # 回転後の点群を編集モード付きで表示し、マウスでBBox選択
    # ここでウィンドウが開く。BBox/ポリゴンを描いて C を押すと cropped.json / cropped.ply が保存される
    o3d.visualization.draw_geometries_with_editing([pcd])

    # 保存された selection_polygon_volume を読み込む
    try:
        vol = o3d.visualization.read_selection_polygon_volume("cropped.json")
    except Exception as e:
        print("cropped.json が見つからないか、読み込みに失敗しました:", e)
        return

    # 選択領域内の点だけを抽出
    lip_pcd = vol.crop_point_cloud(pcd)

    lip_points = np.asarray(lip_pcd.points)
    lip_colors = np.asarray(lip_pcd.colors)
    print(f"唇領域の点群数: {lip_points.shape[0]}")

    # 必要なら 3D 表示
    o3d.visualization.draw_geometries([lip_pcd])

    # ===================== 唇領域のみのグラフ描画 =====================
    fig_lip, axes_lip = plt.subplots(1, 3, figsize=(16, 4))
    plt.rcParams["font.size"] = 20

    # XY
    axes_lip[0].scatter(-lip_points[:, 0], lip_points[:, 1], c=lip_colors, s=1.0)
    axes_lip[0].set_xlabel('X [m]', fontsize=20)
    axes_lip[0].set_ylabel('Y [m]', fontsize=20)
    axes_lip[0].set_title('lip region (XY)')
    axes_lip[0].tick_params(axis='both', labelsize=20)
    axes_lip[0].grid(alpha=0.2)

    # XZ
    axes_lip[1].scatter(-lip_points[:, 0], lip_points[:, 2], c=lip_colors, s=1.0)
    axes_lip[1].set_xlabel('X [m]', fontsize=20)
    axes_lip[1].set_ylabel('Z [m]', fontsize=20)
    axes_lip[1].set_title('lip region (XZ)')
    axes_lip[1].tick_params(axis='both', labelsize=20)
    axes_lip[1].grid(alpha=0.2)

    # ZY
    axes_lip[2].scatter(lip_points[:, 2], lip_points[:, 1], c=lip_colors, s=1.0)
    axes_lip[2].set_xlabel('Z [m]', fontsize=20)
    axes_lip[2].set_ylabel('Y [m]', fontsize=20)
    axes_lip[2].set_title('lip region (ZY)')
    axes_lip[2].tick_params(axis='both', labelsize=20)
    axes_lip[2].grid(alpha=0.2)

    plt.tight_layout()
    plt.savefig("PLY/U_20deg_yoko_lip.png")
    plt.show()


if __name__ == "__main__":
    main()
