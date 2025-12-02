import pandas as pd
import numpy as np
import open3d as o3d

# ==== 設定 ====
CSV_PATH = r"session_20251121_135706/logs/lip_depth_points_3d.csv"  # 実際のパスに変更
TARGET_CAPTURE_ID = 0
# ==============

df = pd.read_csv(CSV_PATH)
df_cap = df[df["capture_id"] == TARGET_CAPTURE_ID]

if df_cap.empty:
    print(f"capture_id = {TARGET_CAPTURE_ID} のデータがありません。")
else:
    # mm → m にして Open3D に渡す（そのままmmでもよいが、一般的にはm）
    points_m = df_cap[["x_cam_mm", "y_cam_mm", "z_cam_mm"]].values / 1000.0

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_m)

    # 3Dビューアで表示（マウスで回転・拡大縮小など）
    o3d.visualization.draw_geometries([pcd])

    # PLYとして保存（他ソフトや別スクリプトからも読み込み可能）
    o3d.io.write_point_cloud("lip_capture0.ply", pcd)
    print("Saved point cloud to lip_capture0.ply")
