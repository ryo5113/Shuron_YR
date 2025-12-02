import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# ==== 設定ここから =====================================================

# 計測で保存された CSV ファイルのパス
CSV_PATH = r"firstmouth/lip_depth_points_3d.csv"  # 自分のパスに変更してください

# 描画したい capture_id を指定（0, 1, 2, ...）
TARGET_CAPTURE_ID = 0

# ==== 設定ここまで =====================================================


def main():
    # CSV読み込み
    df = pd.read_csv(CSV_PATH)

    # 指定した capture_id のデータだけ抽出
    df_cap = df[df["capture_id"] == TARGET_CAPTURE_ID]

    if df_cap.empty:
        print(f"capture_id = {TARGET_CAPTURE_ID} のデータがありません。")
        return

    # 3D座標（カメラ座標系, 単位 mm）
    x = df_cap["x_cam_mm"].values
    y = -df_cap["y_cam_mm"].values
    z = df_cap["z_cam_mm"].values
    landmark_ids = df_cap["landmark_id"].values  # ★ 追加：ID取得

    # # 3Dプロット
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection="3d")

    # ax.scatter(x, y, z, s=30)

    # ax.set_xlabel("X [mm]")  # カメラから見て右方向 (+X)
    # ax.set_ylabel("Y [mm]")  # カメラから見て下方向 (+Y)
    # ax.set_zlabel("Z [mm]")  # カメラから見て前方向（被写体側） (+Z)
    # ax.set_title(f"Lip 3D points (capture_id={TARGET_CAPTURE_ID})")

    # スケールをある程度そろえる（見やすさのため）
    max_range = max(
        x.max() - x.min(),
        y.max() - y.min(),
        z.max() - z.min(),
    )
    if max_range == 0:
        max_range = 1.0

    x_mid = (x.max() + x.min()) / 2.0
    y_mid = (y.max() + y.min()) / 2.0
    z_mid = (z.max() + z.min()) / 2.0

    x_lim = (x_mid - max_range / 2, x_mid + max_range / 2)
    y_lim = (y_mid - max_range / 2, y_mid + max_range / 2)
    z_lim = (z_mid - max_range / 2, z_mid + max_range / 2)

    # 2×2 サブプロット作成
    fig = plt.figure(figsize=(10, 10))
    plt.rcParams["font.size"] = 15

    # 1: XY 平面
    ax_xy = fig.add_subplot(2, 2, 1)
    ax_xy.scatter(x, y)
    for xi, yi, lid in zip(x, y, landmark_ids):  # ★ 各点にIDを表示
        ax_xy.text(xi, yi, str(lid), fontsize=8)
    ax_xy.set_xlabel("X [mm]")
    ax_xy.set_ylabel("Y [mm]")
    ax_xy.set_title(f"XY view (capture_id={TARGET_CAPTURE_ID})")
    ax_xy.set_xlim(x_lim)
    ax_xy.set_ylim(y_lim)
    ax_xy.set_aspect("equal", adjustable="box")

    # 2: XZ 平面
    ax_xz = fig.add_subplot(2, 2, 2)
    ax_xz.scatter(x, z)
    for xi, zi, lid in zip(x, z, landmark_ids):  # ★ 各点にIDを表示
        ax_xz.text(xi, zi, str(lid), fontsize=8)
    ax_xz.set_xlabel("X [mm]")
    ax_xz.set_ylabel("Z [mm]")
    ax_xz.set_title("XZ view")
    ax_xz.set_xlim(x_lim)
    ax_xz.set_ylim(z_lim)
    ax_xz.set_aspect("equal", adjustable="box")

    # 3: YZ 平面
    ax_yz = fig.add_subplot(2, 2, 3)
    ax_yz.scatter(z, y)
    for zi, yi, lid in zip(z, y, landmark_ids):  # ★ 各点にIDを表示
        ax_yz.text(zi, yi, str(lid), fontsize=8)
    ax_yz.set_ylabel("Y [mm]")
    ax_yz.set_xlabel("Z [mm]")
    ax_yz.set_title("ZY view")
    ax_yz.set_ylim(y_lim)
    ax_yz.set_xlim(z_lim)
    ax_yz.set_aspect("equal", adjustable="box")

    # 4: 3D 回転可能ビュー
    ax_3d = fig.add_subplot(2, 2, 4, projection="3d")
    ax_3d.scatter(x, y, z, s=30)
    for xi, yi, zi, lid in zip(x, y, z, landmark_ids):  # ★ 各点にIDを表示
        ax_3d.text(xi, yi, zi, str(lid), fontsize=8)
    ax_3d.set_xlabel("X [mm]")
    ax_3d.set_ylabel("Y [mm]")
    ax_3d.set_zlabel("Z [mm]")
    ax_3d.set_title("3D view (rotatable)")

    ax_3d.set_xlim(x_lim)
    ax_3d.set_ylim(y_lim)
    ax_3d.set_zlim(z_lim)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
