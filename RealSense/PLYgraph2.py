import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ==========================
# ここに重ねて表示したい PLY を列挙してください
# ==========================
PLY_PATHS = [
    # "PLY_dataset_YR/A/mouth_3deg_20260110_183612.ply",
    # "PLY_dataset_1/A/mouth_5deg_20260110_144746.ply",
    # "PLY_dataset_2/A/mouth_-2deg_20260112_130908.ply",
    # "PLY_dataset_3v2/A/mouth_3deg_20260113_173246.ply",
    "PLY_dataset_4/A/mouth_-3deg_20260113_131032.ply",
    # "PLY_dataset_YR/I/mouth_3deg_20260110_184650.ply",
    # "PLY_dataset_1/I/mouth_3deg_20260110_145120.ply",
    # "PLY_dataset_2/I/mouth_-2deg_20260112_131635.ply",
    # "PLY_dataset_3v2/I/mouth_0deg_20260113_173949.ply",
    "PLY_dataset_4/I/mouth_4deg_20260113_132138.ply",
    # "PLY_dataset_YR/U/mouth_2deg_20260110_181941.ply",
    # "PLY_dataset_1/U/mouth_3deg_20260110_145616.ply",
    # "PLY_dataset_2/U/mouth_-1deg_20260112_132225.ply",
    # "PLY_dataset_3v2/U/mouth_1deg_20260113_174609.ply",
    "PLY_dataset_4/U/mouth_-3deg_20260113_132739.ply",
    # "PLY_dataset_YR/E/mouth_3deg_20260110_184953.ply",
    # "PLY_dataset_1/E/mouth_3deg_20260110_150635.ply",
    # "PLY_dataset_2/E/mouth_-1deg_20260112_133031.ply",
    # "PLY_dataset_3v2/E/mouth_3deg_20260113_175148.ply",
    "PLY_dataset_4/E/mouth_-3deg_20260113_133820.ply",
    # "PLY_dataset_YR/O/mouth_4deg_20260110_185616.ply",
    # "PLY_dataset_1/O/mouth_3deg_20260110_151242.ply",
    # "PLY_dataset_2/O/mouth_-1deg_20260112_133619.ply",
    # "PLY_dataset_3v2/O/mouth_2deg_20260113_175707.ply",
    "PLY_dataset_4/O/mouth_-3deg_20260113_134548.ply",
]

# 3投影の表示範囲（元スクリプトの値を踏襲）
XLIM = (-0.1, 0.1)
YLIM = (-0.1, 0.1)

def load_points(ply_path: str) -> np.ndarray:
    pcd = o3d.io.read_point_cloud(ply_path)
    points = np.asarray(pcd.points)  # (N, 3)
    return points

def main():
    plt.rcParams["font.size"] = 20

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # PLYごとに色を変える（Matplotlibの標準カラーマップ）
    cmap = plt.get_cmap("tab10")

    plotted_any = False
    all_points = []

    for i, ply_path in enumerate(PLY_PATHS):
        points = load_points(ply_path)
        mins = points.min(axis=0)          # [minX, minY, minZ]
        maxs = points.max(axis=0)          # [maxX, maxY, maxZ]
        diffs = maxs - mins                # [rangeX, rangeY, rangeZ]

        print(f"  X: min={mins[0]:.6f}, max={maxs[0]:.6f}, diff={diffs[0]:.6f}")
        print(f"  Y: min={mins[1]:.6f}, max={maxs[1]:.6f}, diff={diffs[1]:.6f}")
        print(f"  Z: min={mins[2]:.6f}, max={maxs[2]:.6f}, diff={diffs[2]:.6f}")

        if points.size == 0:
            print(f"[skip] 点群が空です: {ply_path}")
            continue

        all_points.append(points)

        plotted_any = True
        label = Path(ply_path).name
        color = cmap(i % 10)

        # 1: XY
        axes[0].scatter(-points[:, 0], points[:, 1], s=0.5, alpha=0.7, c=[color], label=label)
        # 2: XZ
        axes[1].scatter(-points[:, 0], points[:, 2], s=0.5, alpha=0.7, c=[color], label=label)
        # 3: ZY
        axes[2].scatter(points[:, 2], points[:, 1], s=0.5, alpha=0.7, c=[color], label=label)

    if not plotted_any:
        print("有効な点群が1つも読み込めませんでした。PLY_PATHS とファイル内容を確認してください。")
        return
    
    if len(all_points) == 0:
        print("有効な点群が1つも読み込めませんでした。")
        return

    all_points = np.vstack(all_points)  # (sumN, 3)

    mins = all_points.min(axis=0)   # [minX, minY, minZ]
    maxs = all_points.max(axis=0)   # [maxX, maxY, maxZ]
    diffs = maxs - mins             # [rangeX, rangeY, rangeZ]

    print("[全点群の範囲]")
    print(f"  X: min={mins[0]:.6f}, max={maxs[0]:.6f}, diff={diffs[0]:.6f}")
    print(f"  Y: min={mins[1]:.6f}, max={maxs[1]:.6f}, diff={diffs[1]:.6f}")
    print(f"  Z: min={mins[2]:.6f}, max={maxs[2]:.6f}, diff={diffs[2]:.6f}")

    # 軸設定（元スクリプト踏襲）
    axes[0].set_xlim(XLIM); axes[0].set_ylim(YLIM)
    axes[0].set_xlabel("X [m]", fontsize=20); axes[0].set_ylabel("Y [m]", fontsize=20)
    axes[0].set_title("mouth shape(XY)")
    axes[0].tick_params(axis="both", labelsize=20); axes[0].grid(alpha=0.2)

    axes[1].set_xlim(XLIM); axes[1].set_ylim(YLIM)
    axes[1].set_xlabel("X [m]", fontsize=20); axes[1].set_ylabel("Z [m]", fontsize=20)
    axes[1].set_title("mouth shape(XZ)")
    axes[1].tick_params(axis="both", labelsize=20); axes[1].grid(alpha=0.2)

    axes[2].set_xlim(XLIM); axes[2].set_ylim(YLIM)
    axes[2].set_xlabel("Z [m]", fontsize=20); axes[2].set_ylabel("Y [m]", fontsize=20)
    axes[2].set_title("mouth shape(ZY)")
    axes[2].tick_params(axis="both", labelsize=20); axes[2].grid(alpha=0.2)

    # 重ね描き対象が複数のため凡例を表示（重なる場合は適宜調整してください）
    for ax in axes:
        ax.legend(fontsize=18, loc="best", labels=["A", "I", "U", "E", "O"], markerscale=10)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
