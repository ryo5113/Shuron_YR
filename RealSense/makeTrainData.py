# visualize_ply_voxel.py
from pathlib import Path
import numpy as np
import trimesh
import matplotlib.pyplot as plt

GRID = 30  # faceTrain_SVM.py と同じ:contentReference[oaicite:1]{index=1}

# ★ここだけ、可視化したいPLY 1枚に合わせて変更してください
TARGET_PLY = Path(r"./PLY/YR/mouth/A/mouth_3deg_20260110_183612.ply")

# 出力先
OUT_DIR = Path(r"./PLY/YR/_viz")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def load_points_from_ply(ply_path: Path) -> np.ndarray:
    # faceTrain_SVM.py と同じ読み込み:contentReference[oaicite:2]{index=2}
    geom = trimesh.load(str(ply_path), process=False)
    if hasattr(geom, "vertices") and geom.vertices is not None:
        pts = np.asarray(geom.vertices, dtype=np.float64)
    elif hasattr(geom, "points") and geom.points is not None:
        pts = np.asarray(geom.points, dtype=np.float64)
    else:
        raise ValueError(f"Unsupported PLY content: {ply_path}")

    pts = pts[:, :3]
    pts = pts[np.isfinite(pts).all(axis=1)]
    if len(pts) == 0:
        raise ValueError(f"No valid points in {ply_path}")
    return pts

def points_to_occ_and_norm(points: np.ndarray, grid: int):
    pts = points.astype(np.float64, copy=True)

    mean = pts.mean(axis=0, keepdims=True)   # ★追加：中心
    pts -= mean

    max_abs = np.max(np.abs(pts))            # ★保持：スケール
    if max_abs > 0:
        pts /= max_abs

    pts = np.clip(pts, -1.0, 1.0)

    idx = ((pts + 1.0) * 0.5 * grid).astype(np.int64)
    idx = np.clip(idx, 0, grid - 1)

    occ = np.zeros((grid, grid, grid), dtype=np.uint8)
    occ[idx[:, 0], idx[:, 1], idx[:, 2]] = 1

    return pts, occ, mean.reshape(3), float(max_abs)  # ★変更

def pts_to_grid_coords(pts_norm: np.ndarray, grid: int) -> np.ndarray:
    """
    [-1,1] 空間の点群を、可視化用に [0,grid] へ写像
    （idx計算と整合する表示用座標）
    """
    return (pts_norm + 1.0) * 0.5 * grid

def draw_surface_grid(ax, grid: int, lw: float = 0.2):
    """
    30×30×30 の「グリッドを残す」ため、立方体の6面に格子線を描く
    """
    g = grid
    # x方向の線（y,z固定の面上）
    for i in range(g + 1):
        # z=0, z=g 面
        ax.plot([0, g], [i, i], [0, 0], linewidth=lw)
        ax.plot([0, g], [i, i], [g, g], linewidth=lw)
        ax.plot([0, g], [0, 0], [i, i], linewidth=lw)
        ax.plot([0, g], [g, g], [i, i], linewidth=lw)

        # x=0, x=g 面（y-z格子）
        ax.plot([0, 0], [0, g], [i, i], linewidth=lw)
        ax.plot([g, g], [0, g], [i, i], linewidth=lw)
        ax.plot([0, 0], [i, i], [0, g], linewidth=lw)
        ax.plot([g, g], [i, i], [0, g], linewidth=lw)

def draw_surface_grid_raw(ax, grid: int, center: np.ndarray, half: float, lw: float = 0.2):
    xs = center[0] + np.linspace(-half, half, grid + 1)
    ys = center[1] + np.linspace(-half, half, grid + 1)
    zs = center[2] + np.linspace(-half, half, grid + 1)

    x0, x1 = xs[0], xs[-1]
    y0, y1 = ys[0], ys[-1]
    z0, z1 = zs[0], zs[-1]

    for i in range(grid + 1):
        y = ys[i]
        z = zs[i]
        x = xs[i]

        # z = z0 / z1 面
        ax.plot([x0, x1], [y, y], [z0, z0], linewidth=lw)
        ax.plot([x0, x1], [y, y], [z1, z1], linewidth=lw)

        # y = y0 / y1 面
        ax.plot([x0, x1], [y0, y0], [z, z], linewidth=lw)
        ax.plot([x0, x1], [y1, y1], [z, z], linewidth=lw)

        # x = x0 / x1 面（y-z格子）
        ax.plot([x0, x0], [y0, y1], [z, z], linewidth=lw)
        ax.plot([x1, x1], [y0, y1], [z, z], linewidth=lw)
        ax.plot([x0, x0], [y, y], [z0, z1], linewidth=lw)
        ax.plot([x1, x1], [y, y], [z0, z1], linewidth=lw)

def setup_3d_axes(ax, grid: int, title: str,
                  title_fs=48, label_fs=30, tick_fs=50,
                  labelpad=22, tickpad=14, ticks=(0, 15, 30)):
    ax.set_title(title, fontsize=title_fs, pad=18)

    ax.set_xlim(0, grid); ax.set_ylim(0, grid); ax.set_zlim(0, grid)

    # 軸ラベル（値と重ならないよう labelpad を大きく）
    ax.set_xlabel("X", fontsize=label_fs, labelpad=labelpad)
    ax.set_ylabel("Y", fontsize=label_fs, labelpad=labelpad)
    ax.set_zlabel("Z", fontsize=label_fs, labelpad=labelpad)

    # 目盛り数を減らす（fontsize=30でも重なりにくくする）
    ax.set_xticks(list(ticks))
    ax.set_yticks(list(ticks))
    ax.set_zticks(list(ticks))

    # 目盛り（tick label）を大きく＆軸から離す
    ax.tick_params(axis="x", labelsize=tick_fs, pad=tickpad)
    ax.tick_params(axis="y", labelsize=tick_fs, pad=tickpad)
    ax.zaxis.set_tick_params(labelsize=tick_fs, pad=tickpad)

    try:
        ax.set_box_aspect((1, 1, 1))
    except Exception:
        pass

def setup_3d_axes_raw(
    ax, grid: int, title: str,
    center: np.ndarray, half: float,
    title_fs=56, label_fs=40, tick_fs=30,
    labelpad=26, tickpad=18, ticks_idx=(0, 15, 30), fmt="{:.2f}"
):
    ax.set_title(title, fontsize=title_fs, pad=18)

    # 軸範囲：raw座標（mean±max_abs）
    ax.set_xlim(center[0] - half, center[0] + half)
    ax.set_ylim(center[1] - half, center[1] + half)
    ax.set_zlim(center[2] - half, center[2] + half)

    # 軸ラベル
    ax.set_xlabel("X", fontsize=label_fs, labelpad=labelpad)
    ax.set_ylabel("Y", fontsize=label_fs, labelpad=labelpad)
    ax.set_zlabel("Z", fontsize=label_fs, labelpad=labelpad)

    # ticks（0/15/30など）を raw 座標に変換して位置を設定
    ticks_idx = np.array(list(ticks_idx), dtype=float)
    ns = (ticks_idx / grid) * 2.0 - 1.0  # [-1,1]
    xt = center[0] + ns * half
    yt = center[1] + ns * half
    zt = center[2] + ns * half

    ax.set_xticks(xt); ax.set_yticks(yt); ax.set_zticks(zt)
    ax.set_xticklabels([fmt.format(v) for v in xt])
    ax.set_yticklabels([fmt.format(v) for v in yt])
    ax.set_zticklabels([fmt.format(v) for v in zt])

    # tick文字サイズ＋軸から離す（重なり対策）
    ax.tick_params(axis="x", labelsize=tick_fs, pad=tickpad)
    ax.tick_params(axis="y", labelsize=tick_fs, pad=15)
    ax.zaxis.set_tick_params(labelsize=tick_fs, pad=tickpad)

    try:
        ax.set_box_aspect((1, 1, 1))
    except Exception:
        pass

def save_point_only(pts_grid: np.ndarray, grid: int, out_path: Path):
    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.view_init(elev=20, azim=120)  # 例：既定の向きと逆側になるように調整（azimを180°側へ）
    setup_3d_axes(ax, grid, "Point cloud (normalized) + 30x30x30 grid")
    draw_surface_grid(ax, grid, lw=0.2)
    ax.scatter(pts_grid[:, 0], pts_grid[:, 1], pts_grid[:, 2], s=1)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)

def save_voxel_only(occ: np.ndarray, grid: int, out_path: Path):
    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.view_init(elev=20, azim=120)  # 例：既定の向きと逆側になるように調整（azimを180°側へ）
    setup_3d_axes(ax, grid, "Occupancy voxels (30x30x30)")
    filled = (occ > 0)
    # 立方体（voxel）で表示。エッジも描く＝グリッドが見える
    ax.voxels(filled, edgecolor="k", linewidth=0.2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)

def save_side_by_side(pts_grid: np.ndarray, occ: np.ndarray, grid: int, out_path: Path):
    fig = plt.figure(figsize=(24, 12))  # Word貼り付け用に大きめ

    ax1 = fig.add_subplot(121, projection="3d")
    ax1.view_init(elev=20, azim=120)
    setup_3d_axes(ax1, grid, "Point Cloud",
                  title_fs=56, label_fs=45, tick_fs=45,
                  labelpad=26, tickpad=16, ticks=(0, 10, 20, 30))
    draw_surface_grid(ax1, grid, lw=0.2)
    ax1.scatter(pts_grid[:, 0], pts_grid[:, 1], pts_grid[:, 2], s=1)

    ax2 = fig.add_subplot(122, projection="3d")
    ax2.view_init(elev=20, azim=120)
    setup_3d_axes(ax2, grid, "Occupancy Voxels",
                  title_fs=56, label_fs=45, tick_fs=45,
                  labelpad=26, tickpad=16, ticks=(0, 10, 20, 30))
    ax2.voxels((occ > 0), edgecolor="k", linewidth=0.2)

    # tight_layout だと3Dは詰まりやすいので、余白を固定で確保
    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.06, top=0.88, wspace=0.06)

    fig.savefig(out_path, dpi=300)
    plt.close(fig)

def save_side_by_side_raw_and_voxel(
    pts_raw_rot: np.ndarray, occ_rot: np.ndarray, grid: int,
    center_rot: np.ndarray, half: float, out_path: Path
):
    fig = plt.figure(figsize=(24, 14))

    # ---- Left: raw point cloud with raw axes + 30x30x30 grid overlay ----
    ax1 = fig.add_subplot(121, projection="3d")
    ax1.view_init(elev=20, azim=120)
    setup_3d_axes_raw(
        ax1, grid, "Point Cloud (raw scale)",
        center=center_rot, half=half,
        title_fs=56, label_fs=50, tick_fs=50,
        labelpad=50, tickpad=32, ticks_idx=(0, 10, 20, 30), fmt="{:.2f}"
    )

    draw_surface_grid_raw(ax1, grid, center_rot, half, lw=0.2)
    ax1.scatter(pts_raw_rot[:, 0], pts_raw_rot[:, 1], pts_raw_rot[:, 2], s=1)

    # ---- Right: voxel grid (0..30) but tick labels in raw coordinates ----
    ax2 = fig.add_subplot(122, projection="3d")
    ax2.view_init(elev=20, azim=120)
    setup_3d_axes(
        ax2, grid, "Occupancy Voxels (30×30×30)",
        title_fs=56, label_fs=50, tick_fs=50,
        labelpad=40, tickpad=16, ticks=(0, 10, 20, 30)
    )

    ax2.voxels((occ_rot > 0), edgecolor="k", linewidth=0.2)
    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.06, top=0.88, wspace=0.06)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)

def main():
    pts_raw = load_points_from_ply(TARGET_PLY)
    pts_norm, occ, mean_raw, max_abs = points_to_occ_and_norm(pts_raw, GRID)

    # --- raw側の回転（x, y, z）→（x, z, -y） ---
    pts_raw_rot = pts_raw.copy()
    pts_raw_rot[:, 1], pts_raw_rot[:, 2] = pts_raw[:, 2], -pts_raw[:, 1]

    mean_rot = mean_raw.copy()
    mean_rot[1], mean_rot[2] = mean_raw[2], -mean_raw[1]

    pts_grid = pts_to_grid_coords(pts_norm, GRID)
    pts_grid_rot = pts_grid.copy()
    pts_grid_rot[:, 1] = pts_grid[:, 2]              # Y' = Z
    pts_grid_rot[:, 2] = GRID - pts_grid[:, 1]       # Z' = (GRID - Y)  ※上下反転

    occ_rot = np.transpose(occ, (0, 2, 1))           # (X,Y,Z) -> (X,Z,Y)
    occ_rot = np.flip(occ_rot, axis=2)               # Z' = GRID - Y に対応（上下反転）

    save_point_only(pts_grid_rot, GRID, OUT_DIR / "point_only.png")
    save_voxel_only(occ_rot, GRID, OUT_DIR / "voxel_only.png")
    save_side_by_side(pts_grid_rot, occ_rot, GRID, OUT_DIR / "side_by_side.png")
    save_side_by_side_raw_and_voxel(
        pts_raw_rot, occ_rot, GRID, mean_rot, max_abs, OUT_DIR / "side_by_side2.png"
    )

    print("saved:", OUT_DIR)

if __name__ == "__main__":
    main()
