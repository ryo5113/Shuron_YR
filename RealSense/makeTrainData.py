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
    """
    faceTrain_SVM.py の occupancy_grid_features と同じ前処理で
    - 正規化後点群 pts_norm（[-1,1]）
    - 占有配列 occ（grid,grid,grid）(0/1)
    を返す
    :contentReference[oaicite:3]{index=3}
    """
    pts = points.astype(np.float64, copy=True)

    # center:contentReference[oaicite:4]{index=4}
    pts -= pts.mean(axis=0, keepdims=True)

    # scale:contentReference[oaicite:5]{index=5}
    max_abs = np.max(np.abs(pts))
    if max_abs > 0:
        pts /= max_abs

    # clip:contentReference[oaicite:6]{index=6}
    pts = np.clip(pts, -1.0, 1.0)

    # [-1,1] -> [0, grid-1]:contentReference[oaicite:7]{index=7}
    idx = ((pts + 1.0) * 0.5 * grid).astype(np.int64)
    idx = np.clip(idx, 0, grid - 1)

    # occupancy:contentReference[oaicite:8]{index=8}
    occ = np.zeros((grid, grid, grid), dtype=np.uint8)
    occ[idx[:, 0], idx[:, 1], idx[:, 2]] = 1

    return pts, occ

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

def setup_3d_axes(ax, grid: int, title: str):
    ax.set_title(title)
    ax.set_xlim(0, grid); ax.set_ylim(0, grid); ax.set_zlim(0, grid)
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
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
    fig = plt.figure(figsize=(14, 7))
    plt.rcParams["font.size"] = 25
    ax1 = fig.add_subplot(121, projection="3d")
    ax1.view_init(elev=20, azim=120)  # 例：既定の向きと逆側になるように調整（azimを180°側へ）
    setup_3d_axes(ax1, grid, "Point cloud")
    draw_surface_grid(ax1, grid, lw=0.2)
    ax1.scatter(pts_grid[:, 0], pts_grid[:, 1], pts_grid[:, 2], s=1)

    ax2 = fig.add_subplot(122, projection="3d")
    ax2.view_init(elev=20, azim=120)  # 例：既定の向きと逆側になるように調整（azimを180°側へ）
    setup_3d_axes(ax2, grid, "Occupancy voxels")
    ax2.voxels((occ > 0), edgecolor="k", linewidth=0.2)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)

def main():
    pts_raw = load_points_from_ply(TARGET_PLY)
    pts_norm, occ = points_to_occ_and_norm(pts_raw, GRID)
    pts_grid = pts_to_grid_coords(pts_norm, GRID)
    pts_grid_rot = pts_grid.copy()
    pts_grid_rot[:, 1] = pts_grid[:, 2]              # Y' = Z
    pts_grid_rot[:, 2] = GRID - pts_grid[:, 1]       # Z' = (GRID - Y)  ※上下反転

    occ_rot = np.transpose(occ, (0, 2, 1))           # (X,Y,Z) -> (X,Z,Y)
    occ_rot = np.flip(occ_rot, axis=2)               # Z' = GRID - Y に対応（上下反転）

    save_point_only(pts_grid_rot, GRID, OUT_DIR / "point_only.png")
    save_voxel_only(occ_rot, GRID, OUT_DIR / "voxel_only.png")
    save_side_by_side(pts_grid_rot, occ_rot, GRID, OUT_DIR / "side_by_side.png")

    print("saved:", OUT_DIR)

if __name__ == "__main__":
    main()
