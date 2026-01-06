# predict_ply_svm_no_cli.py
# 保存済みSVMモデル(joblib) + 入力PLY -> 10x10x10占有特徴 -> 推論

from pathlib import Path
import numpy as np
import trimesh
import joblib


# =========================
# 設定（ここだけ編集）
# =========================
MODEL_PATH = Path(r"PLY_dataset/ply_svm_model.joblib")  
INPUT_PLY  = Path(r"PLY/svm/mouth/U/mouth_-2deg_20260106_172411.ply")  # 例: Path(r"/mnt/data/mouth_-36deg_20260105_162936.ply")
# =========================


def load_points_from_ply(ply_path: Path) -> np.ndarray:
    """
    PLYを読み込み、Nx3 の点群(float)を返す。
    trimesh には PLY をロードする機能がある :contentReference[oaicite:2]{index=2}
    """
    geom = trimesh.load(str(ply_path), process=False)

    # trimesh.Trimesh / trimesh.PointCloud 等を想定
    if hasattr(geom, "vertices") and geom.vertices is not None:
        pts = np.asarray(geom.vertices, dtype=np.float64)
    elif hasattr(geom, "points") and geom.points is not None:
        pts = np.asarray(geom.points, dtype=np.float64)
    else:
        raise ValueError(f"Unsupported PLY content: {ply_path}")

    if pts.ndim != 2 or pts.shape[1] < 3:
        raise ValueError(f"Invalid point array shape {pts.shape} in {ply_path}")

    pts = pts[:, :3]
    pts = pts[np.isfinite(pts).all(axis=1)]
    if len(pts) == 0:
        raise ValueError(f"No valid points in {ply_path}")
    return pts


def occupancy_grid_features(points: np.ndarray, grid: int) -> np.ndarray:
    """
    学習時と同じ前処理:
    - 中心化
    - 最大絶対値でスケール
    - [-1, 1] にクリップ
    - grid^3 の占有(0/1)グリッド -> flatten
    """
    pts = points.astype(np.float64, copy=True)

    pts -= pts.mean(axis=0, keepdims=True)

    max_abs = np.max(np.abs(pts))
    if max_abs > 0:
        pts /= max_abs

    pts = np.clip(pts, -1.0, 1.0)

    idx = ((pts + 1.0) * 0.5 * grid).astype(np.int64)
    idx = np.clip(idx, 0, grid - 1)

    occ = np.zeros((grid, grid, grid), dtype=np.uint8)
    occ[idx[:, 0], idx[:, 1], idx[:, 2]] = 1

    return occ.reshape(-1).astype(np.float64)


def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"MODEL_PATH not found: {MODEL_PATH}")
    if not INPUT_PLY.exists():
        raise FileNotFoundError(f"INPUT_PLY not found: {INPUT_PLY}")

    # joblib.load で保存済みオブジェクトを復元 :contentReference[oaicite:3]{index=3}
    payload = joblib.load(MODEL_PATH)

    if not isinstance(payload, dict) or "pipeline" not in payload:
        raise ValueError("Model file format is unexpected. Expected dict with key 'pipeline'.")

    pipeline = payload["pipeline"]
    label_classes = payload.get("label_classes", None)
    grid = int(payload.get("grid", 10))  # 保存側に grid がある前提（なければ10）

    pts = load_points_from_ply(INPUT_PLY)
    feat = occupancy_grid_features(pts, grid=grid).reshape(1, -1)

    # SVC.predict はクラスラベルを返す :contentReference[oaicite:4]{index=4}
    pred_idx = int(pipeline.predict(feat)[0])

    if label_classes is not None:
        pred_label = str(label_classes[pred_idx])
    else:
        # label_classes が無い場合は整数ラベルのみ
        pred_label = str(pred_idx)

    print("=== Inference ===")
    print(f"model: {MODEL_PATH}")
    print(f"input: {INPUT_PLY}")
    print(f"grid: {grid} -> feature_dim: {grid ** 3}")
    print(f"pred_index: {pred_idx}")
    print(f"pred_label: {pred_label}")

    # 可能なら decision_function も表示（SVCに存在） :contentReference[oaicite:5]{index=5}
    if hasattr(pipeline, "decision_function"):
        try:
            score = pipeline.decision_function(feat)
            print(f"decision_function: {np.asarray(score).ravel()}")
        except Exception:
            pass


if __name__ == "__main__":
    main()
