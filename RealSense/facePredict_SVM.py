# predict_ply_svm_no_cli.py
# 保存済みSVMモデル(joblib) + 入力PLY -> 10x10x10占有特徴 -> 推論

from pathlib import Path
import numpy as np
import trimesh
import joblib


# =========================
# 設定（ここだけ編集）
# =========================
MODEL_PATH = Path(r"PLY_dataset_YR/ply_svm_model.joblib")  
INPUT_PLY  = Path(r"Testdata/E/pre_E_YR.ply")  # 例: Path(r"/mnt/data/mouth_-36deg_20260105_162936.ply")
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

def softmax(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    x = x - np.max(x)  # 数値安定化
    e = np.exp(x)
    s = e / np.sum(e)
    return s

def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"MODEL_PATH not found: {MODEL_PATH}")
    if not INPUT_PLY.exists():
        raise FileNotFoundError(f"INPUT_PLY not found: {INPUT_PLY}")

    # joblib.load で保存済みオブジェクトを復元
    payload = joblib.load(MODEL_PATH)
    label_names = None
    if isinstance(payload, dict):
        label_names = payload.get("label_order", None)     # GridSearch版の学習スクリプトに対応
        if label_names is None:
            label_names = payload.get("label_classes", None)  # 旧版に対応

    if not isinstance(payload, dict) or "model" not in payload:
        raise ValueError("Model file format is unexpected. Expected dict with key 'model'.")

    model = payload["model"]
    grid = int(payload.get("grid", 10))  # 保存側に grid がある前提（なければ10）

    pts = load_points_from_ply(INPUT_PLY)
    feat = occupancy_grid_features(pts, grid=grid).reshape(1, -1)

    # SVC.predict はクラスラベルを返す
    pred_idx = int(model.predict(feat)[0])

    label_classes = payload.get("label_classes", None)
    if label_classes is not None:
        pred_label = str(label_classes[pred_idx])
    else:
        # label_classes が無い場合は整数ラベルのみ
        pred_label = str(pred_idx)

        # まずクラス予測（整数ラベル）
    pred_idx = int(model.predict(feat)[0])

    # ===== 割合（％）の算出 =====
    # SVCで確率を出すには probability=True が必要。可能なら predict_proba を使う
    probs = None
    scores = None

    if hasattr(model, "predict_proba"):
        try:
            probs = np.asarray(model.predict_proba(feat)[0], dtype=np.float64)
        except Exception:
            probs = None

    if probs is None and hasattr(model, "decision_function"):
        try:
            df = np.asarray(model.decision_function(feat), dtype=np.float64)
            df = df.reshape(-1)

            # 2値分類の decision_function は1つだけ返る場合があるので、その場合は2クラスに拡張
            #（今回あなたは5ラベルなので通常ここは通りません）
            if df.size == 1:
                s = float(df[0])
                scores = np.array([-s, s], dtype=np.float64)
            else:
                scores = df

            probs = softmax(scores)  # “確率”ではなく、スコアを見やすく正規化した割合
        except Exception:
            probs = None

    if probs is None:
        raise RuntimeError("Cannot compute score/probability: model has neither usable predict_proba nor decision_function.")

    # ラベル名を準備（無い場合は 0..n-1）
    n_cls = len(probs)
    if label_names is None:
        label_names = [str(i) for i in range(n_cls)]
    else:
        label_names = [str(x) for x in label_names]

    # top2
    order = np.argsort(probs)[::-1]
    top1, top2 = int(order[0]), int(order[1]) if n_cls >= 2 else (int(order[0]), None)

    # 表示用
    def fmt(i: int) -> str:
        return f"{label_names[i]}: {probs[i]*100:.1f}%"

    print("=== Inference ===")
    print(f"model: {MODEL_PATH}")
    print(f"input: {INPUT_PLY}")
    print(f"grid: {grid} -> feature_dim: {grid ** 3}")

    print(f"pred_label: {label_names[pred_idx]}  ({probs[pred_idx]*100:.1f}%)")

    print("top2:")
    print(f"  1) {fmt(top1)}")
    if top2 is not None:
        print(f"  2) {fmt(top2)}")

    print("all_labels:")
    for i in order:
        print(f"  - {fmt(int(i))}")

    # 参考として decision_function も残したい場合（表示は短く）
    if scores is not None:
        print(f"decision_function (raw): {scores}")

if __name__ == "__main__":
    main()
