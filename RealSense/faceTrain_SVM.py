# PLY点群 -> 10x10x10 占有(0/1) -> flatten(1000次元) -> SVM学習 -> 保存
#
# SVM入力は (n_samples, n_features) の固定長ベクトル 

from pathlib import Path
import numpy as np
import trimesh
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import joblib


# =========================
# 設定（ここだけ編集）
# =========================
DATA_ROOT = Path(r"./PLY_dataset")   # ラベル別フォルダを含むルート
GRID = 10                             # 10 -> 10x10x10 = 1000次元
TEST_SIZE = 0.3
SEED = 42
OUT_MODEL = Path("PLY_dataset/ply_svm_model.joblib")
OUT_CM_PNG = Path("PLY_dataset/confusion_matrix.png") # もし混同行列を画像保存する場合のパス
CM_DPI = 200
# =========================


def load_points_from_ply(ply_path: Path) -> np.ndarray:
    """
    PLYを読み込み、Nx3 の点群を返す。
    trimesh は PLY を含む複数形式をロード可能 :contentReference[oaicite:4]{index=4}
    """
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


def occupancy_grid_features(points: np.ndarray, grid: int) -> np.ndarray:
    """
    点群を中心化+スケールして [-1, 1] に収め、grid^3 の占有(0/1)グリッドにする。
    返り値: (grid^3,) の0/1ベクトル
    """
    pts = points.astype(np.float64, copy=True)

    # center
    pts -= pts.mean(axis=0, keepdims=True)

    # scale
    max_abs = np.max(np.abs(pts))
    if max_abs > 0:
        pts /= max_abs

    # clip
    pts = np.clip(pts, -1.0, 1.0)

    # [-1,1] -> [0, grid-1]
    idx = ((pts + 1.0) * 0.5 * grid).astype(np.int64)
    idx = np.clip(idx, 0, grid - 1)

    occ = np.zeros((grid, grid, grid), dtype=np.uint8)
    occ[idx[:, 0], idx[:, 1], idx[:, 2]] = 1

    return occ.reshape(-1).astype(np.float64)


def collect_dataset(data_root: Path, grid: int):
    if not data_root.exists():
        raise FileNotFoundError(f"DATA_ROOT not found: {data_root}")

    X_list, y_list = [], []

    label_dirs = sorted([p for p in data_root.iterdir() if p.is_dir()], key=lambda p: p.name)
    if len(label_dirs) == 0:
        raise ValueError(f"No label directories under: {data_root}")

    for ld in label_dirs:
        for pf in sorted(ld.glob("*.ply")):
            pts = load_points_from_ply(pf)
            feat = occupancy_grid_features(pts, grid=grid)
            X_list.append(feat)
            y_list.append(ld.name)

    if len(X_list) == 0:
        raise ValueError(f"No PLY files found under: {data_root}")

    X = np.vstack(X_list)
    y = np.array(y_list, dtype=object)
    return X, y


def main():
    X, y_str = collect_dataset(DATA_ROOT, GRID)

    le = LabelEncoder()
    y = le.fit_transform(y_str)

    # 少数データだと stratify が失敗する場合があるので fallback
    try:
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=SEED, stratify=y
        )
    except ValueError:
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=SEED, stratify=None
        )

    # SVC: 入力は (n_samples, n_features) の特徴量 
    clf = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("svc", SVC(kernel="rbf", degree=2, C=5, gamma="scale")),
        ]
    )

    clf.fit(X_tr, y_tr)
    pred = clf.predict(X_te)

    print("=== Dataset ===")
    print(f"DATA_ROOT: {DATA_ROOT}")
    print(f"samples: {len(X)}")
    print(f"labels: {list(le.classes_)}")
    print(f"GRID: {GRID} -> feature_dim: {GRID ** 3}")

    print("=== Eval ===")
    print(f"accuracy: {accuracy_score(y_te, pred):.4f}")
    print("confusion_matrix:")
    cm = confusion_matrix(y_te, pred)
    print(cm)
    print("classification_report:")
    print(classification_report(y_te, pred, target_names=le.classes_))

    # ===== 混同行列の描画・保存（追加） =====
    # ConfusionMatrixDisplay で描画できる :contentReference[oaicite:1]{index=1}
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)
    disp.plot(values_format="d", xticks_rotation=45)

    # matplotlib の savefig で保存 :contentReference[oaicite:2]{index=2}
    disp.figure_.tight_layout()
    disp.figure_.savefig(OUT_CM_PNG, dpi=CM_DPI)
    plt.close(disp.figure_)

    print(f"saved confusion matrix: {OUT_CM_PNG}")

    payload = {"pipeline": clf, "label_classes": le.classes_, "grid": GRID}
    joblib.dump(payload, OUT_MODEL)
    print(f"saved: {OUT_MODEL}")


if __name__ == "__main__":
    main()
