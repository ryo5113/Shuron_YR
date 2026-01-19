# faceTrain_SVM_multiscale.py
# PLY点群 -> (複数グリッド)ボクセル点数カウント -> concat -> SVM
# 既存: faceTrain_SVM_dens.py をベースにマルチスケール化

from pathlib import Path
import json
import numpy as np
import trimesh
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
)

# =========================
# 設定（ここだけ編集）
# =========================
DATA_ROOT = Path(r"./PLY_dataset_all")  # ラベル別フォルダを含むルート

# ★マルチスケール：ここに使いたいグリッドサイズを並べる
GRIDS = [5, 15, 25]   # 例：粗→中→細（必要に応じて変更）

TEST_SIZE = 0.3
SEED = 42

LABEL_ORDER = ["A", "I", "U", "E", "O"]

OUT_DIR = Path(r"./PLY_dataset_all")
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_MODEL = OUT_DIR / "ply_svm_model_multiscale.joblib"
OUT_CM_PNG = OUT_DIR / "confusion_matrix_multiscale.png"
OUT_META_JSON = OUT_DIR / "meta_multiscale.json"
CM_DPI = 200

SVM_KERNEL = "rbf"
C_GRID = [0.1, 1, 3, 5, 10, 30, 100]
GAMMA_GRID = ["scale", "auto"]
CLASS_WEIGHT = None
PROBABILITY = True
CV_SPLITS = 5
# =========================


def load_points_from_ply(ply_path: Path) -> np.ndarray:
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


def normalize_points_to_unit(points: np.ndarray) -> np.ndarray:
    """
    既存実装（center -> scale -> clip）を共通化。
    結果は [-1, 1] に収まる点群。
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
    return pts


def voxel_count_features_from_unit(pts_unit: np.ndarray, grid: int) -> np.ndarray:
    """
    pts_unit: [-1,1] に正規化済みの点群
    grid: ボクセル分割数
    return: grid^3 次元（点数カウント）のflatten
    """
    # [-1,1] -> [0, grid-1]
    idx = ((pts_unit + 1.0) * 0.5 * grid).astype(np.int64)
    idx = np.clip(idx, 0, grid - 1)

    counts = np.zeros((grid, grid, grid), dtype=np.float64)
    np.add.at(counts, (idx[:, 0], idx[:, 1], idx[:, 2]), 1.0)

    return counts.reshape(-1)


def multiscale_voxel_features(points: np.ndarray, grids: list[int]) -> np.ndarray:
    """
    ★マルチスケール特徴量：
    同一正規化座標系上で、複数gridのボクセル点数特徴を連結して返す
    """
    pts_unit = normalize_points_to_unit(points)

    feats = []
    for g in grids:
        feats.append(voxel_count_features_from_unit(pts_unit, g))

    return np.concatenate(feats, axis=0)


def collect_dataset_fixed_order(data_root: Path, grids: list[int], label_order: list[str]):
    """
    label_order の順に data_root/label/*.ply を読み込む（マルチスケール版）
    """
    if not data_root.exists():
        raise FileNotFoundError(f"DATA_ROOT not found: {data_root}")

    X_list, y_list = [], []

    for lab in label_order:
        lab_dir = data_root / lab
        if not lab_dir.exists():
            raise FileNotFoundError(f"Label dir not found: {lab_dir}")

        for pf in sorted(lab_dir.glob("*.ply")):
            pts = load_points_from_ply(pf)
            feat = multiscale_voxel_features(pts, grids=grids)
            X_list.append(feat)
            y_list.append(lab)

    if len(X_list) == 0:
        raise ValueError(f"No PLY files found under: {data_root}")

    X = np.vstack(X_list)
    y_str = np.array(y_list, dtype=object)
    return X, y_str


def build_pipeline() -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("svc", SVC(
            kernel=SVM_KERNEL,
            class_weight=CLASS_WEIGHT,
            probability=PROBABILITY,
            random_state=SEED,
        )),
    ])


def save_confusion_matrix_png(path: Path, cm: np.ndarray, label_names: list[str], dpi: int = 200) -> None:
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_names)
    plt.rcParams["font.size"] = 16
    disp.plot(values_format="d", xticks_rotation=45)
    disp.figure_.tight_layout()
    disp.figure_.savefig(path, dpi=dpi)
    plt.close(disp.figure_)


def main():
    # 1) データ作成（ラベル順固定）
    X, y_str = collect_dataset_fixed_order(DATA_ROOT, GRIDS, LABEL_ORDER)

    # 2) 文字ラベル -> 整数（順番固定）
    label_to_id = {lab: i for i, lab in enumerate(LABEL_ORDER)}
    y = np.array([label_to_id[s] for s in y_str], dtype=np.int64)

    # 3) 外側ホールドアウト
    idx_all = np.arange(len(y))
    idx_tr, idx_te, y_tr, y_te = train_test_split(
        idx_all, y,
        test_size=TEST_SIZE,
        random_state=SEED,
        stratify=y
    )
    X_tr, X_te = X[idx_tr], X[idx_te]

    # 4) 内側CVでGridSearch（train側のみで最適化）
    cv = StratifiedKFold(n_splits=CV_SPLITS, shuffle=True, random_state=SEED)
    param_grid = {
        "svc__C": C_GRID,
        "svc__gamma": GAMMA_GRID,
    }

    grid = GridSearchCV(
        estimator=build_pipeline(),
        param_grid=param_grid,
        scoring="accuracy",
        cv=cv,
        n_jobs=-1,
        refit=True,
        return_train_score=True,
    )
    grid.fit(X_tr, y_tr)

    best_model = grid.best_estimator_
    y_pred = best_model.predict(X_te)

    # 5) 評価（順番固定）
    labels_fixed = list(range(len(LABEL_ORDER)))
    acc = float(accuracy_score(y_te, y_pred))

    cm = confusion_matrix(y_te, y_pred, labels=labels_fixed)
    report = classification_report(
        y_te, y_pred,
        labels=labels_fixed,
        target_names=LABEL_ORDER,
        digits=4
    )

    feature_dim = int(sum([g ** 3 for g in GRIDS]))

    print("=== Dataset ===")
    print(f"DATA_ROOT: {DATA_ROOT}")
    print(f"samples: {len(X)}")
    print(f"labels(order fixed): {LABEL_ORDER}")
    print(f"GRIDS: {GRIDS} -> feature_dim: {feature_dim}")

    print("=== Model Selection (train CV) ===")
    print(f"best_cv_score: {grid.best_score_:.4f}")
    print(f"best_params: {grid.best_params_}")

    print("=== Test Eval (holdout) ===")
    print(f"accuracy: {acc:.4f}")
    print("confusion_matrix (fixed order A,I,U,E,O):")
    print(cm)
    print("classification_report:")
    print(report)

    # 6) 混同行列の保存
    save_confusion_matrix_png(OUT_CM_PNG, cm, LABEL_ORDER, dpi=CM_DPI)
    print(f"saved confusion matrix: {OUT_CM_PNG}")

    # 7) モデル保存（payload）
    payload = {
        "model": best_model,
        "label_order": LABEL_ORDER,
        "grids": GRIDS,
        "best_params": grid.best_params_,
        "best_cv_score": float(grid.best_score_),
        "test_accuracy": acc,
    }
    joblib.dump(payload, OUT_MODEL)
    print(f"saved model: {OUT_MODEL}")

    # 8) メタ情報保存
    meta = {
        "label_order": LABEL_ORDER,
        "grids": GRIDS,
        "best_params": grid.best_params_,
        "best_cv_score": float(grid.best_score_),
        "test_accuracy": acc,
    }
    OUT_META_JSON.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved meta: {OUT_META_JSON}")


if __name__ == "__main__":
    main()
