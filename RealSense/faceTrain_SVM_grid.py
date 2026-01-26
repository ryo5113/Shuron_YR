# 目的:
# - 複数DATA_ROOT（被験者ごとのフォルダ）× 複数GRID を評価して
# - 被験者別の精度推移と、全被験者平均の精度推移をグラフ化する
#
# 前提:
# - 各DATA_ROOT配下に A/I/U/E/O の各フォルダがあり、その中に .ply がある
# - faceTrain_SVM.py / faceTrain_SVM_dens.py と同じ前処理・学習器構成を踏襲

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
from sklearn.metrics import accuracy_score

# =========================
# 設定（ここだけ編集）
# =========================

# 被験者ごとのデータルートを複数指定（例）
DATA_ROOTS = [
    Path(r"./NN/mouth_ply"),
]

# 評価したいグリッドサイズを複数指定
GRIDS = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70]

# どちらの特徴で評価するか
# - "occ": 占有(0/1)  ※ faceTrain_SVM.py 相当
# - "dens": 点数カウント(密度) ※ faceTrain_SVM_dens.py 相当（相対密度化やN追加は使わない）
FEATURE_MODE = "occ"  # "occ" or "dens"

TEST_SIZE = 0.3
SEED = 42

LABEL_ORDER = ["A", "I", "U", "E", "O"]

# SVM + GridSearch（faceTrain_* と同等の設定想定）
SVM_KERNEL = "rbf"
C_GRID = [0.1, 1, 3, 5, 10, 30, 100]
GAMMA_GRID = ["scale", "auto"]
CLASS_WEIGHT = None
PROBABILITY = True
CV_SPLITS = 5

# 出力
OUT_DIR = Path("./grid_sweep_results")
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_PNG = OUT_DIR / "grid_sweep_accuracy.png"
OUT_JSON = OUT_DIR / "grid_sweep_accuracy.json"

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


def normalize_points(points: np.ndarray) -> np.ndarray:
    # faceTrain_* と同様: center -> scale -> clip
    pts = points.astype(np.float64, copy=True)

    pts -= pts.mean(axis=0, keepdims=True)

    max_abs = np.max(np.abs(pts))
    if max_abs > 0:
        pts /= max_abs

    pts = np.clip(pts, -1.0, 1.0)
    return pts


def voxel_features(points: np.ndarray, grid: int, mode: str) -> np.ndarray:
    """
    mode:
      - "occ"  : 占有(0/1)
      - "dens" : 点数カウント
    """
    pts = normalize_points(points)

    # [-1,1] -> [0, grid-1]
    idx = ((pts + 1.0) * 0.5 * grid).astype(np.int64)
    idx = np.clip(idx, 0, grid - 1)

    if mode == "occ":
        occ = np.zeros((grid, grid, grid), dtype=np.uint8)
        occ[idx[:, 0], idx[:, 1], idx[:, 2]] = 1
        return occ.reshape(-1).astype(np.float64)

    if mode == "dens":
        counts = np.zeros((grid, grid, grid), dtype=np.float64)
        np.add.at(counts, (idx[:, 0], idx[:, 1], idx[:, 2]), 1.0)
        return counts.reshape(-1)

    raise ValueError(f"Unknown FEATURE_MODE: {mode}")


def collect_dataset_fixed_order(data_root: Path, grid: int, label_order: list[str], mode: str):
    if not data_root.exists():
        raise FileNotFoundError(f"DATA_ROOT not found: {data_root}")

    X_list, y_list = [], []

    for lab in label_order:
        lab_dir = data_root / lab
        if not lab_dir.exists():
            raise FileNotFoundError(f"Label dir not found: {lab_dir}")

        for pf in sorted(lab_dir.glob("*.ply")):
            pts = load_points_from_ply(pf)
            feat = voxel_features(pts, grid=grid, mode=mode)
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


def evaluate_one(data_root: Path, grid: int, label_order: list[str], mode: str) -> dict:
    # 1) データ作成
    X, y_str = collect_dataset_fixed_order(data_root, grid, label_order, mode)

    # 2) 文字ラベル -> 整数（順番固定）
    label_to_id = {lab: i for i, lab in enumerate(label_order)}
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

    # 4) train側のみでGridSearchCV
    cv = StratifiedKFold(n_splits=CV_SPLITS, shuffle=True, random_state=SEED)
    param_grid = {"svc__C": C_GRID, "svc__gamma": GAMMA_GRID}

    gs = GridSearchCV(
        estimator=build_pipeline(),
        param_grid=param_grid,
        scoring="accuracy",
        cv=cv,
        n_jobs=-1,
        refit=True,
        return_train_score=False,
    )
    gs.fit(X_tr, y_tr)

    best_model = gs.best_estimator_
    y_pred = best_model.predict(X_te)
    test_acc = float(accuracy_score(y_te, y_pred))

    return {
        "data_root": str(data_root),
        "grid": int(grid),
        "feature_mode": mode,
        "samples": int(len(X)),
        "feature_dim": int(grid ** 3),
        "best_cv_score": float(gs.best_score_),
        "best_params": gs.best_params_,
        "test_accuracy": test_acc,
    }


def plot_results(results_by_subject: dict, grids: list[int], out_png: Path):
    # results_by_subject[subject_name][grid] = test_accuracy
    plt.figure(figsize=(10, 6))

    # # 被験者ごとのライン
    # for subj, g2acc in results_by_subject.items():
    #     ys = [g2acc.get(g, np.nan) for g in grids]
    #     plt.plot(grids, ys, marker="o", label=subj)

    # 平均ライン
    mean_ys = []
    for g in grids:
        vals = []
        for subj in results_by_subject:
            v = results_by_subject[subj].get(g, None)
            if v is not None:
                vals.append(v)
        mean_ys.append(float(np.mean(vals)) if len(vals) > 0 else np.nan)

    plt.plot(grids, mean_ys, marker="o", linewidth=3)

    plt.xlabel("GRID size", fontsize=30)
    plt.ylabel("Test Accuracy", fontsize=30)
    plt.title("Grid Sweep Accuracy", fontsize=30)
    plt.xticks(fontsize=25)
    plt.yticks(fontsize=25)
    plt.grid(alpha=0.3)
    #plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def main():
    all_runs = []
    results_by_subject = {}

    for data_root in DATA_ROOTS:
        subj_name = Path(data_root).name
        results_by_subject[subj_name] = {}

        print(f"\n=== Subject: {subj_name} ({data_root}) ===")
        for g in GRIDS:
            r = evaluate_one(data_root, g, LABEL_ORDER, FEATURE_MODE)
            all_runs.append(r)
            results_by_subject[subj_name][g] = r["test_accuracy"]
            print(f"GRID={g:>3}  test_acc={r['test_accuracy']:.4f}  best_cv={r['best_cv_score']:.4f}  best={r['best_params']}")

    # グラフ保存
    plot_results(results_by_subject, GRIDS, OUT_PNG)
    print(f"\nSaved plot: {OUT_PNG}")

    # JSON保存（再現・記録用）
    out_obj = {
        "feature_mode": FEATURE_MODE,
        "grids": GRIDS,
        "data_roots": [str(p) for p in DATA_ROOTS],
        "runs": all_runs,
    }
    OUT_JSON.write_text(json.dumps(out_obj, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved json : {OUT_JSON}")


if __name__ == "__main__":
    main()
