# faceTrain_SVM_subject_eval.py
# PLY点群 -> GRID^3 占有(0/1) -> flatten -> SVM
# 被験者ごとの固定テスト(7:3)で、
#   (1) 被験者別モデル精度
#   (2) 全データ(被験者train結合)モデル精度 + 全被験者test合算 混同行列/精度
# を出す。ハイパラは train 内のみ GridSearchCV（CV評価）で決める。

from pathlib import Path
import json
import csv
import numpy as np
import trimesh
import matplotlib
matplotlib.use("Agg")
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
# SETTINGS（ここだけ編集）
# =========================

# 被験者ごとのデータルート（各ルート直下に A/I/U/E/O フォルダがある前提）
# 例:
#   ./ALL2/mouth_ply_A/A/*.ply
#   ./ALL2/mouth_ply_A/I/*.ply ...
SUBJECT_DATA_ROOTS = {
    "A": Path(r"./DataSet/mouth_ply_A"),
    "B": Path(r"./DataSet/mouth_ply_B"),
    "C": Path(r"./DataSet/mouth_ply_C"),
    "D": Path(r"./DataSet/mouth_ply_D"),
    "E": Path(r"./DataSet/mouth_ply_E"),
}

GRID = 30            # 固定
TEST_SIZE = 0.3      # 固定
SEED = 42

# 表示・評価のラベル順固定（混同行列/レポートもこの順）
LABEL_ORDER = ["A", "I", "U", "E", "O"]

# 出力ルート（ここに ALL_MODEL / SUBJECT_MODELS などが作られる）
OUT_ROOT = Path(r"./ALL2/mouth_ply_subject_eval_out")
OUT_ROOT.mkdir(parents=True, exist_ok=True)

CM_DPI = 200

# SVM + GridSearch（必要なら範囲だけ編集。要求では「その他はCV」なのでここは現行踏襲）
SVM_KERNEL = "rbf"
C_GRID = [0.1, 1, 3, 5, 10, 20]
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


def occupancy_grid_features(points: np.ndarray, grid: int) -> np.ndarray:
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


def build_pipeline(seed: int) -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("svc", SVC(
            kernel=SVM_KERNEL,
            class_weight=CLASS_WEIGHT,
            probability=PROBABILITY,
            random_state=seed,
        )),
    ])


def save_confusion_matrix_png(path: Path, cm: np.ndarray, label_names: list[str], dpi: int = 200) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_names)
    plt.rcParams["font.size"] = 24
    disp.plot(values_format="d")
    disp.figure_.tight_layout()
    disp.figure_.savefig(path, dpi=dpi)
    plt.close(disp.figure_)


def collect_dataset_fixed_order_for_subject(subject_root: Path, grid: int, label_order: list[str]):
    """
    subject_root/label/*.ply を label_order の順に読み込む
    """
    if not subject_root.exists():
        raise FileNotFoundError(f"Subject root not found: {subject_root}")

    X_list, y_list = [], []
    for lab in label_order:
        lab_dir = subject_root / lab
        if not lab_dir.exists():
            raise FileNotFoundError(f"Label dir not found: {lab_dir}")

        for pf in sorted(lab_dir.glob("*.ply")):
            pts = load_points_from_ply(pf)
            feat = occupancy_grid_features(pts, grid=grid)
            X_list.append(feat)
            y_list.append(lab)

    if len(X_list) == 0:
        raise ValueError(f"No PLY files found under: {subject_root}")

    X = np.vstack(X_list)
    y_str = np.array(y_list, dtype=object)
    return X, y_str


def fit_gridsearch(X_tr: np.ndarray, y_tr: np.ndarray, seed: int) -> GridSearchCV:
    cv = StratifiedKFold(n_splits=CV_SPLITS, shuffle=True, random_state=seed)
    param_grid = {
        "svc__C": C_GRID,
        "svc__gamma": GAMMA_GRID,
    }
    grid = GridSearchCV(
        estimator=build_pipeline(seed),
        param_grid=param_grid,
        scoring="accuracy",
        cv=cv,
        n_jobs=-1,
        refit=True,
        return_train_score=True,
    )
    grid.fit(X_tr, y_tr)
    return grid


def main():
    # 文字ラベル -> 整数（順番固定）
    label_to_id = {lab: i for i, lab in enumerate(LABEL_ORDER)}
    labels_fixed = list(range(len(LABEL_ORDER)))

    # ===========================
    # 1) 被験者ごとにデータ読み込み + 固定split作成
    # ===========================
    subj_data = {}  # subj -> dict(X, y, idx_tr, idx_te)
    for subj, root in SUBJECT_DATA_ROOTS.items():
        X, y_str = collect_dataset_fixed_order_for_subject(root, GRID, LABEL_ORDER)
        y = np.array([label_to_id[s] for s in y_str], dtype=np.int64)

        idx_all = np.arange(len(y))
        try:
            idx_tr, idx_te, _, _ = train_test_split(
                idx_all, y,
                test_size=TEST_SIZE,
                random_state=SEED,
                stratify=y
            )
        except Exception as e:
            raise RuntimeError(
                f"[split error] subject={subj}: stratified split failed. "
                f"各ラベルのサンプル数が少ない可能性があります。詳細: {e}"
            )

        subj_data[subj] = {
            "root": root,
            "X": X,
            "y": y,
            "idx_tr": idx_tr,
            "idx_te": idx_te,
        }

    subjects = list(SUBJECT_DATA_ROOTS.keys())

    # ===========================
    # 2) 全データモデル（各被験者train結合で学習）
    # ===========================
    all_model_dir = OUT_ROOT / "ALL_MODEL"
    all_model_dir.mkdir(parents=True, exist_ok=True)

    X_tr_all = np.vstack([subj_data[s]["X"][subj_data[s]["idx_tr"]] for s in subjects])
    y_tr_all = np.concatenate([subj_data[s]["y"][subj_data[s]["idx_tr"]] for s in subjects])

    grid_all = fit_gridsearch(X_tr_all, y_tr_all, seed=SEED)
    best_all_model = grid_all.best_estimator_

    # 被験者別評価 + 全被験者test合算
    cm_sum = np.zeros((len(LABEL_ORDER), len(LABEL_ORDER)), dtype=np.int64)
    correct_sum = 0
    total_sum = 0

    per_subj_dir = all_model_dir / "per_subject_eval"
    per_subj_dir.mkdir(parents=True, exist_ok=True)

    per_subj_rows = []

    for subj in subjects:
        X = subj_data[subj]["X"]
        y = subj_data[subj]["y"]
        idx_te = subj_data[subj]["idx_te"]

        y_te = y[idx_te]
        y_pred = best_all_model.predict(X[idx_te])

        acc = float(accuracy_score(y_te, y_pred))
        cm = confusion_matrix(y_te, y_pred, labels=labels_fixed)

        cm_sum += cm
        correct_sum += int(np.sum(y_pred == y_te))
        total_sum += int(len(y_te))

        sd = per_subj_dir / subj
        sd.mkdir(parents=True, exist_ok=True)
        save_confusion_matrix_png(sd / "confusion_matrix.png", cm, LABEL_ORDER, dpi=CM_DPI)
        save_text = classification_report(
            y_te, y_pred, labels=labels_fixed, target_names=LABEL_ORDER, digits=4
        )
        (sd / "report.txt").write_text(save_text, encoding="utf-8")

        per_subj_rows.append({"subject": subj, "n_test": int(len(y_te)), "acc_all_model": acc})

    overall_acc = float(correct_sum / total_sum) if total_sum else 0.0
    save_confusion_matrix_png(all_model_dir / "confusion_matrix_ALL_subject_tests.png", cm_sum, LABEL_ORDER, dpi=CM_DPI)

    # ALLモデル保存
    joblib.dump(
        {
            "model": best_all_model,
            "label_order": LABEL_ORDER,
            "grid": GRID,
            "best_params": grid_all.best_params_,
            "best_cv_score": float(grid_all.best_score_),
            "overall_accuracy_on_all_subject_tests": overall_acc,
        },
        all_model_dir / "ply_svm_model_ALL.joblib"
    )

    meta_all = {
        "subjects": subjects,
        "subject_roots": {k: str(v) for k, v in SUBJECT_DATA_ROOTS.items()},
        "label_order": LABEL_ORDER,
        "grid": GRID,
        "test_size": TEST_SIZE,
        "seed": SEED,
        "cv_splits": CV_SPLITS,
        "best_params": grid_all.best_params_,
        "best_cv_score_on_all_train": float(grid_all.best_score_),
        "overall_accuracy_on_all_subject_tests": overall_acc,
    }
    (all_model_dir / "meta.json").write_text(json.dumps(meta_all, ensure_ascii=False, indent=2), encoding="utf-8")

    with (all_model_dir / "accuracy_per_subject.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["subject", "n_test", "acc_all_model"])
        for r in per_subj_rows:
            w.writerow([r["subject"], r["n_test"], f"{r['acc_all_model']:.6f}"])
        w.writerow(["__OVERALL__", total_sum, f"{overall_acc:.6f}"])

    acc_all_map = {r["subject"]: r["acc_all_model"] for r in per_subj_rows}

    # ===========================
    # 3) 被験者別モデル（各被験者trainで学習→自分のtestで評価）
    # ===========================
    subj_models_dir = OUT_ROOT / "SUBJECT_MODELS"
    subj_models_dir.mkdir(parents=True, exist_ok=True)

    compare_rows = []

    for subj in subjects:
        X = subj_data[subj]["X"]
        y = subj_data[subj]["y"]
        idx_tr = subj_data[subj]["idx_tr"]
        idx_te = subj_data[subj]["idx_te"]

        grid_sub = fit_gridsearch(X[idx_tr], y[idx_tr], seed=SEED)
        best_sub_model = grid_sub.best_estimator_

        y_te = y[idx_te]
        y_pred = best_sub_model.predict(X[idx_te])

        acc_sub = float(accuracy_score(y_te, y_pred))
        cm_sub = confusion_matrix(y_te, y_pred, labels=labels_fixed)
        report_sub = classification_report(
            y_te, y_pred, labels=labels_fixed, target_names=LABEL_ORDER, digits=4
        )

        od = subj_models_dir / subj
        od.mkdir(parents=True, exist_ok=True)

        save_confusion_matrix_png(od / "confusion_matrix.png", cm_sub, LABEL_ORDER, dpi=CM_DPI)
        (od / "report.txt").write_text(report_sub, encoding="utf-8")

        joblib.dump(
            {
                "model": best_sub_model,
                "label_order": LABEL_ORDER,
                "grid": GRID,
                "best_params": grid_sub.best_params_,
                "best_cv_score": float(grid_sub.best_score_),
                "test_accuracy": acc_sub,
            },
            od / "ply_svm_model_subject.joblib"
        )

        meta_sub = {
            "subject": subj,
            "subject_root": str(subj_data[subj]["root"]),
            "label_order": LABEL_ORDER,
            "grid": GRID,
            "test_size": TEST_SIZE,
            "seed": SEED,
            "cv_splits": CV_SPLITS,
            "best_params": grid_sub.best_params_,
            "best_cv_score_on_subject_train": float(grid_sub.best_score_),
            "test_accuracy_on_subject_test": acc_sub,
        }
        (od / "meta.json").write_text(json.dumps(meta_sub, ensure_ascii=False, indent=2), encoding="utf-8")

        compare_rows.append({
            "subject": subj,
            "n_test": int(len(y_te)),
            "acc_subject_model": acc_sub,
            "acc_all_model": float(acc_all_map.get(subj, np.nan)),
        })

    with (OUT_ROOT / "accuracy_compare_subject_vs_all.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["subject", "n_test", "acc_subject_model", "acc_all_model"])
        for r in compare_rows:
            w.writerow([r["subject"], r["n_test"], f"{r['acc_subject_model']:.6f}", f"{r['acc_all_model']:.6f}"])

    print("DONE")
    print(f"Saved to: {OUT_ROOT}")
    print(f"ALL model overall acc (all subject tests) = {overall_acc:.4f}")


if __name__ == "__main__":
    main()
