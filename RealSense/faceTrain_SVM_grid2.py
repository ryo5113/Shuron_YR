# - 全組合せ(2スケール/3スケール)を完全走査
# - sweep_rows に test_acc_fast を追加（評価データ精度）
# - グラフは test_acc_fast の上位Kを「グリッド組合せラベル付き」で表示

from pathlib import Path
import json
import itertools
import numpy as np
import trimesh
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, ConfusionMatrixDisplay

# =========================
# 設定（ここだけ編集）
# =========================
DATA_ROOT = Path(r"./PLY_dataset_3v2")   # ラベル別フォルダを含むルート（A/I/U/E/O）
LABEL_ORDER = ["A", "I", "U", "E", "O"]

TEST_SIZE = 0.3
SEED = 42

OUT_DIR = Path(r"./PLY_dataset_3v2")
OUT_DIR.mkdir(parents=True, exist_ok=True)

CM_DPI = 200

# --- 比較したい特徴モード ---
FEATURE_MODES = ["dens", "occ"]   # 片方だけなら ["dens"] など

# --- マルチスケール探索条件（全列挙） ---
GRID_CANDIDATES = list(range(10, 61, 5))  # 10..60 step 5（11候補）
SCALE_CHOICES = [2, 3]                   # 2スケール/3スケール
TOPK_FOR_REFINED_SEARCH = 5              # 上位K個だけ精密化
PLOT_TOPK = 30                           # グラフに出す上位K（test精度）

# --- (A) まずは安く評価するための固定SVM設定（候補探索用） ---
FAST_C = 10.0
FAST_GAMMA = "scale"
FAST_CV_SPLITS = 3  # CVも保存するが、主に test_acc_fast を使う

# --- (B) 上位候補だけ精密化（必要なら） ---
REFINE_SVM_PARAMS = True
REFINE_CV_SPLITS = 5
C_GRID = [0.1, 1, 3, 5, 10, 30, 100]
GAMMA_GRID = ["scale", "auto"]

SVM_KERNEL = "rbf"
CLASS_WEIGHT = None
PROBABILITY = True
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
    pts = points.astype(np.float64, copy=True)
    pts -= pts.mean(axis=0, keepdims=True)
    max_abs = np.max(np.abs(pts))
    if max_abs > 0:
        pts /= max_abs
    pts = np.clip(pts, -1.0, 1.0)
    return pts


def voxel_features_from_unit(pts_unit: np.ndarray, grid: int, mode: str) -> np.ndarray:
    idx = ((pts_unit + 1.0) * 0.5 * grid).astype(np.int64)
    idx = np.clip(idx, 0, grid - 1)

    if mode == "dens":
        vol = np.zeros((grid, grid, grid), dtype=np.float64)
        np.add.at(vol, (idx[:, 0], idx[:, 1], idx[:, 2]), 1.0)
        return vol.reshape(-1)

    if mode == "occ":
        vol = np.zeros((grid, grid, grid), dtype=np.uint8)
        vol[idx[:, 0], idx[:, 1], idx[:, 2]] = 1
        return vol.reshape(-1).astype(np.float64)

    raise ValueError(f"Unknown mode: {mode}")


def multiscale_voxel_features(points: np.ndarray, grids: list[int], mode: str) -> np.ndarray:
    pts_unit = normalize_points_to_unit(points)
    feats = [voxel_features_from_unit(pts_unit, g, mode=mode) for g in grids]
    return np.concatenate(feats, axis=0)


def collect_dataset_fixed_order(data_root: Path, grids: list[int], label_order: list[str], mode: str):
    if not data_root.exists():
        raise FileNotFoundError(f"DATA_ROOT not found: {data_root}")

    X_list, y_list = [], []
    for lab in label_order:
        lab_dir = data_root / lab
        if not lab_dir.exists():
            raise FileNotFoundError(f"Label dir not found: {lab_dir}")
        for pf in sorted(lab_dir.glob("*.ply")):
            pts = load_points_from_ply(pf)
            feat = multiscale_voxel_features(pts, grids=grids, mode=mode)
            X_list.append(feat)
            y_list.append(lab)

    if len(X_list) == 0:
        raise ValueError(f"No PLY files found under: {data_root}")

    X = np.vstack(X_list)
    y_str = np.array(y_list, dtype=object)
    return X, y_str


def build_pipeline(C_value: float, gamma_value) -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("svc", SVC(
            kernel=SVM_KERNEL,
            C=C_value,
            gamma=gamma_value,
            class_weight=CLASS_WEIGHT,
            probability=PROBABILITY,
            random_state=SEED,
        )),
    ])


def cv_score_on_train(X_tr: np.ndarray, y_tr: np.ndarray, C_value: float, gamma_value, cv_splits: int) -> float:
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=SEED)
    pipe = build_pipeline(C_value, gamma_value)

    scores = []
    for tr_idx, va_idx in cv.split(X_tr, y_tr):
        pipe.fit(X_tr[tr_idx], y_tr[tr_idx])
        pred = pipe.predict(X_tr[va_idx])
        scores.append(accuracy_score(y_tr[va_idx], pred))

    return float(np.mean(scores))


def refined_svm_search(X_tr: np.ndarray, y_tr: np.ndarray) -> GridSearchCV:
    cv = StratifiedKFold(n_splits=REFINE_CV_SPLITS, shuffle=True, random_state=SEED)
    param_grid = {"svc__C": C_GRID, "svc__gamma": GAMMA_GRID}

    base = Pipeline([
        ("scaler", StandardScaler()),
        ("svc", SVC(
            kernel=SVM_KERNEL,
            class_weight=CLASS_WEIGHT,
            probability=PROBABILITY,
            random_state=SEED,
        )),
    ])

    grid = GridSearchCV(
        estimator=base,
        param_grid=param_grid,
        scoring="accuracy",
        cv=cv,
        n_jobs=-1,
        refit=True,
        return_train_score=False,
    )
    grid.fit(X_tr, y_tr)
    return grid


def save_confusion_matrix_png(path: Path, cm: np.ndarray, label_names: list[str], dpi: int = 200) -> None:
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_names)
    plt.rcParams["font.size"] = 16
    disp.plot(values_format="d", xticks_rotation=45)
    disp.figure_.tight_layout()
    disp.figure_.savefig(path, dpi=dpi)
    plt.close(disp.figure_)


def combo_to_str(combo: list[int]) -> str:
    return "+".join(str(g) for g in combo)


def plot_topk_test_accuracy(path: Path, rows_sorted: list[dict], title: str, topk: int):
    top = rows_sorted[:min(topk, len(rows_sorted))]
    labels = [combo_to_str(r["grids"]) for r in top]
    ys = [r["test_acc_fast"] for r in top]

    plt.rcParams["font.size"] = 12
    plt.figure(figsize=(max(10, int(len(top) * 0.5)), 6))
    plt.bar(range(len(top)), ys)
    plt.xticks(range(len(top)), labels, rotation=60, ha="right")
    plt.xlabel("Grid combo (multi-scale)")
    plt.ylabel("Test accuracy (fast model)")
    plt.title(title)
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def all_grid_combos() -> list[list[int]]:
    combos = []
    for k in SCALE_CHOICES:
        for c in itertools.combinations(GRID_CANDIDATES, k):
            combos.append(list(c))
    return combos  # 55 + 165 = 220 通り


def run_one_mode(mode: str):
    OUT_MODEL = OUT_DIR / f"ply_svm_model_multiscale_fullscan_{mode}.joblib"
    OUT_CM_PNG = OUT_DIR / f"confusion_matrix_multiscale_fullscan_{mode}.png"
    OUT_META_JSON = OUT_DIR / f"meta_multiscale_fullscan_{mode}.json"
    OUT_SWEEP_JSON = OUT_DIR / f"grid_combo_sweep_fullscan_{mode}.json"
    OUT_SWEEP_PNG = OUT_DIR / f"grid_combo_sweep_fullscan_{mode}.png"

    # train/test 分割を固定するため、最初にyを作る（特徴はダミーで良い）
    X0, y_str0 = collect_dataset_fixed_order(DATA_ROOT, grids=[GRID_CANDIDATES[0]], label_order=LABEL_ORDER, mode=mode)
    label_to_id = {lab: i for i, lab in enumerate(LABEL_ORDER)}
    y0 = np.array([label_to_id[s] for s in y_str0], dtype=np.int64)

    idx_all = np.arange(len(y0))
    idx_tr, idx_te, y_tr, y_te = train_test_split(
        idx_all, y0,
        test_size=TEST_SIZE,
        random_state=SEED,
        stratify=y0
    )

    combos = all_grid_combos()
    print(f"[{mode}] total combos = {len(combos)} (should be 220)")

    sweep_rows = []

    for i, combo in enumerate(combos):
        X, y_str = collect_dataset_fixed_order(DATA_ROOT, grids=combo, label_order=LABEL_ORDER, mode=mode)
        y = np.array([label_to_id[s] for s in y_str], dtype=np.int64)

        X_tr, X_te = X[idx_tr], X[idx_te]

        # (A) fast CV（参考値）
        cv_acc = cv_score_on_train(X_tr, y_tr, C_value=FAST_C, gamma_value=FAST_GAMMA, cv_splits=FAST_CV_SPLITS)

        # (A) fast test accuracy（あなたの要望：評価データ精度）
        fast_model = build_pipeline(FAST_C, FAST_GAMMA)
        fast_model.fit(X_tr, y_tr)
        y_pred_fast = fast_model.predict(X_te)
        test_acc_fast = float(accuracy_score(y_te, y_pred_fast))

        sweep_rows.append({
            "idx": int(i),
            "grids": combo,
            "feature_dim": int(sum([g ** 3 for g in combo])),
            "cv_acc_fast": float(cv_acc),
            "test_acc_fast": float(test_acc_fast),
        })

        if (i % 20) == 0:
            print(f"[{mode} {i:03d}/{len(combos)}] grids={combo} test_acc_fast={test_acc_fast:.4f} cv_acc_fast={cv_acc:.4f}")

    # test精度でソート（グラフもこれ）
    sweep_rows_sorted = sorted(sweep_rows, key=lambda r: r["test_acc_fast"], reverse=True)
    top_rows = sweep_rows_sorted[:max(1, TOPK_FOR_REFINED_SEARCH)]

    OUT_SWEEP_JSON.write_text(json.dumps({
        "mode": mode,
        "data_root": str(DATA_ROOT),
        "grid_candidates": GRID_CANDIDATES,
        "scale_choices": SCALE_CHOICES,
        "n_combos": len(combos),
        "fast_eval": {"C": FAST_C, "gamma": FAST_GAMMA, "cv_splits": FAST_CV_SPLITS},
        "rows_sorted_by_test": sweep_rows_sorted,
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    plot_topk_test_accuracy(OUT_SWEEP_PNG, sweep_rows_sorted, title=f"Top-{PLOT_TOPK} test accuracy (fast) [{mode}]", topk=PLOT_TOPK)

    print(f"\n=== [{mode}] Top combos by test_acc_fast ===")
    for r in top_rows:
        print(f"grids={r['grids']}  test_acc_fast={r['test_acc_fast']:.4f}  cv_acc_fast={r['cv_acc_fast']:.4f}  dim={r['feature_dim']}")

    # (B) 上位候補だけ精密化（任意）
    best = None
    for rank, r in enumerate(top_rows):
        combo = r["grids"]
        X, y_str = collect_dataset_fixed_order(DATA_ROOT, grids=combo, label_order=LABEL_ORDER, mode=mode)
        y = np.array([label_to_id[s] for s in y_str], dtype=np.int64)
        X_tr, X_te = X[idx_tr], X[idx_te]

        if REFINE_SVM_PARAMS:
            gs = refined_svm_search(X_tr, y_tr)
            model = gs.best_estimator_
            best_cv = float(gs.best_score_)
            best_params = gs.best_params_
        else:
            model = build_pipeline(FAST_C, FAST_GAMMA)
            model.fit(X_tr, y_tr)
            best_cv = float(r["cv_acc_fast"])
            best_params = {"svc__C": FAST_C, "svc__gamma": FAST_GAMMA}

        y_pred = model.predict(X_te)
        test_acc = float(accuracy_score(y_te, y_pred))

        cand = {
            "rank_in_top": int(rank),
            "grids": combo,
            "feature_dim": int(sum([g ** 3 for g in combo])),
            "best_cv_score": best_cv,
            "best_params": best_params,
            "test_accuracy": test_acc,
            "model": model,
        }

        print(f"\n[{mode} refine {rank}] grids={combo}")
        print(f"  best_cv_score={best_cv:.4f}  test_acc={test_acc:.4f}  best_params={best_params}")

        if (best is None) or (cand["test_accuracy"] > best["test_accuracy"]):
            best = cand

    best_model = best["model"]
    best_grids = best["grids"]

    # confusion matrix / report
    X_best, y_str_best = collect_dataset_fixed_order(DATA_ROOT, grids=best_grids, label_order=LABEL_ORDER, mode=mode)
    y_best = np.array([label_to_id[s] for s in y_str_best], dtype=np.int64)
    X_tr_best, X_te_best = X_best[idx_tr], X_best[idx_te]

    y_pred_best = best_model.predict(X_te_best)

    cm = confusion_matrix(y_te, y_pred_best, labels=list(range(len(LABEL_ORDER))))
    report = classification_report(
        y_te, y_pred_best,
        labels=list(range(len(LABEL_ORDER))),
        target_names=LABEL_ORDER,
        digits=4
    )

    print(f"\n=== FINAL BEST ({mode}) ===")
    print(f"BEST_GRIDS: {best_grids}")
    print(f"feature_dim: {best['feature_dim']}")
    print(f"best_cv_score: {best['best_cv_score']:.4f}")
    print(f"best_params: {best['best_params']}")
    print(f"test_accuracy: {best['test_accuracy']:.4f}")
    print(report)

    save_confusion_matrix_png(OUT_CM_PNG, cm, LABEL_ORDER, dpi=CM_DPI)

    payload = {
        "model": best_model,
        "label_order": LABEL_ORDER,
        "grids": best_grids,
        "mode": mode,
        "best_params": best["best_params"],
        "best_cv_score": float(best["best_cv_score"]),
        "test_accuracy": float(best["test_accuracy"]),
        "search": {
            "grid_candidates": GRID_CANDIDATES,
            "scale_choices": SCALE_CHOICES,
            "n_combos": len(combos),
            "fast_eval": {"C": FAST_C, "gamma": FAST_GAMMA, "cv_splits": FAST_CV_SPLITS},
            "topk_refine": TOPK_FOR_REFINED_SEARCH,
            "refine_svm_params": REFINE_SVM_PARAMS,
            "plot_topk": PLOT_TOPK,
        }
    }
    joblib.dump(payload, OUT_MODEL)

    meta = {
        "mode": mode,
        "label_order": LABEL_ORDER,
        "best_grids": best_grids,
        "feature_dim": best["feature_dim"],
        "best_params": best["best_params"],
        "best_cv_score": float(best["best_cv_score"]),
        "test_accuracy": float(best["test_accuracy"]),
        "outputs": {
            "model": str(OUT_MODEL),
            "confusion_matrix_png": str(OUT_CM_PNG),
            "sweep_json": str(OUT_SWEEP_JSON),
            "sweep_png": str(OUT_SWEEP_PNG),
        }
    }
    OUT_META_JSON.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    return meta


def main():
    summaries = []
    for mode in FEATURE_MODES:
        summaries.append(run_one_mode(mode))

    print("\n=== SUMMARY ===")
    for s in summaries:
        print(f"mode={s['mode']} best_grids={s['best_grids']} test_acc={s['test_accuracy']:.4f}")
        print(f"  sweep_json={s['outputs']['sweep_json']}")
        print(f"  sweep_png ={s['outputs']['sweep_png']}")


if __name__ == "__main__":
    main()
