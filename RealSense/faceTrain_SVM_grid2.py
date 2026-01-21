# faceTrain_SVM_grid2_fixed_full.py
# 目的:
# - 全組合せ(2スケール/3スケール)を完全走査
# - 探索(順位付け)はCV精度で実施
# - CV上位Kに対してのみ評価データ(test)精度を算出・可視化（= CV後にtestで評価）
# - 最終モデルは「上位Kの中で(必要ならSVMパラメータもCVで最適化した上で) test精度最大」を採用
#
# 修正点（元スクリプトからの主な修正）:
# - OUT_SWEEP_JSON の未定義を解消
# - sweep_rows.append の二重化を解消（全comboはCVのみ1行）
# - best の上書きを解消（best_refined / best_fast を分離）
# - グラフ/CSVは test を持つ top_rows を対象（全comboにtestを要求しない）
# - run_one_mode() が meta を return（main() が落ちない）

from pathlib import Path
import json
import itertools
import math
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
FEATURE_MODES = ["occ"]   # 例: ["dens", "occ"] にすると両方回せる

# --- マルチスケール探索条件（全列挙） ---
GRID_CANDIDATES = list(range(10, 61, 5))  # 20..75 step 5（候補数=12）
SCALE_CHOICES = [2, 3]                   # 2スケール/3スケール
TOPK_FOR_TEST_EVAL = 30                  # CV上位Kに対してのみ test 精度を算出（表示・CSVもこの範囲）
TOPK_FOR_REFINED_SEARCH = 5              # さらにその中から上位Kを精密化（SVMのC/gammaをCVで最適化）して最終決定
PLOT_TOPK = 10                           # TopKの棒グラフ表示件数（test精度）

# --- (A) 安価な探索用SVM設定（CVスコア算出用） ---
FAST_C = 10.0
FAST_GAMMA = "scale"
FAST_CV_SPLITS = 3

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


def all_grid_combos() -> list[list[int]]:
    combos = []
    for k in SCALE_CHOICES:
        for c in itertools.combinations(GRID_CANDIDATES, k):
            combos.append(list(c))
    return combos


def save_rank_csv(path: Path, rows: list[dict], include_test: bool):
    with path.open("w", encoding="utf-8") as f:
        if include_test:
            f.write("rank,cv_acc_fast,test_acc_fast,feature_dim,grids\n")
        else:
            f.write("rank,cv_acc_fast,feature_dim,grids\n")

        for i, r in enumerate(rows, start=1):
            grids_str = combo_to_str(r["grids"])
            if include_test:
                f.write(f"{i},{r['cv_acc_fast']:.6f},{r['test_acc_fast']:.6f},{r['feature_dim']},{grids_str}\n")
            else:
                f.write(f"{i},{r['cv_acc_fast']:.6f},{r['feature_dim']},{grids_str}\n")


def plot_cv_hist(path: Path, rows_all: list[dict], title: str):
    ys = [r["cv_acc_fast"] for r in rows_all]
    plt.rcParams["font.size"] = 18
    plt.figure(figsize=(10, 6))
    plt.hist(ys, bins=20)
    plt.xlabel("CV accuracy (fast)")
    plt.ylabel("Count (number of grid-combos)")
    plt.title(title)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def plot_topk_test_accuracy_hbar(path: Path, rows_with_test_sorted: list[dict], title: str, topk: int = 10):
    top = rows_with_test_sorted[:min(topk, len(rows_with_test_sorted))]
    labels = [combo_to_str(r["grids"]) for r in top]
    ys = [r["test_acc_fast"] for r in top]

    plt.rcParams["font.size"] = 18
    plt.figure(figsize=(12, max(4, 0.55 * len(top))))
    plt.barh(range(len(top)), ys)
    plt.yticks(range(len(top)), labels)
    plt.gca().invert_yaxis()
    plt.xlabel("Test accuracy (fast model)")
    plt.title(title)

    for i, v in enumerate(ys):
        plt.text(v, i, f" {v:.3f}", va="center")

    plt.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def run_one_mode(mode: str):
    # 出力（mode別）
    OUT_SWEEP_JSON = OUT_DIR / f"grid_combo_sweep_fullscan_{mode}.json"
    OUT_CV_HIST = OUT_DIR / f"cvacc_hist_{mode}.png"
    OUT_TOPK_TEST = OUT_DIR / f"top{PLOT_TOPK}_testacc_{mode}.png"
    OUT_RANK_CV_ALL = OUT_DIR / f"rank_all_by_cv_{mode}.csv"
    OUT_RANK_TOP_TEST = OUT_DIR / f"rank_top_by_test_{mode}.csv"

    OUT_MODEL = OUT_DIR / f"ply_svm_model_multiscale_{mode}.joblib"
    OUT_CM_PNG = OUT_DIR / f"confusion_matrix_multiscale_{mode}.png"
    OUT_META_JSON = OUT_DIR / f"meta_multiscale_{mode}.json"

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
    n = len(GRID_CANDIDATES)
    expected = sum(math.comb(n, k) for k in SCALE_CHOICES)
    print(f"[{mode}] total combos = {len(combos)} (expected {expected}; candidates={n})")

    # ------------------------------------------------------------
    # (1) 全combo: CV精度のみ計算（探索はCVで実施）
    # ------------------------------------------------------------
    sweep_rows = []
    for i, combo in enumerate(combos):
        X, y_str = collect_dataset_fixed_order(DATA_ROOT, grids=combo, label_order=LABEL_ORDER, mode=mode)
        y = np.array([label_to_id[s] for s in y_str], dtype=np.int64)
        X_tr, X_te = X[idx_tr], X[idx_te]  # X_teはここでは使わない（testは後段）

        cv_acc = cv_score_on_train(X_tr, y_tr, C_value=FAST_C, gamma_value=FAST_GAMMA, cv_splits=FAST_CV_SPLITS)

        sweep_rows.append({
            "idx": int(i),
            "grids": combo,
            "feature_dim": int(sum([g ** 3 for g in combo])),
            "cv_acc_fast": float(cv_acc),
        })

        if (i % 20) == 0:
            print(f"[{mode} {i:03d}/{len(combos)}] grids={combo} cv_acc_fast={cv_acc:.4f}")

    # CVでソート（探索結果）
    sweep_rows_sorted = sorted(sweep_rows, key=lambda r: r["cv_acc_fast"], reverse=True)

    # 全件のCVランキングCSV + CV分布ヒストグラム
    save_rank_csv(OUT_RANK_CV_ALL, sweep_rows_sorted, include_test=False)
    plot_cv_hist(OUT_CV_HIST, sweep_rows_sorted, title=f"CV accuracy distribution (all combos) [{mode}]")

    # JSON（全comboはCVのみ）
    OUT_SWEEP_JSON.write_text(json.dumps({
        "mode": mode,
        "data_root": str(DATA_ROOT),
        "grid_candidates": GRID_CANDIDATES,
        "scale_choices": SCALE_CHOICES,
        "n_combos": len(combos),
        "expected_combos": expected,
        "fast_eval": {"C": FAST_C, "gamma": FAST_GAMMA, "cv_splits": FAST_CV_SPLITS},
        "rows_sorted_by_cv": sweep_rows_sorted,
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    # ------------------------------------------------------------
    # (2) CV上位Kに対してのみ test 精度を算出（= CV後にtestで評価）
    # ------------------------------------------------------------
    top_for_test = sweep_rows_sorted[:max(1, TOPK_FOR_TEST_EVAL)]

    rows_with_test = []
    for r in top_for_test:
        combo = r["grids"]
        X, y_str = collect_dataset_fixed_order(DATA_ROOT, grids=combo, label_order=LABEL_ORDER, mode=mode)
        y = np.array([label_to_id[s] for s in y_str], dtype=np.int64)
        X_tr, X_te = X[idx_tr], X[idx_te]

        fast_model = build_pipeline(FAST_C, FAST_GAMMA)
        fast_model.fit(X_tr, y_tr)
        y_pred_fast = fast_model.predict(X_te)
        test_acc_fast = float(accuracy_score(y_te, y_pred_fast))

        rr = dict(r)  # cv情報を保持
        rr["test_acc_fast"] = test_acc_fast
        rows_with_test.append(rr)

    # testでソート（CV上位Kの中での比較）
    rows_with_test_sorted = sorted(rows_with_test, key=lambda r: r["test_acc_fast"], reverse=True)

    # TopのtestランキングCSV + 上位PLOT_TOPKを可視化
    save_rank_csv(OUT_RANK_TOP_TEST, rows_with_test_sorted, include_test=True)
    plot_topk_test_accuracy_hbar(
        OUT_TOPK_TEST,
        rows_with_test_sorted,
        title=f"Top-{PLOT_TOPK} by TEST accuracy (evaluated after CV) [{mode}]",
        topk=PLOT_TOPK
    )

    best_fast = rows_with_test_sorted[0]
    print(f"[{mode}] BEST (fast model among CV top-{len(rows_with_test_sorted)}): grids={best_fast['grids']}, "
          f"cv={best_fast['cv_acc_fast']:.4f}, test={best_fast['test_acc_fast']:.4f}")

    # ------------------------------------------------------------
    # (3) さらに上位候補だけ精密化（SVMのC/gammaをCVで最適化）して最終モデル決定
    #     ※候補は「CV上位Kにtestを付けた集合」から、さらに上位Kを使う
    # ------------------------------------------------------------
    top_rows = rows_with_test_sorted[:max(1, TOPK_FOR_REFINED_SEARCH)]

    best_refined = None
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
            "rank_in_refine_pool": int(rank),
            "grids": combo,
            "feature_dim": int(sum([g ** 3 for g in combo])),
            "best_cv_score": best_cv,
            "best_params": best_params,
            "test_accuracy": test_acc,
            "model": model,
        }

        print(f"\n[{mode} refine {rank}] grids={combo}")
        print(f"  refined_cv={best_cv:.4f}  refined_test={test_acc:.4f}  best_params={best_params}")

        if (best_refined is None) or (cand["test_accuracy"] > best_refined["test_accuracy"]):
            best_refined = cand

    # ------------------------------------------------------------
    # (4) 最終モデル出力（混同行列/レポート/モデル保存）
    # ------------------------------------------------------------
    best_model = best_refined["model"]
    best_grids = best_refined["grids"]
    best_test_acc = best_refined["test_accuracy"]

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
    print(f"feature_dim: {best_refined['feature_dim']}")
    print(f"best_cv_score: {best_refined['best_cv_score']:.4f}")
    print(f"best_params: {best_refined['best_params']}")
    print(f"test_accuracy: {best_test_acc:.4f}")
    print(report)

    save_confusion_matrix_png(OUT_CM_PNG, cm, LABEL_ORDER, dpi=CM_DPI)

    # 保存（モデル）
    payload = {
        "model": best_model,
        "label_order": LABEL_ORDER,
        "grids": best_grids,
        "mode": mode,
        "best_params": best_refined["best_params"],
        "best_cv_score": float(best_refined["best_cv_score"]),
        "test_accuracy": float(best_test_acc),
        "search": {
            "grid_candidates": GRID_CANDIDATES,
            "scale_choices": SCALE_CHOICES,
            "n_combos": len(combos),
            "expected_combos": expected,
            "fast_eval": {"C": FAST_C, "gamma": FAST_GAMMA, "cv_splits": FAST_CV_SPLITS},
            "topk_for_test_eval": TOPK_FOR_TEST_EVAL,
            "topk_for_refined_search": TOPK_FOR_REFINED_SEARCH,
            "refine_svm_params": REFINE_SVM_PARAMS,
        }
    }
    joblib.dump(payload, OUT_MODEL)

    meta = {
        "mode": mode,
        "best_grids": best_grids,
        "test_accuracy": float(best_test_acc),
        "outputs": {
            "sweep_json": str(OUT_SWEEP_JSON),
            "cv_hist_png": str(OUT_CV_HIST),
            "topk_test_png": str(OUT_TOPK_TEST),
            "rank_all_by_cv_csv": str(OUT_RANK_CV_ALL),
            "rank_top_by_test_csv": str(OUT_RANK_TOP_TEST),
            "confusion_matrix_png": str(OUT_CM_PNG),
            "model_joblib": str(OUT_MODEL),
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
        print(f"  cv_hist  ={s['outputs']['cv_hist_png']}")
        print(f"  topk_test={s['outputs']['topk_test_png']}")
        print(f"  rank_cv  ={s['outputs']['rank_all_by_cv_csv']}")
        print(f"  rank_test={s['outputs']['rank_top_by_test_csv']}")
        print(f"  cm_png   ={s['outputs']['confusion_matrix_png']}")
        print(f"  model    ={s['outputs']['model_joblib']}")


if __name__ == "__main__":
    main()
