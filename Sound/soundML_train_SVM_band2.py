# ---------------------------------------------------------
# [目的]
#   被験者別データセット(A〜E)から「被験者ごとに固定の7:3(test)」を作り、
#   1) 被験者別モデル：各被験者のtrainで学習→自分のtestで評価
#   2) 全データモデル：全被験者trainを結合して学習→各被験者testで評価
#      + 全被験者test合算の混同行列/精度も出す
#
#   - バンド幅: 60Hz固定
#   - 内側CV: train側のみで GridSearchCV（既存仕様のまま）
# ---------------------------------------------------------

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import joblib

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


# ===========================
# 実験設定（既存踏襲）
# ===========================
TEST_SIZE = 0.3
RANDOM_STATE = 42

FMIN = 0
FMAX = 8000
WINDOW = "hann"
ZERO_MEAN = True
USE_LOG1P = True

TARGET_SR = 48000
FIXED_NFFT = 65536

BAND_HZ = 60  # 固定

SVM_KERNEL = "rbf"

CV_SPLITS = 5
CV_SHUFFLE = True

C_GRID = [0.001, 0.005, 0.01, 0.05, 0.1, 1.0, 3.0, 5.0, 10.0]
GAMMA_GRID = ["scale"]
CLASS_WEIGHT = "balanced"
PROBABILITY = True


@dataclass
class Sample:
    path: Path
    label: str
    subject: str


def make_window(n: int, name: str) -> np.ndarray:
    name = name.lower()
    if name == "hann":
        return np.hanning(n).astype(np.float32)
    if name == "hamming":
        return np.hamming(n).astype(np.float32)
    if name == "rect":
        return np.ones(n, dtype=np.float32)
    raise ValueError(f"Unknown window: {name}")


def read_wav_mono_float32(wav_path: Path) -> Tuple[np.ndarray, int]:
    import wave
    with wave.open(str(wav_path), "rb") as wf:
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        sr = wf.getframerate()
        n_frames = wf.getnframes()
        raw = wf.readframes(n_frames)

    if sampwidth == 2:
        x = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    elif sampwidth == 4:
        x = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
    else:
        raise ValueError(f"Unsupported sample width: {sampwidth} bytes (file={wav_path})")

    if n_channels > 1:
        x = x.reshape(-1, n_channels).mean(axis=1)

    return x.astype(np.float32), int(sr)


def wav_to_fft_mag(wav_path: Path, nfft: int, sr: int) -> np.ndarray:
    x, sr_read = read_wav_mono_float32(wav_path)
    if sr_read != sr:
        raise ValueError(f"sr mismatch: {sr_read} vs {sr} (file={wav_path})")

    if ZERO_MEAN:
        x = x - float(np.mean(x))

    if len(x) > nfft:
        raise ValueError(f"Input longer than NFFT. len={len(x)} > nfft={nfft} (file={wav_path})")

    x_pad = np.zeros(nfft, dtype=np.float32)
    x_pad[:len(x)] = x

    w = make_window(nfft, WINDOW)
    X = np.fft.rfft(x_pad * w, n=nfft)
    mag = np.abs(X).astype(np.float32)
    return mag


def mag_to_equal_band_features_sum(
    mag: np.ndarray,
    freqs: np.ndarray,
    fmin: float,
    fmax: float,
    band_hz: float,
) -> np.ndarray:
    edges = np.arange(float(fmin), float(fmax) + float(band_hz), float(band_hz), dtype=np.float32)
    n_bands = int(len(edges) - 1)
    feat = np.zeros(n_bands, dtype=np.float32)

    for i in range(n_bands):
        lo = float(edges[i])
        hi = float(edges[i + 1])

        if i == n_bands - 1:
            sel = (freqs >= lo) & (freqs <= hi)
        else:
            sel = (freqs >= lo) & (freqs < hi)

        feat[i] = float(np.sum(mag[sel])) if np.any(sel) else 0.0

    return feat


def build_pipeline() -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(
            kernel=SVM_KERNEL,
            class_weight=CLASS_WEIGHT,
            probability=PROBABILITY,
            random_state=RANDOM_STATE,
            break_ties=True,
        )),
    ])


def save_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def save_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def save_cm_png(path: Path, cm: np.ndarray, label_names: List[str], vmax: int = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.rcParams["font.size"] = 22
    disp = ConfusionMatrixDisplay(cm, display_labels=label_names)
    disp.plot(values_format="d")
    ax = plt.gca()
    ax.tick_params(axis='x', rotation=60)
    if vmax is not None and disp.im_ is not None:
        disp.im_.set_clim(0, vmax)
    plt.tight_layout(pad=1.2)
    plt.savefig(path, dpi=200)
    plt.close()


def collect_subject_label_wavs(subject: str, wav_root: Path) -> List[Sample]:
    # wav_root/label/*.wav
    samples: List[Sample] = []
    if not wav_root.exists():
        raise FileNotFoundError(f"wav_root not found: {wav_root}")

    for label_dir in sorted([p for p in wav_root.iterdir() if p.is_dir()]):
        label = label_dir.name
        for wav_path in sorted(label_dir.glob("*.wav")):
            samples.append(Sample(path=wav_path, label=label, subject=subject))

    if len(samples) == 0:
        raise RuntimeError(f"No wav files found under: {wav_root}")

    return samples


def fit_gridsearch(X_train: np.ndarray, y_train: np.ndarray) -> GridSearchCV:
    cv = StratifiedKFold(n_splits=CV_SPLITS, shuffle=CV_SHUFFLE, random_state=RANDOM_STATE)
    param_grid = {"svm__C": C_GRID, "svm__gamma": GAMMA_GRID}

    pipe = build_pipeline()
    grid = GridSearchCV(
        estimator=pipe,
        param_grid=param_grid,
        scoring="accuracy",
        cv=cv,
        n_jobs=-1,
        refit=True,
        return_train_score=True,
    )
    grid.fit(X_train, y_train)
    return grid

# ===========================
# SETTINGS（ここだけ編集）
# ===========================
SUBJECT_BASE_DIR = r"C:\Users\edu01\Documents\GitHub\Shuron_YR\Sound\word_Ex1\10times_DataSet"
SUBJECTS = ["A", "B", "C", "D", "E"]
DATASET_PREFIX = "svm_wav_dataset_"
CM_VMAX = 15  # 混同行列の色の最大値（Noneなら自動）

OUT_DIR = r"C:\Users\edu01\Documents\GitHub\Shuron_YR\Sound\word_Ex1\10times_Ex1_A\trained_subject_eval_60Hz"
# ===========================

def main():
    base = Path(SUBJECT_BASE_DIR)
    out_root = Path(OUT_DIR)
    out_root.mkdir(parents=True, exist_ok=True)

    subjects = SUBJECTS

    # 1) 全サンプル収集（subject付き）
    all_samples: List[Sample] = []
    for subj in subjects:
        wav_root = base / f"{DATASET_PREFIX}{subj}"
        all_samples.extend(collect_subject_label_wavs(subj, wav_root))

    label_names = sorted(list({s.label for s in all_samples}))
    label_to_id = {lab: i for i, lab in enumerate(label_names)}
    y_all = np.array([label_to_id[s.label] for s in all_samples], dtype=np.int64)

    # 2) 被験者ごとに固定 train/test を作る
    subj_to_indices: Dict[str, np.ndarray] = {}
    for subj in subjects:
        idxs = np.array([i for i, s in enumerate(all_samples) if s.subject == subj], dtype=np.int64)
        subj_to_indices[subj] = idxs

    subj_split: Dict[str, Dict[str, Any]] = {}
    for subj, idxs in subj_to_indices.items():
        y_sub = y_all[idxs]
        try:
            idx_tr, idx_te, y_tr, y_te = train_test_split(
                idxs, y_sub,
                test_size=TEST_SIZE,
                random_state=RANDOM_STATE,
                stratify=y_sub
            )
        except Exception as e:
            raise RuntimeError(
                f"[split error] subject={subj}: stratified split failed. "
                f"各ラベルのサンプル数が少ない可能性があります。詳細: {e}"
            )
        subj_split[subj] = {"idx_tr": idx_tr, "idx_te": idx_te}

    idx_tr_all = np.concatenate([subj_split[subj]["idx_tr"] for subj in subjects], axis=0)

    # 3) FFT mag を全サンプルで1度だけ計算
    sr = TARGET_SR
    nfft = FIXED_NFFT
    freqs = np.fft.rfftfreq(nfft, d=1.0 / sr).astype(np.float32)

    nyq = sr / 2.0
    if float(FMAX) > nyq:
        raise ValueError(f"FMAX({FMAX}) must be <= Nyquist({nyq}). sr={sr}")

    mags = [wav_to_fft_mag(s.path, nfft=nfft, sr=sr) for s in all_samples]

    # 4) 60Hz特徴量作成
    feats = []
    for mag in mags:
        feat = mag_to_equal_band_features_sum(mag, freqs, FMIN, float(FMAX), float(BAND_HZ))
        if USE_LOG1P:
            feat = np.log1p(feat)
        feats.append(feat)
    X_all = np.stack(feats, axis=0).astype(np.float32)

    # ===========================
    # (I) 全データモデル（train結合）
    # ===========================
    all_model_dir = out_root / "ALL_MODEL"
    all_model_dir.mkdir(parents=True, exist_ok=True)

    grid_all = fit_gridsearch(X_all[idx_tr_all], y_all[idx_tr_all])

    joblib.dump(
        {"model": grid_all.best_estimator_, "label_names": label_names},
        all_model_dir / "model.joblib"
    )

    cm_sum = np.zeros((len(label_names), len(label_names)), dtype=np.int64)
    correct_sum = 0
    total_sum = 0

    per_subj_rows = []

    per_subj_dir = all_model_dir / "per_subject_eval"
    per_subj_dir.mkdir(parents=True, exist_ok=True)

    for subj in subjects:
        idx_te = subj_split[subj]["idx_te"]
        X_te = X_all[idx_te]
        y_te = y_all[idx_te]

        y_pred = grid_all.predict(X_te)
        acc = float(accuracy_score(y_te, y_pred))
        cm = confusion_matrix(y_te, y_pred, labels=np.arange(len(label_names)))

        cm_sum += cm
        correct_sum += int(np.sum(y_pred == y_te))
        total_sum += int(len(y_te))

        sd = per_subj_dir / subj
        sd.mkdir(parents=True, exist_ok=True)
        save_cm_png(sd / "confusion_matrix.png", cm, label_names, vmax=CM_VMAX)
        save_text(sd / "report.txt", classification_report(y_te, y_pred, target_names=label_names, digits=4))

        per_subj_rows.append({"subject": subj, "n_test": int(len(y_te)), "acc_all_model": acc})

    overall_acc = float(correct_sum / total_sum) if total_sum else 0.0
    save_cm_png(all_model_dir / "confusion_matrix_ALL_subject_tests.png", cm_sum, label_names, vmax=CM_VMAX)
    save_json(all_model_dir / "meta.json", {
        "subjects": subjects,
        "labels": label_names,
        "band_hz": float(BAND_HZ),
        "test_size": float(TEST_SIZE),
        "random_state": int(RANDOM_STATE),
        "cv_splits": int(CV_SPLITS),
        "best_params": dict(grid_all.best_params_),
        "cv_best_score_on_all_train": float(grid_all.best_score_),
        "overall_accuracy_on_all_subject_tests": overall_acc,
    })

    with (all_model_dir / "accuracy_per_subject.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["subject", "n_test", "acc_all_model"])
        for r in per_subj_rows:
            w.writerow([r["subject"], r["n_test"], f"{r['acc_all_model']:.6f}"])
        w.writerow(["__OVERALL__", total_sum, f"{overall_acc:.6f}"])

    acc_all_map = {r["subject"]: r["acc_all_model"] for r in per_subj_rows}

    # ===========================
    # (II) 被験者別モデル
    # ===========================
    subj_models_dir = out_root / "SUBJECT_MODELS"
    subj_models_dir.mkdir(parents=True, exist_ok=True)

    compare_rows = []

    for subj in subjects:
        idx_tr = subj_split[subj]["idx_tr"]
        idx_te = subj_split[subj]["idx_te"]

        grid_sub = fit_gridsearch(X_all[idx_tr], y_all[idx_tr])

        y_pred = grid_sub.predict(X_all[idx_te])
        y_te = y_all[idx_te]
        acc_sub = float(accuracy_score(y_te, y_pred))
        cm_sub = confusion_matrix(y_te, y_pred, labels=np.arange(len(label_names)))

        od = subj_models_dir / subj
        od.mkdir(parents=True, exist_ok=True)

        joblib.dump(
            {"model": grid_sub.best_estimator_, "label_names": label_names},
            od / "model.joblib"
        )
        save_cm_png(od / "confusion_matrix.png", cm_sub, label_names, vmax=int(cm_sub.max()))
        save_text(od / "report.txt", classification_report(y_te, y_pred, target_names=label_names, digits=4))
        save_json(od / "meta.json", {
            "subject": subj,
            "labels": label_names,
            "band_hz": float(BAND_HZ),
            "test_size": float(TEST_SIZE),
            "random_state": int(RANDOM_STATE),
            "cv_splits": int(CV_SPLITS),
            "best_params": dict(grid_sub.best_params_),
            "cv_best_score_on_subject_train": float(grid_sub.best_score_),
            "test_accuracy_on_subject_test": acc_sub,
        })

        compare_rows.append({
            "subject": subj,
            "n_test": int(len(y_te)),
            "acc_subject_model": acc_sub,
            "acc_all_model": float(acc_all_map.get(subj, np.nan)),
        })

    with (out_root / "accuracy_compare_subject_vs_all.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["subject", "n_test", "acc_subject_model", "acc_all_model"])
        for r in compare_rows:
            w.writerow([r["subject"], r["n_test"], f"{r['acc_subject_model']:.6f}", f"{r['acc_all_model']:.6f}"])

    print("DONE")
    print(f"Saved to: {out_root}")
    print(f"ALL model overall acc (all subject tests) = {overall_acc:.4f}")


if __name__ == "__main__":
    main()
