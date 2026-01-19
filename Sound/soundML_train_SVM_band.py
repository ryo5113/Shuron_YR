# ---------------------------------------------------------
# [目的]
#   ラベル別に wav を集めて、FFT特徴量を作り、SVMを学習して保存する。
#   本版は「バンド幅 (band_hz)」も含めて、訓練データ内部のCVで選び、
#   最後にホールドアウトテスト(=評価データ)で1回だけ評価する構成。
#
#   - 外側: train/test を 8:2 に固定（テストは“封印”）
#   - 内側: train側のみで GridSearchCV（poly-SVMのC/degree/gamma等）
#   - band_hz は外側ループで比較し、CVスコア最大を採用
#
# 参照: 元の soundML_train_SVM_multi.py の構成を踏襲しつつ、評価設計を変更
# ---------------------------------------------------------
import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Any
import matplotlib
matplotlib.use("Agg")  # GUI(Tk)を使わずPNG保存するため
import matplotlib.pyplot as plt

import numpy as np
import joblib

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# ===========================
# 実験設定（スクリプト内で編集）
# ===========================
# 外側ホールドアウト（教師:評価 = 7:3）
TEST_SIZE = 0.3
RANDOM_STATE = 42

# FFT特徴量
FMIN = 0
FMAX = 8000   # 16kHz収録ならナイキスト=8000Hz
WINDOW = "hann"  # hann / hamming / rect
ZERO_MEAN = True
USE_LOG1P = True

# 固定FFT条件（ユーザー提示）
TARGET_SR = 48000 # サンプリング周波数 [Hz] 
FIXED_NFFT = 65536

# バンド幅候補（band_hz）
BAND_HZ_LIST = [1, 2, 3, 4, 5, 8, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]

# SVM（poly固定）
SVM_KERNEL = "rbf"

# 内側CV（ハイパーパラメータ探索）
CV_SPLITS = 5 
CV_SHUFFLE = True

# 探索グリッド（必要な範囲だけ編集）
C_GRID = [0.001, 0.005, 0.01, 0.05, 0.1, 1.0, 3.0, 5.0, 10.0]
#DEGREE_GRID = [2, 3]
GAMMA_GRID = ["scale"]  # 必要なら ["scale","auto"] などにする
CLASS_WEIGHT = "balanced"
PROBABILITY = True

@dataclass
class Sample:
    path: Path
    label: str

def make_window(n: int, name: str) -> np.ndarray:
    name = name.lower()
    if name == "hann":
        return np.hanning(n).astype(np.float32)
    if name == "hamming":
        return np.hamming(n).astype(np.float32)
    if name == "rect":
        return np.ones(n, dtype=np.float32)
    raise ValueError(f"Unknown window: {name}")

def collect_labeled_wavs(wav_root: Path) -> List[Sample]:
    """wav_root/label/*.wav を収集（label=サブフォルダ名）"""
    samples: List[Sample] = []
    if not wav_root.exists():
        raise FileNotFoundError(f"wav_root not found: {wav_root}")

    for label_dir in sorted([p for p in wav_root.iterdir() if p.is_dir()]):
        label = label_dir.name
        for wav_path in sorted(label_dir.glob("*.wav")):
            samples.append(Sample(path=wav_path, label=label))

    if len(samples) == 0:
        raise RuntimeError(f"No wav files found under: {wav_root}")

    return samples

def read_wav_mono_float32(wav_path: Path) -> Tuple[np.ndarray, int]:
    """
    wave標準ライブラリで読み込み（モノラル化してfloat32へ）
    16bit/32bit PCM を想定。
    """
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
    """
    wav全区間から rFFT 振幅(mag) を作る（nfft固定）
      - 長さが短い: ゼロ埋め
      - 長さが長い: エラー（特徴次元が崩れるため）
    """
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
    """
    FFT振幅スペクトル mag を、fmin〜fmax を band_hz 等間隔で区切って
    各バンド内の「振幅和」を特徴量として返す。
    """
    edges = np.arange(float(fmin), float(fmax) + float(band_hz), float(band_hz), dtype=np.float32)
    n_bands = int(len(edges) - 1)
    feat = np.zeros(n_bands, dtype=np.float32)

    for i in range(n_bands):
        lo = float(edges[i])
        hi = float(edges[i + 1])

        # 最終バンドだけ上端(hi)を含める
        if i == n_bands - 1:
            sel = (freqs >= lo) & (freqs <= hi)
        else:
            sel = (freqs >= lo) & (freqs < hi)

        if np.any(sel):
            feat[i] = float(np.sum(mag[sel]))
        else:
            feat[i] = 0.0

    return feat

def build_pipeline() -> Pipeline:
    """StandardScaler + poly-SVM（パラメータはGridSearchで上書き）"""
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

def save_confusion_matrix_png(path: Path, cm: np.ndarray, label_names: List[str], vmax: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        plt.rcParams["font.size"] = 16
        disp = ConfusionMatrixDisplay(cm, display_labels=label_names)
        disp.plot(values_format="d")
        # ここで色スケールを固定（imshowのclimを固定）
        if disp.im_ is not None:
            disp.im_.set_clim(0, vmax)

        # 既に作られているcolorbarにも反映（作成済みなら）
        fig = plt.gcf()
        ax = plt.gca()
        ax.tick_params(axis='x', rotation=45)

        plt.tight_layout(pad=1.2)
        plt.savefig(path, dpi=200)
        plt.close()
    except Exception as e:
        save_text(path.with_suffix(".error.txt"), str(e))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wav_root", type=str, default="word_Ex1/svm_wav_dataset_all",
                        help="ラベル別にwavが入っているルートフォルダ")
    parser.add_argument("--model_dir", type=str, default="word_Ex1//trained_all_svm_model_band",
                        help="出力先フォルダ")
    args = parser.parse_args()

    wav_root = Path(args.wav_root)
    out_root = Path(args.model_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    # 1) データ収集
    samples = collect_labeled_wavs(wav_root)
    label_names = sorted(list({s.label for s in samples}))
    label_to_id = {lab: i for i, lab in enumerate(label_names)}
    y = np.array([label_to_id[s.label] for s in samples], dtype=np.int64)

    # 2) 外側ホールドアウト分割（この test は最終評価専用）
    idx_all = np.arange(len(samples))
    idx_tr, idx_te, y_tr, y_te = train_test_split(
        idx_all, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    # 3) FFT mag を全サンプルで1度だけ計算（band_hzごとに再計算しない）
    sr = TARGET_SR
    nfft = FIXED_NFFT
    freqs = np.fft.rfftfreq(nfft, d=1.0 / sr).astype(np.float32)

    # ナイキストチェック（等号は許容：rfftはNyquist binを持つ）
    nyq = sr / 2.0
    if float(FMAX) > nyq:
        raise ValueError(f"FMAX({FMAX}) must be <= Nyquist({nyq}). sr={sr}")

    mags: List[np.ndarray] = []
    for s in samples:
        mags.append(wav_to_fft_mag(s.path, nfft=nfft, sr=sr))

    # 4) band_hz ごとに特徴量作成train側のみでGridSearchCV（内側CV）CVスコアで band_hz を比較
    cv = StratifiedKFold(n_splits=CV_SPLITS, shuffle=CV_SHUFFLE, random_state=RANDOM_STATE)

    param_grid = {
        "svm__C": C_GRID,
        #"svm__degree": DEGREE_GRID,
        "svm__gamma": GAMMA_GRID,
    }

    sweep_rows: List[Dict[str, Any]] = []
    best_overall = None  # (band_hz, best_cv_score, grid_obj, feature_dim)
    band_results = {} 

    for band_hz in BAND_HZ_LIST:
        feats = []
        for mag in mags:
            feat = mag_to_equal_band_features_sum(
                mag=mag,
                freqs=freqs,
                fmin=FMIN,
                fmax=float(FMAX),
                band_hz=float(band_hz),
            )
            if USE_LOG1P:
                feat = np.log1p(feat)
            feats.append(feat)

        X = np.stack(feats, axis=0).astype(np.float32)
        X_tr = X[idx_tr]
        X_te = X[idx_te]

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

        grid.fit(X_tr, y_tr)

        y_pred_te_band = grid.predict(X_te)
        cm_te_band = confusion_matrix(y_te, y_pred_te_band)
        test_acc_band = float(accuracy_score(y_te, y_pred_te_band))

        band_dir = out_root / f"band_{int(band_hz):03d}Hz"
        save_confusion_matrix_png(
            band_dir / "confusion_matrix_test.png",
            cm_te_band,
            label_names,
            vmax=int(np.bincount(y_te).max())
        )

        best_cv = float(grid.best_score_)
        best_params = dict(grid.best_params_)
        feat_dim = int(X.shape[1])

        sweep_rows.append({
            "band_hz": float(band_hz),
            "feature_dim": feat_dim,
            "cv_mean_accuracy": best_cv,
            "test_accuracy": test_acc_band,
            "best_params": best_params,
        })
        band_results[float(band_hz)] = {
            "grid": grid,                 # GridSearchCV（best_estimator_を含む）
            "feature_dim": feat_dim,
            "cv_mean_accuracy": best_cv,
            "test_accuracy": test_acc_band,
            "best_params": best_params,
            "X_te": X_te,                 # テスト評価・混同行列作成用
        }

        if (best_overall is None) or (best_cv > best_overall["cv_mean_accuracy"]):
            best_overall = {
                "band_hz": float(band_hz),
                "feature_dim": feat_dim,
                "cv_mean_accuracy": best_cv,
                "test_accuracy": test_acc_band,
                "best_params": best_params,
                "grid": grid,
                "X_te": X_te,  # 保存用
            }

        print(f"[band_hz={band_hz:>5}] CV(best)={best_cv:.4f} TEST={test_acc_band:.4f} feat_dim={feat_dim} params={best_params}")

    assert best_overall is not None

    def save_model_bundle(
        out_dir: Path,
        grid_obj: GridSearchCV,
        band_hz: float,
        feature_dim: int,
        cv_mean_accuracy: float,
        test_accuracy: float,
        best_params: Dict[str, Any],
        X_te_local: np.ndarray,
        reasons: List[str],
    ) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)

        # 推論用モデル保存
        joblib.dump(
            {"model": grid_obj.best_estimator_, "label_names": label_names},
            out_dir / "model.joblib"
        )

        # テストでのレポート・混同行列
        y_pred_local = grid_obj.predict(X_te_local)
        cm_local = confusion_matrix(y_te, y_pred_local)
        report_local = classification_report(y_te, y_pred_local, target_names=label_names, digits=4)

        save_text(out_dir / "report.txt", report_local)

        try:
            plt.rcParams["font.size"] = 18
            disp = ConfusionMatrixDisplay(cm_local, display_labels=label_names)
            disp.plot(values_format="d")
            plt.tight_layout()
            plt.savefig(out_dir / "confusion_matrix.png", dpi=200)
            plt.close()
        except Exception as e:
            save_text(out_dir / "confusion_matrix_error.txt", str(e))

        meta_local = {
            "wav_root": str(wav_root),
            "n_samples": int(len(samples)),
            "labels": label_names,
            "outer_holdout": {
                "test_size": float(TEST_SIZE),
                "random_state": int(RANDOM_STATE),
            },
            "inner_cv": {
                "cv_splits": int(CV_SPLITS),
                "shuffle": bool(CV_SHUFFLE),
                "random_state": int(RANDOM_STATE),
                "scoring": "accuracy",
            },
            "fft": {
                "sr": int(sr),
                "nfft": int(nfft),
                "fmin": float(FMIN),
                "fmax": float(FMAX),
                "window": WINDOW,
                "zero_mean": bool(ZERO_MEAN),
                "use_log1p": bool(USE_LOG1P),
            },
            "feature": {
                "band_hz": float(band_hz),
                "aggregation": "sum(|X_k|) over bins in band",
                "feature_dim": int(feature_dim),
            },
            "svm": {
                "kernel": SVM_KERNEL,
                **best_params,
                "class_weight": CLASS_WEIGHT,
                "probability": bool(PROBABILITY),
            },
            "cv_mean_accuracy": float(cv_mean_accuracy),
            "test_accuracy": float(test_accuracy),
            "export_reasons": reasons,
        }
        save_json(out_dir / "meta.json", meta_local)

    # 60Hzは必ず出力
    MUST_EXPORT_BAND = 60.0

    # sweep_rows から上位3を抽出（同値は band_hz が小さい方を先にする）
    top_cv = sorted(sweep_rows, key=lambda r: (-r["cv_mean_accuracy"], r["band_hz"]))[:3]
    top_te = sorted(sweep_rows, key=lambda r: (-r["test_accuracy"], r["band_hz"]))[:3]

    export_reasons = {}  # band_hz -> reasons(list)
    def add_reason(b: float, reason: str):
        export_reasons.setdefault(float(b), [])
        if reason not in export_reasons[float(b)]:
            export_reasons[float(b)].append(reason)

    add_reason(MUST_EXPORT_BAND, "must_export_60Hz")
    for r in top_cv:
        add_reason(r["band_hz"], "top3_cv_mean_accuracy")
    for r in top_te:
        add_reason(r["band_hz"], "top3_test_accuracy")

    export_bands = sorted(export_reasons.keys())

    export_root = out_root / "EXPORTED_models"
    export_root.mkdir(parents=True, exist_ok=True)

    for b in export_bands:
        if b not in band_results:
            print(f"[WARN] band_results not found for band_hz={b}")
            continue

        info = band_results[b]
        out_dir = export_root / f"band_{int(b):03d}Hz"
        save_model_bundle(
            out_dir=out_dir,
            grid_obj=info["grid"],
            band_hz=b,
            feature_dim=info["feature_dim"],
            cv_mean_accuracy=info["cv_mean_accuracy"],
            test_accuracy=info["test_accuracy"],
            best_params=info["best_params"],
            X_te_local=info["X_te"],
            reasons=export_reasons[b],
        )

    print(f"[EXPORT] saved {len(export_bands)} models under: {export_root}")

    # 5) CVで選ばれた band_hz & params のモデルを、テストで1回だけ評価
    best_band = best_overall["band_hz"]
    best_grid: GridSearchCV = best_overall["grid"]
    X_te_best = best_overall["X_te"]

    y_pred = best_grid.predict(X_te_best)
    test_acc = float(accuracy_score(y_te, y_pred))
    cm = confusion_matrix(y_te, y_pred)
    report = classification_report(y_te, y_pred, target_names=label_names, digits=4)

    # 6) 保存
    # (a) スイープ結果（bandごとのCVベスト）
    save_json(out_root / "band_sweep_cv_results.json", sweep_rows)
    with (out_root / "band_sweep_cv_results.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["band_hz", "feature_dim", "cv_mean_accuracy", "test_accuracy", "best_params_json"])
        for r in sweep_rows:
            w.writerow([r["band_hz"], r["feature_dim"], r["cv_mean_accuracy"], r["test_accuracy"], json.dumps(r["best_params"], ensure_ascii=False)])

    # (b) 最良モデル一式
    best_dir = out_root / f"BEST_band_{int(best_band):03d}Hz"
    best_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(
        {"model": best_grid.best_estimator_, "label_names": label_names},
        best_dir / "model.joblib"
    )

    meta = {
        "wav_root": str(wav_root),
        "n_samples": int(len(samples)),
        "labels": label_names,
        "outer_holdout": {
            "test_size": float(TEST_SIZE),
            "random_state": int(RANDOM_STATE),
        },
        "inner_cv": {
            "cv_splits": int(CV_SPLITS),
            "shuffle": bool(CV_SHUFFLE),
            "random_state": int(RANDOM_STATE),
            "scoring": "accuracy",
        },
        "fft": {
            "sr": int(sr),
            "nfft": int(nfft),
            "fmin": float(FMIN),
            "fmax": float(FMAX),
            "window": WINDOW,
            "zero_mean": bool(ZERO_MEAN),
            "use_log1p": bool(USE_LOG1P),
        },
        "feature": {
            "band_hz": float(best_band),
            "aggregation": "sum(|X_k|) over bins in band",
            "feature_dim": int(best_overall["feature_dim"]),
        },
        "svm": {
            "kernel": SVM_KERNEL,
            **best_overall["best_params"],
            "class_weight": CLASS_WEIGHT,
            "probability": bool(PROBABILITY),
        },
        "cv_best_mean_accuracy": float(best_overall["cv_mean_accuracy"]),
        "test_accuracy": test_acc,
    }
    save_json(best_dir / "meta.json", meta)
    save_text(best_dir / "report.txt", report)

    # 混同行列画像
    try:
        plt.rcParams["font.size"] = 18
        disp = ConfusionMatrixDisplay(cm, display_labels=label_names)
        disp.plot(values_format="d")
        plt.tight_layout()
        plt.savefig(best_dir / "confusion_matrix.png", dpi=200)
        plt.close()
    except Exception as e:
        save_text(best_dir / "confusion_matrix_error.txt", str(e))

    print("\n======================")
    print(f"BEST band_hz = {best_band} Hz (TEST acc = {best_overall['test_accuracy']:.4f}, CV mean acc = {best_overall['cv_mean_accuracy']:.4f})")
    print(f"Saved to: {best_dir}")
    print("======================\n")


if __name__ == "__main__":
    main()
