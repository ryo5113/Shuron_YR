# soundML_train_SVM.py
# ---------------------------------------------------------
# [目的] ラベル別に wav を集めて、FFT特徴量CSVを作り、
#       SVMを学習してモデルを保存
# ---------------------------------------------------------
import argparse
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, ConfusionMatrixDisplay


# ===== 固定設定（必要ならここだけ変更）=====
RANDOM_STATE = 42
TEST_SIZE = 0.2          # 例: 0.2 = 8:2
FMAX = 2000.0            # 例: 5000Hzまで使う
FMIN = 0.0               # 0Hzから使う（等間隔バンド化の下限）
BAND_HZ = 50.0           # 50Hzごとの等間隔バンド幅
TARGET_SR = 48000        # wavのサンプリング周波数が全て同一である前提
USE_LOG1P = True         # 振幅をlog1pにするか
ZERO_MEAN = True         # 平均を引くか
WINDOW = "hann"          # 窓関数
# =====================


@dataclass
class Sample:
    path: Path
    label: str


def read_wav_mono_float32(wav_path: Path) -> Tuple[np.ndarray, int]:
    """
    [機能] wav読み込み（モノラル化・float32化）
    ※ scipy を使わずに wave + numpy でやると面倒なので、np.frombuffer等を避け、
      ここでは scipy が無い環境を想定して最小限にしています。
    """
    import wave

    with wave.open(str(wav_path), "rb") as wf:
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        fr = wf.getframerate()
        n_frames = wf.getnframes()
        raw = wf.readframes(n_frames)

    if sampwidth == 2:
        x = np.frombuffer(raw, dtype=np.int16)
        x = x.astype(np.float32) / 32768.0
    elif sampwidth == 4:
        x = np.frombuffer(raw, dtype=np.int32)
        x = x.astype(np.float32) / 2147483648.0
    else:
        raise ValueError(f"Unsupported sampwidth={sampwidth} (file={wav_path})")

    if n_channels > 1:
        x = x.reshape(-1, n_channels).mean(axis=1)

    return x.astype(np.float32), int(fr)


def make_window(n: int, name: str) -> np.ndarray:
    if name == "hann":
        return np.hanning(n).astype(np.float32)
    if name == "hamming":
        return np.hamming(n).astype(np.float32)
    if name == "rect":
        return np.ones(n, dtype=np.float32)
    raise ValueError(f"Unknown window: {name}")


def mag_to_equal_band_features(
    mag: np.ndarray,
    freqs: np.ndarray,
    fmin: float,
    fmax: float,
    band_hz: float,
) -> np.ndarray:
    """
    [機能] FFT振幅スペクトル mag を、fmin〜fmax を band_hz 等間隔で区切って
          各バンド内の「平均振幅」を特徴量（低次元化）として返す。
          例: fmin=0, fmax=2000, band_hz=50 -> 40次元
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
            feat[i] = float(np.mean(mag[sel]))
        else:
            feat[i] = 0.0

    return feat


def next_pow2(n: int) -> int:
    return 1 if n <= 1 else 2 ** int(math.ceil(math.log2(n)))


def collect_labeled_wavs(wav_root: Path) -> List[Sample]:
    """
    [機能] wav_root/label/*.wav を収集（label=サブフォルダ名）
    """
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


def compute_global_nfft(samples: List[Sample]) -> Tuple[int, int]:
    """
    [機能] 全wavの最大長を見て、共通のnfftを決める（最大長以上の最小の2^k）
    同時にサンプリング周波数もチェック（同一である前提）
    """
    max_len = 0
    sr = None

    for s in samples:
        x, sr_read = read_wav_mono_float32(s.path)
        if sr is None:
            sr = sr_read
        elif sr_read != sr:
            raise ValueError(f"Sampling rate mismatch: {sr_read} vs {sr} (file={s.path})")
        max_len = max(max_len, len(x))

    if sr is None:
        raise RuntimeError("Failed to read any wav files.")
    if sr != TARGET_SR:
        raise ValueError(f"TARGET_SR mismatch: wav sr={sr} but TARGET_SR={TARGET_SR}. Fix TARGET_SR.")

    nfft = next_pow2(max_len)
    return nfft, sr


def wav_to_fft_feature(wav_path: Path, nfft: int, sr: int, fmax: float) -> np.ndarray:
    """
    [機能] wav全区間を使ってFFT特徴量を作る（全サンプルでnfft共通）
      - 長さが短い: ゼロ埋め
      - 長さが長い: そのまま（ただし max_len から決めたnfftより長いケースは想定外）
    """
    x, sr_read = read_wav_mono_float32(wav_path)
    if sr_read != sr:
        raise ValueError(f"sr mismatch: {sr_read} vs {sr} (file={wav_path})")

    if ZERO_MEAN:
        x = x - float(np.mean(x))

    if len(x) > nfft:
        # 学習時に決めたnfft（=最大長基準）を超える入力は、特徴次元が崩れるのでエラーにします
        raise ValueError(f"Input longer than NFFT. len={len(x)} > nfft={nfft} (file={wav_path})")

    x_pad = np.zeros(nfft, dtype=np.float32)
    x_pad[:len(x)] = x

    w = make_window(nfft, WINDOW)
    X = np.fft.rfft(x_pad * w, n=nfft)
    mag = np.abs(X).astype(np.float32)

    freqs = np.fft.rfftfreq(nfft, d=1.0 / sr).astype(np.float32)

    # --- 等間隔バンド（0〜fmax を BAND_HZ ごと）に集約して低次元化 ---
    feat = mag_to_equal_band_features(
        mag=mag,
        freqs=freqs,
        fmin=FMIN,
        fmax=float(fmax),
        band_hz=BAND_HZ,
    )

    if USE_LOG1P:
        feat = np.log1p(feat)

    return feat.astype(np.float32)


def build_fft_csv_from_wavs(
    wav_root: Path,
    out_csv: Path,
    fmax: float,
) -> Dict:
    """
    [機能] wav_root から FFT特徴量CSVを作成。
    CSV列: feat_000, feat_001, ..., feat_N, label（最後列）
    （ラベルが最後列という点は program09_01.py の作りに合わせています）
    """
    samples = collect_labeled_wavs(wav_root)
    nfft, sr = compute_global_nfft(samples)

    # 1本目で次元数を決める
    feat0 = wav_to_fft_feature(samples[0].path, nfft=nfft, sr=sr, fmax=fmax)
    feat_dim = int(len(feat0))

    labels = sorted(list({s.label for s in samples}))

    # CSVヘッダ
    header = [f"feat_{i:03d}" for i in range(feat_dim)] + ["label"]

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8") as f:
        f.write(",".join(header) + "\n")

        for s in samples:
            feat = wav_to_fft_feature(s.path, nfft=nfft, sr=sr, fmax=fmax)
            if len(feat) != feat_dim:
                raise RuntimeError(f"Feature dim mismatch: {len(feat)} vs {feat_dim} (file={s.path})")

            row = [f"{v:.10g}" for v in feat] + [s.label]
            f.write(",".join(row) + "\n")

    meta = {
        "wav_root": str(wav_root.resolve()),
        "csv_path": str(out_csv.resolve()),
        "labels": labels,
        "sr": int(sr),
        "nfft": int(nfft),
        "fmax": float(fmax),
        "fmin": float(FMIN),
        "band_hz": float(BAND_HZ),
        "n_bands": int((float(fmax) - float(FMIN)) / float(BAND_HZ)),
        "use_log1p": bool(USE_LOG1P),
        "zero_mean": bool(ZERO_MEAN),
        "window": str(WINDOW),
        "n_samples": int(len(samples)),
        "feature_dim": int(feat_dim),
    }
    return meta


def load_csv_dataset(csv_path: Path) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    [機能] FFT特徴量CSV
      - 最後列は label
      - 他は float
    """
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)

    # 1行目はヘッダ
    with csv_path.open("r", encoding="utf-8") as f:
        header = f.readline().strip().split(",")
        rows = []
        for line in f:
            cols = line.strip().split(",")
            rows.append(cols)

    if len(rows) == 0:
        raise RuntimeError(f"Empty CSV: {csv_path}")

    feat_cols = header[:-1]
    label_col = header[-1]
    if label_col != "label":
        raise ValueError(f"Last column must be 'label' but got {label_col}")

    X = np.array([[float(c) for c in r[:-1]] for r in rows], dtype=np.float32)
    y_labels = [r[-1] for r in rows]
    label_names = sorted(list(set(y_labels)))
    label_to_id = {lab: i for i, lab in enumerate(label_names)}
    y = np.array([label_to_id[lab] for lab in y_labels], dtype=np.int64)

    return X, y, label_names


def build_clf() -> Pipeline:
    """
    [機能] SVM + StandardScaler を組む
    """
    # パイプラインにするのが一般的だが、推論側で同じ scaler を使いたいのでここでは明示
    # gamma='scale' は sklearn デフォルト（特徴量分散に基づく）
    clf = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(
            kernel="poly", 
            degree=3,
            C=15,# polyカーネルで良い感じだった値に変更(MAX 0.706)
            #C=1,  # linearカーネルで使う初期値(MAX 0.667)
            #C=1, # rbfカーネルで使う初期値(MAX 0.732)
            probability=True,            # predict_probaを使うため（以前のSVM版でも採用）:contentReference[oaicite:5]{index=5}
            class_weight="balanced",
            random_state=RANDOM_STATE,
            break_ties=True,
        )),
    ])
    return clf


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wav_root", type=str, default="ML_wav_dataset_word",
                        help="ラベル別にwavが入っているルートフォルダ（未完成ならここを指定）")
    parser.add_argument("--train_csv", type=str, default="word/learning_fft_dataset.csv",
                        help="生成/利用する学習CSV")
    parser.add_argument("--model_dir", type=str, default="word/trained_svm_model",
                        help="model.joblib / meta.json の保存先")
    args = parser.parse_args()

    wav_root = Path(args.wav_root)
    train_csv = Path(args.train_csv)
    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    # CSVが無ければ wav_root から作る
    if not train_csv.exists():
        meta_build = build_fft_csv_from_wavs(wav_root=wav_root, out_csv=train_csv, fmax=FMAX)
    else:
        # CSVが既にある場合でも、推論用に sr/nfft/fmax 等は必ず保存する
        samples = collect_labeled_wavs(wav_root)
        nfft, sr = compute_global_nfft(samples)

        # feature_dim はCSVから分かる（Xの列数）
        X_tmp, _, _ = load_csv_dataset(train_csv)
        feat_dim = int(X_tmp.shape[1])

        meta_build = {
            "wav_root": str(wav_root.resolve()),
            "csv_path": str(train_csv.resolve()),
            "sr": int(sr),
            "nfft": int(nfft),
            "fmax": float(FMAX),
            "fmin": float(FMIN),
            "band_hz": float(BAND_HZ),
            "n_bands": int((float(FMAX) - float(FMIN)) / float(BAND_HZ)),
            "use_log1p": bool(USE_LOG1P),
            "zero_mean": bool(ZERO_MEAN),
            "window": str(WINDOW),
            "feature_dim": int(feat_dim),
        }

    # 学習
    X, y, label_names = load_csv_dataset(train_csv)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    clf = build_clf()
    clf.fit(X_tr, y_tr)

    y_pred = clf.predict(X_te)

    acc = float(accuracy_score(y_te, y_pred))
    cm = confusion_matrix(y_te, y_pred)
    report = classification_report(y_te, y_pred, target_names=label_names, digits=4)

    # --- confusion matrix を画像として保存 ---
    try:
        import matplotlib.pyplot as plt
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_names)
        disp.plot(values_format="d")
        plt.title("Confusion Matrix")
        plt.tight_layout()
        cm_path = model_dir / "confusion_matrix.png"
        plt.savefig(cm_path, dpi=200)
        plt.close()
    except Exception as e:
        print(f"[WARN] Failed to save confusion matrix image: {e}")

    # 保存
    model_path = model_dir / "model.joblib"
    joblib.dump({"model": clf, "label_names": label_names}, model_path)

    meta = {
        "label_names": label_names,
        "random_state": int(RANDOM_STATE),
        "test_size": float(TEST_SIZE),
        "accuracy": float(acc),
        "confusion_matrix": cm.tolist(),
        **meta_build,  # sr/nfft/fmax など（CSV生成した場合に入る）
    }
    (model_dir / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    (model_dir / "report.txt").write_text(report, encoding="utf-8")

    print(f"[OK] accuracy={acc:.4f}")
    print(report)
    print(f"[SAVED] {model_path}")
    print(f"[SAVED] {model_dir / 'meta.json'}")
    print(f"[SAVED] {model_dir / 'report.txt'}")


if __name__ == "__main__":
    main()
