# soundML_train_SVM.py
# ---------------------------------------------------------
# 機能:
#  1) ラベル別wavフォルダから、FFT特徴量CSVを自動生成（未作成なら）
#  2) CSVを読み込んで train/test split（比率は固定でOKなら定数で）
#  3) SVM（program09準拠: poly, degree=2）で学習
#  4) 評価（accuracy, confusion_matrix, classification_report）
#  5) model.joblib と meta.json（NFFT, fmax, sr, labelsなど）を保存
# ---------------------------------------------------------

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
from joblib import dump

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# ===== 固定設定（必要ならここだけ変更）=====
RANDOM_STATE = 42
TEST_SIZE = 0.3          # 例: 0.2 = 8:2
FMAX = 2000.0            # 例: 5000Hzまで使う
TARGET_SR = 48000        # wavのサンプリング周波数が全て同一である前提
USE_LOG1P = True         # 振幅をlog1pにするか
ZERO_MEAN = True         # 平均を引くか
WINDOW = "hann"          # 窓関数
# =========================================


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
        x = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    elif sampwidth == 4:
        x = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
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


def next_pow2(n: int) -> int:
    if n <= 1:
        return 1
    return 1 << (int(n - 1).bit_length())


def collect_labeled_wavs(wav_root: Path) -> List[Sample]:
    """
    [機能] wav_root 配下の「サブフォルダ名＝ラベル」としてwavを収集
    """
    if not wav_root.exists():
        raise FileNotFoundError(f"wav_root not found: {wav_root}")

    samples: List[Sample] = []
    for label_dir in sorted([d for d in wav_root.iterdir() if d.is_dir()]):
        label = label_dir.name
        wavs = sorted([p for p in label_dir.rglob("*.wav") if p.is_file()])
        for w in wavs:
            samples.append(Sample(path=w, label=label))

    if not samples:
        raise ValueError(f"No wav files found under: {wav_root}")
    return samples


def compute_global_nfft(samples: List[Sample]) -> Tuple[int, int]:
    """
    [機能] 全サンプルを走査し、最大長（サンプル数）を基準に NFFT を決める。
    program09_02.py の考え方（L以上の最小の2べき）に準拠。:contentReference[oaicite:2]{index=2}
    """
    max_len = 0
    sr0 = None

    for s in samples:
        x, sr = read_wav_mono_float32(s.path)
        if sr0 is None:
            sr0 = sr
        if sr != sr0:
            raise ValueError(f"Mixed sample rates: {sr0} vs {sr} (file={s.path})")
        if sr != TARGET_SR:
            raise ValueError(f"Unexpected sr={sr}, expected={TARGET_SR} (file={s.path})")

        max_len = max(max_len, len(x))

    nfft = next_pow2(max_len)
    return nfft, int(sr0 if sr0 is not None else TARGET_SR)


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

    freqs = np.fft.rfftfreq(nfft, d=1.0 / sr)
    mask = freqs <= float(fmax)
    feat = mag[mask]

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
    （ラベルが最後列という点は program09_01.py の作りに合わせます）:contentReference[oaicite:3]{index=3}
    """
    samples = collect_labeled_wavs(wav_root)
    nfft, sr = compute_global_nfft(samples)

    # 1本だけ作って次元数確定
    feat0 = wav_to_fft_feature(samples[0].path, nfft=nfft, sr=sr, fmax=fmax)
    feat_dim = int(feat0.shape[0])

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        header = [f"feat_{i:04d}" for i in range(feat_dim)] + ["label"]
        w.writerow(header)

        for s in samples:
            feat = wav_to_fft_feature(s.path, nfft=nfft, sr=sr, fmax=fmax)
            if feat.shape[0] != feat_dim:
                raise RuntimeError("feature_dim mismatch (unexpected).")
            row = [float(v) for v in feat.tolist()] + [s.label]
            w.writerow(row)

    labels = sorted(list({s.label for s in samples}))
    meta = {
        "wav_root": str(wav_root.resolve()),
        "csv_path": str(out_csv.resolve()),
        "labels": labels,
        "sr": int(sr),
        "nfft": int(nfft),
        "fmax": float(fmax),
        "use_log1p": bool(USE_LOG1P),
        "zero_mean": bool(ZERO_MEAN),
        "window": str(WINDOW),
        "n_samples": int(len(samples)),
        "feature_dim": int(feat_dim),
    }
    return meta


def load_csv_dataset(csv_path: Path) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    [機能] FFT特徴量CSVを読み込み、X, y, labels を返す
    """
    import pandas as pd

    df = pd.read_csv(csv_path)
    if "label" not in df.columns:
        raise ValueError("CSV must contain 'label' column.")

    y_str = df["label"].astype(str).to_numpy()
    X = df.drop(columns=["label"]).to_numpy(dtype=np.float32)

    label_names = sorted(list(set(y_str.tolist())))
    label_to_id = {name: i for i, name in enumerate(label_names)}
    y = np.array([label_to_id[s] for s in y_str], dtype=np.int64)

    return X, y, label_names


def build_clf() -> Pipeline:
    """
    [機能] SVM分類器（StandardScaler + SVC）
    program09_01.py の例は poly, degree=2 を使用。:contentReference[oaicite:4]{index=4}
    """
    clf = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(
            kernel="linear", # 変更: linear カーネルに変更
            degree=3,
            C=0.01,
            probability=True,            # predict_probaを使うため（以前のSVM版でも採用）:contentReference[oaicite:5]{index=5}
            class_weight="balanced",
            random_state=RANDOM_STATE,
            break_ties=True,
        )),
    ])
    return clf


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wav_root", type=str, default="ML_wav_dataset",
                        help="ラベル別にwavが入っているルートフォルダ（未完成ならここを指定）")
    parser.add_argument("--train_csv", type=str, default="ML_SVM/learning_fft_dataset.csv",
                        help="生成/利用する学習CSV")
    parser.add_argument("--model_dir", type=str, default="ML_SVM/trained_svm_model",
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
        meta_build = {"csv_path": str(train_csv.resolve())}

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

    # 保存
    dump(clf, model_dir / "model.joblib")

    meta = {
        "label_names": label_names,
        "random_state": int(RANDOM_STATE),
        "test_size": float(TEST_SIZE),
        "accuracy": float(acc),
        "confusion_matrix": cm.tolist(),
        **meta_build,  # sr/nfft/fmax など（CSV生成した場合に入る）
    }
    (model_dir / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    print("=== TRAIN/EVAL DONE (SVM + FFT from CSV) ===")
    print("model_dir:", str(model_dir.resolve()))
    print("train_csv:", str(train_csv.resolve()))
    print("accuracy :", acc)
    print("labels   :", label_names)
    print("confusion_matrix:", cm.tolist())
    print(report)


if __name__ == "__main__":
    main()
