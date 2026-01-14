# -*- coding: utf-8 -*-
"""
verify_whisperSVM_compatible.py
- whisperSVM.py と完全に同一の:
  (1) 正規化設定・normalize_text()
  (2) ファイル名からのラベル抽出 extract_label_from_filename()
  を使って、学習済みモデルで推論・（可能なら）評価を行う

使い方:
- 下の MODEL_PATH と INPUT_GLOBS をスクリプト内で編集して実行
"""

import os
import re
import glob
import unicodedata
from collections import Counter

import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay


# =========================================================
# ここだけ編集（スクリプト内指定）
# =========================================================
MODEL_PATH = "whisper_text_linearsvm.joblib"

# 推論したい txt を指定（複数OK）
INPUT_GLOBS = [
    r"word_Ex1\10times_Ex1_A\shakana6\shakana6_segmented_pydub.txt",
    r"word_Ex1\10times_Ex1_B\takana6\takana6_segmented_pydub.txt",
    r"word_Ex1\10times_Ex1_C\thakana6\thakana6_segmented_pydub.txt",
    # r"another_folder\**\*_segmented_pydub.txt",
]

# whisperSVM.py 同様：番号で絞りたい場合（例：1〜5のみ）
#USE_NUMS = {1, 2, 3, 4, 5}
# =========================================================


# =========================================================
# ★ここから下は whisperSVM.py と「同一」の実装（コピー）
# =========================================================

# 正解ラベル（あなたの5ラベル）
LABELS = ["sakana", "shakana", "thakana", "tyakana", "takana"]

# 文字列正規化（表記ゆれ吸収：例「シャカナ」「しゃかな。」→「しゃかな」へ）
NORMALIZE_NFKC = True
STRIP_TRAILING_PUNCT = True         # 末尾の。など除去
STRIP_SURROUNDING_QUOTES = True     # 「さかな」みたいな引用符を除去
UNIFY_KANA = True                   # カタカナ→ひらがな

# 末尾につきがちな句読点等
TRAILING_PUNCT_RE = re.compile(r"[。．\.!,！？\s]+$")
# 両端の引用符など
SURROUNDING_QUOTES_RE = re.compile(r'^[「『"“”]+|[」』"“”]+$')

# segments行: "01\t1.49-2.09\tさかな" / "01  1.49-2.09  さかな"
SEGMENT_LINE_RE = re.compile(
    r"^\s*(\d{1,3})\s+(\d+(?:\.\d+)?)-(\d+(?:\.\d+)?)\s+(.+?)\s*$"
)

# ファイル名例: takana4_segmented_pydub.txt -> label="takana"
BASENAME_RE = re.compile(r"^([a-zA-Z]+)\d+_segmented_pydub\.txt$")

NUM_FROM_DIR_RE = re.compile(
    r"[\\/](sakana|shakana|takana|thakana|tyakana)(\d+)[\\/]\1\2_segmented_pydub\.txt$",
    re.IGNORECASE
)

def kata_to_hira(s: str) -> str:
    res = []
    for ch in s:
        code = ord(ch)
        # カタカナ → ひらがな
        if 0x30A1 <= code <= 0x30F4:
            res.append(chr(code - 0x60))
        else:
            res.append(ch)
    return "".join(res)

def normalize_text(text: str) -> str:
    s = text.strip()
    if NORMALIZE_NFKC:
        s = unicodedata.normalize("NFKC", s)

    if STRIP_SURROUNDING_QUOTES:
        # 「」等を複数回剥がす
        while True:
            new_s = SURROUNDING_QUOTES_RE.sub("", s).strip()
            if new_s == s:
                break
            s = new_s

    if STRIP_TRAILING_PUNCT:
        s = TRAILING_PUNCT_RE.sub("", s)

    if UNIFY_KANA:
        s = kata_to_hira(s)

    return s

def extract_label_from_filename(path: str) -> str:
    base = os.path.basename(path)
    m = BASENAME_RE.match(base)
    if not m:
        return ""
    lab = m.group(1).lower()
    return lab if lab in LABELS else ""

def extract_transcript_from_segment_line(line: str) -> str:
    # タブが混ざっても良いように空白化して正規表現にかける
    line2 = line.replace("\t", " ").strip()
    m = SEGMENT_LINE_RE.match(line2)
    if not m:
        return ""
    return m.group(4).strip()

# def keep_by_number(path: str) -> bool:
#     p = path.replace("\\", "/")
#     m = NUM_FROM_DIR_RE.search(p)
#     if not m:
#         return False
#     num = int(m.group(2))
#     return num in USE_NUMS

# =========================================================
# ★ここまでが whisperSVM.py と同一部分
# =========================================================


def canonical_path(p: str) -> str:
    # 重複排除用（Windowsの大小文字/相対差も吸収）
    return os.path.normcase(os.path.realpath(os.path.abspath(os.path.normpath(p))))


def collect_paths() -> list[str]:
    paths = []
    for g in INPUT_GLOBS:
        paths.extend(glob.glob(g, recursive=True))

    # 番号で絞る（1〜5など）
    #paths = [p for p in paths if keep_by_number(p)]

    # 重複排除
    seen = set()
    uniq = []
    for p in paths:
        cp = canonical_path(p)
        if cp in seen:
            continue
        seen.add(cp)
        uniq.append(p)
    return sorted(uniq)


def load_samples(paths: list[str]):
    X, y, meta = [], [], []  # meta: (file, seg_idx, start_end_str, raw_text, norm_text)
    for p in paths:
        label = extract_label_from_filename(p)  # 取れなければ ""（=評価不可だが推論は可能）
        in_segments = False

        with open(p, "r", encoding="utf-8") as f:
            for raw in f:
                line = raw.rstrip("\n")
                if line.strip() == "[segments]":
                    in_segments = True
                    continue
                if not in_segments or not line.strip():
                    continue

                tr_raw = extract_transcript_from_segment_line(line)
                if not tr_raw:
                    continue

                tr_norm = normalize_text(tr_raw)
                X.append(tr_norm)
                y.append(label)
                meta.append((p, tr_raw, tr_norm))
    return X, y, meta


def main():
    obj = joblib.load(MODEL_PATH)
    model = obj["model"]
    labels_order = obj.get("labels", LABELS)

    paths = collect_paths()
    print("files:", len(paths))
    if not paths:
        print("[ERROR] 対象ファイルが0件です。INPUT_GLOBS / USE_NUMS を確認してください。")
        return

    X, y_true, meta = load_samples(paths)
    print("samples:", len(X))
    if not X:
        print("[ERROR] [segments] から文字列が抽出できませんでした。SEGMENT_LINE_RE 等を確認してください。")
        return

    # 推論
    y_pred = model.predict(X)

    # 1) まず「推論結果だけ」一覧（必要最小限）
    print("\n=== Predictions (first 50) ===")
    for i, ((p, raw_text, norm_text), yp) in enumerate(zip(meta, y_pred), 1):
        if i > 50:
            break
        print(f"{i:04d} pred={yp}  input_norm='{norm_text}'  input_raw='{raw_text}'  file='{p}'")

    # 2) 真のラベルが取れるデータだけ評価（ファイル名が規則通りの場合）
    idx_eval = [i for i, yt in enumerate(y_true) if yt in labels_order]
    if not idx_eval:
        print("\n[INFO] ファイル名から正解ラベルが取得できないため、評価（混同行列等）はスキップします。")
        return

    y_eval = [y_true[i] for i in idx_eval]
    p_eval = [y_pred[i] for i in idx_eval]

    print("\n=== Evaluation (only samples with true label from filename) ===")
    print("label_counts:", dict(Counter(y_eval)))
    print(classification_report(y_eval, p_eval, digits=4))

    cm = confusion_matrix(y_eval, p_eval, labels=labels_order)
    print("labels order:", labels_order)
    print(cm)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels_order)
    fig, ax = plt.subplots(figsize=(7, 7))
    disp.plot(ax=ax, values_format="d")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig("confusion_matrix_eval.png", dpi=200)
    plt.show()
    print("[OK] saved: confusion_matrix_eval.png")


if __name__ == "__main__":
    main()
