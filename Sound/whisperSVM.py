# -*- coding: utf-8 -*-
import os
import re
import unicodedata
from collections import Counter
import glob
import matplotlib.pyplot as plt

import joblib
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

# =========================================================
# ここだけ編集（スクリプト内指定）
# =========================================================
ROOT_GLOBS = [
    r"word_Ex1\10times_Ex1_A\**\*_segmented_pydub.txt",
    r"word_Ex1\10times_Ex1_B\**\*_segmented_pydub.txt",
    r"word_Ex1\10times_Ex1_C\**\*_segmented_pydub.txt",
    r"word_Ex1\10times_Ex1_D\**\*_segmented_pydub.txt",

    # 別フォルダも追加で指定できる（例）
    r"word\10times_01\**\*_segmented_pydub.txt",
    r"word\10times_02\**\*_segmented_pydub.txt",
    r"word\10times_03\**\*_segmented_pydub.txt",
    r"word\10times_04\**\*_segmented_pydub.txt",
    r"word\10times_05\**\*_segmented_pydub.txt",
]

USE_NUMS = {1, 2, 3, 4, 5}   # 1〜5だけ使う（6は使わない）

# 正解ラベル（あなたの5ラベル）
LABELS = ["sakana", "shakana", "thakana", "tyakana", "takana"]

MODEL_OUT = "whisper_text_linearsvm.joblib"

# 文字列正規化（表記ゆれ吸収：例「シャカナ」「しゃかな。」→「しゃかな」へ）
NORMALIZE_NFKC = True
STRIP_TRAILING_PUNCT = True         # 末尾の。など除去
STRIP_SURROUNDING_QUOTES = True     # 「さかな」みたいな引用符を除去
UNIFY_KANA = True                   # カタカナ→ひらがな

# 文字n-gram（短い語でも効きやすい）
NGRAM_RANGE = (2, 4)                # 2〜4文字n-gram
# =========================================================


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


def load_dataset(paths):
    X, y = [], []
    stats = {
        "files_total": 0,
        "files_used": 0,
        "files_skipped_label": 0,
        "lines_skipped_nonsegment": 0,
        "samples": 0,
    }

    for p in paths:
        stats["files_total"] += 1
        label = extract_label_from_filename(p)
        if not label:
            stats["files_skipped_label"] += 1
            continue

        in_segments = False
        used_any = False

        with open(p, "r", encoding="utf-8") as f:
            for raw in f:
                line = raw.rstrip("\n")

                if line.strip() == "[segments]":
                    in_segments = True
                    continue
                if not in_segments:
                    continue
                if not line.strip():
                    continue

                transcript_raw = extract_transcript_from_segment_line(line)
                if not transcript_raw:
                    stats["lines_skipped_nonsegment"] += 1
                    continue

                X.append(normalize_text(transcript_raw))
                y.append(label)
                used_any = True

        if used_any:
            stats["files_used"] += 1

    stats["samples"] = len(X)
    return X, y, stats

def keep_by_number(path: str) -> bool:
    p = path.replace("\\", "/")
    m = NUM_FROM_DIR_RE.search(p)
    if not m:
        # 想定外形式は除外（必要なら True にして残す運用も可能）
        return False
    num = int(m.group(2))
    return num in USE_NUMS

def main():
    # 1) 複数globからまとめて拾う
    paths = []
    for g in ROOT_GLOBS:
        paths.extend(glob.glob(g, recursive=True))

    # 2) 番号でフィルタ（1〜5だけ）
    paths = [p for p in paths if keep_by_number(p)]

    def canonical_path(p: str) -> str:
        # 絶対パス化 → 正規化 → 大文字小文字差を吸収（Windows対策）→ 実体参照（シンボリックリンク等対策）
        return os.path.normcase(os.path.realpath(os.path.abspath(os.path.normpath(p))))

    # --- globで集めた直後の paths を dedupe ---
    canon = [canonical_path(p) for p in paths]

    # 重複がどれだけあるか確認（任意：原因調査に有用）
    dup_counts = Counter(canon)
    dups = {k: v for k, v in dup_counts.items() if v > 1}
    print(f"before_dedupe: {len(paths)} files")
    print(f"unique_files: {len(dup_counts)} files")
    print(f"duplicated_files: {len(dups)} (examples: {list(dups.items())[:5]})")

    # 一意化（元のパス文字列を代表として残す）
    seen = set()
    unique_paths = []
    for p in paths:
        cp = canonical_path(p)
        if cp in seen:
            continue
        seen.add(cp)
        unique_paths.append(p)

    paths = sorted(unique_paths)
    print(f"after_dedupe: {len(paths)} files")

    if not paths:
        print("[ERROR] TEXT_FILES に指定したファイルが見つかりません。パスを確認してください。")
        return

    X, y, stats = load_dataset(paths)
    if not X:
        print("[ERROR] サンプルを読み込めませんでした。[segments] 形式や正規表現を確認してください。")
        print("stats:", stats)
        return

    print("=== Dataset summary ===")
    print("stats:", stats)
    print("label_counts:", dict(Counter(y)))
    print("example_X_head:", X[:10])
    print()

    # サンプルが少なすぎると stratify 分割に失敗することがあるため、最低限の安全策
    # （本格運用ではファイル数を増やしてください）
    test_size = 0.2

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y if len(set(y)) > 1 else None
    )

    clf = Pipeline([
        ("tfidf", TfidfVectorizer(analyzer="char", ngram_range=NGRAM_RANGE)),
        ("svm", LinearSVC()),
    ])

    clf.fit(X_train, y_train)

    print("=== Evaluation ===")
    y_pred = clf.predict(X_test)

    print(classification_report(y_test, y_pred, digits=4))

    cm = confusion_matrix(y_test, y_pred, labels=LABELS)
    print("labels order:", LABELS)
    print(cm)

    # --- 画像として表示（matplotlib）---
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=LABELS)
    plt.rcParams["font.size"] = 18
    fig, ax = plt.subplots(figsize=(7, 7))
    disp.plot(ax=ax, values_format="d")  # "d"=整数表示
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=200)
    plt.show()

    joblib.dump(
        {"model": clf, "labels": LABELS, "ngram_range": NGRAM_RANGE},
        MODEL_OUT
    )
    print(f"[OK] saved: {MODEL_OUT}")


if __name__ == "__main__":
    main()
