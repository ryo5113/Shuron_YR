# -*- coding: utf-8 -*-
import os, re, glob, unicodedata
from collections import Counter, defaultdict
import numpy as np
import matplotlib.pyplot as plt

# =========================
# ここだけ編集（入力指定）
# =========================
ROOT_GLOBS = [
    # r"word_Ex1\10times_Ex1_A\**\*_segmented_pydub.txt",
    # r"word_Ex1\10times_Ex1_B\**\*_segmented_pydub.txt",
    # r"word_Ex1\10times_Ex1_C\**\*_segmented_pydub.txt",
    # r"word_Ex1\10times_Ex1_D\**\*_segmented_pydub.txt",
    r"word\10times_01\**\*_segmented_pydub.txt",
    r"word\10times_02\**\*_segmented_pydub.txt",
    r"word\10times_03\**\*_segmented_pydub.txt",
    r"word\10times_04\**\*_segmented_pydub.txt",
    r"word\10times_05\**\*_segmented_pydub.txt",
]
USE_NUMS = {1, 2, 3, 4, 5}  # 例：1〜5だけ使う
TOPK_ERRORS = 10           # 各ラベルで誤り上位K件を表示
OUT_CSV = "whisper_transcript_table_all.csv"
OUT_PNG = "whisper_transcript_matrix_YR.png"
SHOW_COUNTS_TEXT = True  # セル内に数値を描くか

# 正しい文字おこし（あなたの確定）
correct_text = {
    "sakana":  "さかな",
    "shakana": "しゃかな",
    "thakana": "すぁかな",
    "tyakana": "ちゃかな",
    "takana":  "たかな",
}
# =========================

# ===== whisperSVM.py と同等の前提 =====
LABELS = ["sakana", "shakana", "thakana", "tyakana", "takana"]  # :contentReference[oaicite:4]{index=4}

NORMALIZE_NFKC = True
STRIP_TRAILING_PUNCT = True
STRIP_SURROUNDING_QUOTES = True
UNIFY_KANA = True

TRAILING_PUNCT_RE = re.compile(r"[。．\.!,！？\s]+$")
SURROUNDING_QUOTES_RE = re.compile(r'^[「『"“”]+|[」』"“”]+$')

SEGMENT_LINE_RE = re.compile(
    r"^\s*(\d{1,3})\s+(\d+(?:\.\d+)?)-(\d+(?:\.\d+)?)\s+(.+?)\s*$"
)
BASENAME_RE = re.compile(r"^([a-zA-Z]+)\d+_segmented_pydub\.txt$")
NUM_FROM_DIR_RE = re.compile(
    r"[\\/](sakana|shakana|takana|thakana|tyakana)(\d+)[\\/]\1\2_segmented_pydub\.txt$",
    re.IGNORECASE
)

def kata_to_hira(s: str) -> str:
    res = []
    for ch in s:
        code = ord(ch)
        if 0x30A1 <= code <= 0x30F4:  # カタカナ
            res.append(chr(code - 0x60))  # ひらがな
        else:
            res.append(ch)
    return "".join(res)

def normalize_text(text: str) -> str:
    s = text.strip()
    if NORMALIZE_NFKC:
        s = unicodedata.normalize("NFKC", s)
    if STRIP_SURROUNDING_QUOTES:
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
    line2 = line.replace("\t", " ").strip()
    m = SEGMENT_LINE_RE.match(line2)
    if not m:
        return ""
    return m.group(4).strip()

def keep_by_number(path: str) -> bool:
    p = path.replace("\\", "/")
    m = NUM_FROM_DIR_RE.search(p)
    if not m:
        return False
    num = int(m.group(2))
    return num in USE_NUMS

def canonical_path(p: str) -> str:
    return os.path.normcase(os.path.realpath(os.path.abspath(os.path.normpath(p))))

def collect_paths():
    paths = []
    for g in ROOT_GLOBS:
        paths.extend(glob.glob(g, recursive=True))
    paths = [p for p in paths if keep_by_number(p)]

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

def main():
    paths = collect_paths()
    if not paths:
        print("[ERROR] 対象ファイルがありません。ROOT_GLOBS / USE_NUMS を確認してください。")
        return

    # 集計：true_label -> Counter(transcript_norm)
    counts = {lab: Counter() for lab in LABELS}
    total_by_label = Counter()

    # 先に集計する（ここが重要）
    for p in paths:
        true_label = extract_label_from_filename(p)
        if not true_label:
            continue

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

                tr = normalize_text(tr_raw)
                counts[true_label][tr] += 1
                total_by_label[true_label] += 1

    # 正規化後の正解文字列
    correct_norm = {lab: normalize_text(txt) for lab, txt in correct_text.items()}

    # (1) 列（transcript）を作る：全ラベルの「正解」＋「誤り上位K」をユニオン
    columns = []
    seen = set()

    def add_col(s: str):
        if s not in seen:
            seen.add(s)
            columns.append(s)

    for lab in LABELS:
        add_col(correct_norm[lab])

    for lab in LABELS:
        errs = [(t, c) for (t, c) in counts[lab].most_common() if t != correct_norm[lab]]
        for t, _ in errs[:TOPK_ERRORS]:
            add_col(t)

    # (2) 行列を作る：rows=true_label, cols=transcript
    mat = np.zeros((len(LABELS), len(columns)), dtype=int)
    for i, lab in enumerate(LABELS):
        for j, tr in enumerate(columns):
            mat[i, j] = counts[lab][tr]

    # (3) 画像化（ヒートマップ）
    plt.rcParams["font.family"] = "MS Gothic"  # 日本語対応
    plt.rcParams["font.size"] = 12
    fig_w = max(10, 0.6 * len(columns))
    fig_h = 6
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    im = ax.imshow(mat, aspect="auto")

    ax.set_yticks(range(len(LABELS)))
    ax.set_yticklabels(LABELS)
    ax.set_xticks(range(len(columns)))
    ax.set_xticklabels(columns, rotation=45, ha="right")
    ax.set_xlabel("Whisper transcript")
    ax.set_ylabel("True label")
    ax.set_title("Whisper transcript matrix")

    if SHOW_COUNTS_TEXT:
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                v = mat[i, j]
                if v != 0:
                    ax.text(j, i, str(v), ha="center", va="center", fontsize=8)

    fig.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=200)
    plt.close(fig)
    print(f"[OK] saved: {OUT_PNG}")

    # ロング表（CSV用）
    rows = []  # (true_label, transcript_norm, count, is_correct)

    for p in paths:
        true_label = extract_label_from_filename(p)
        if not true_label:
            continue

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

                tr = normalize_text(tr_raw)
                counts[true_label][tr] += 1
                total_by_label[true_label] += 1

    for lab in LABELS:
        for tr, c in counts[lab].most_common():
            is_correct = (tr == normalize_text(correct_text[lab]))
            rows.append((lab, tr, c, is_correct))

    # 表示（各ラベル：正解カウント＋誤り上位）
    print("=== Whisper transcript summary (per true label) ===")
    for lab in LABELS:
        corr = normalize_text(correct_text[lab])
        corr_count = counts[lab][corr]
        print(f"\n[{lab}] total={total_by_label[lab]}  correct('{corr}')={corr_count}")

        # 誤り一覧
        errs = [(t, c) for (t, c) in counts[lab].most_common() if t != corr]
        if not errs:
            print("  errors: none")
        else:
            print(f"  errors(top{TOPK_ERRORS}):")
            for t, c in errs[:TOPK_ERRORS]:
                print(f"    {t}: {c}")

    # CSV保存（1つの表として扱いやすい）
    with open(OUT_CSV, "w", encoding="utf-8") as wf:
        wf.write("true_label,transcript_norm,count,is_correct\n")
        for lab, tr, c, is_correct in rows:
            wf.write(f"{lab},{tr},{c},{int(is_correct)}\n")
    print(f"\n[OK] saved: {OUT_CSV}")

if __name__ == "__main__":
    main()
