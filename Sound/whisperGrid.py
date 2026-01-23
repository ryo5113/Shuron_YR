"""
- 既存の *_segmented_pydub.txt から、
  (A) 大分類（5ラベル + 無反応 + 魚系想定外 + 文脈外）の行列画像
  (B) 魚系想定外 / 文脈外 の詳細（上位K）画像
  (C) 推定規則（P(label|transcript) の経験分布）を JSON に保存
"""

import os, re, glob, json, unicodedata
from collections import Counter, defaultdict

import numpy as np
import matplotlib.pyplot as plt

# =========================
# 設定（スクリプト内で指定）
# =========================
ROOT_GLOBS = [
    r"word_Ex1\10times_Ex1_A\**\*_segmented_pydub.txt",
    r"word_Ex1\10times_Ex1_B\**\*_segmented_pydub.txt",
    r"word_Ex1\10times_Ex1_C\**\*_segmented_pydub.txt",
    r"word_Ex1\10times_Ex1_D\**\*_segmented_pydub.txt",
    r"word\10times_01\**\*_segmented_pydub.txt",
    r"word\10times_02\**\*_segmented_pydub.txt",
    r"word\10times_03\**\*_segmented_pydub.txt",
    r"word\10times_04\**\*_segmented_pydub.txt",
    r"word\10times_05\**\*_segmented_pydub.txt",
]
USE_NUMS = {1, 2, 3, 4, 5}

# 画像出力
OUT_DIR = "./whisperGrid_output"
OUT_MAIN_PNG = os.path.join(OUT_DIR, "whisper_group_matrix.png")
OUT_FISH_DETAIL_PNG = os.path.join(OUT_DIR, "whisper_fish_detail.png")
OUT_OOC_DETAIL_PNG  = os.path.join(OUT_DIR, "whisper_ooc_detail.png")

# 詳細図に載せる誤り文字列の列数（上位K）
TOPK_DETAIL = 20

# 規則ファイル
OUT_RULE_JSON = os.path.join(OUT_DIR, "whisper_rule.json")

# 日本語フォント（環境により変えてください）
# 文字化けする場合はここを変更（例: "Meiryo", "MS Gothic" など）
PREFERRED_FONTS = ["Meiryo", "MS Gothic", "IPAexGothic", "Noto Sans CJK JP"]
# =========================


# ===== ラベルと正解文字列（あなたの確定） =====
LABELS = ["sakana", "shakana", "thakana", "tyakana", "takana"]
CORRECT_TEXT = {
    "sakana":  "さかな",
    "shakana": "しゃかな",
    "thakana": "すぁかな",
    "tyakana": "ちゃかな",
    "takana":  "たかな",
}

# ===== whisperGrid.py と同系統の抽出・正規化（崩さない想定） =====
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

def _set_jp_font():
    # 使えるフォントがあれば設定（なければデフォルト）
    for f in PREFERRED_FONTS:
        try:
            plt.rcParams["font.family"] = f
            return
        except Exception:
            pass

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

# ===== 大分類（あなたの定義） =====
CORRECT_NORM = {lab: normalize_text(txt) for lab, txt in CORRECT_TEXT.items()}

def is_fish_unexpected(tr_norm: str) -> bool:
    """
    魚系想定外誤認識の判定ルール（ユーザー指定）:
    - 語末が「かな」
    - または「ー」を除去すると5ラベルのいずれかに一致（=長音混入）
    ※ただし完全一致（5ラベル）は先に除外して使う想定
    """
    if not tr_norm:
        return False

    if tr_norm.endswith("かな"):
        return True

    tr_no_bar = tr_norm.replace("ー", "")
    if tr_no_bar != tr_norm:
        if tr_no_bar in set(CORRECT_NORM.values()):
            return True

    return False

def classify_bucket(tr_norm: str):
    """
    戻り値:
      - bucket: one of
        "sakana|shakana|thakana|tyakana|takana|fish_unexpected|out_of_context"
      - fine: 具体文字列（詳細表用）
    """

    # 5ラベル完全一致（正しい/他ラベルへの誤認識 どちらもここに入る）
    for lab, corr in CORRECT_NORM.items():
        if tr_norm == corr:
            return lab, tr_norm

    if is_fish_unexpected(tr_norm):
        return "fish_unexpected", tr_norm

    return "out_of_context", tr_norm

def plot_matrix(mat, row_labels, col_labels, out_png, title, show_text=True):
    _set_jp_font()

    # ---- 追加: 体裁パラメータ（予稿向けに調整しやすくする） ----
    BASE_FIG_W = 7.0
    PER_COL_W  = 0.35     # 小さめ（列が増えても横長になりにくい）
    FIG_H      = 4.2
    TICK_FONTSIZE = 16    # 軸ラベル（小さめ）
    TITLE_FONTSIZE = 16
    AXIS_LABEL_FONTSIZE = 16
    CELL_FONTSIZE = 16     # ★ セル内数値だけ大きく
    ROTATION = 30          # 45→30（余白を減らす）

    fig_w = max(BASE_FIG_W, PER_COL_W * len(col_labels))
    fig, ax = plt.subplots(figsize=(fig_w, FIG_H))
    im = ax.imshow(mat, aspect="auto")

    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=TICK_FONTSIZE)

    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=ROTATION, ha="right", fontsize=TICK_FONTSIZE)

    ax.set_title(title, fontsize=TITLE_FONTSIZE)
    ax.set_xlabel("Transcript Result", fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_ylabel("True label", fontsize=AXIS_LABEL_FONTSIZE)

    # セル内数値（頻度）を大きく
    if show_text:
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                v = int(mat[i, j])
                if v != 0:
                    ax.text(j, i, str(v), ha="center", va="center", fontsize=CELL_FONTSIZE)

    # カラーバーは小さめに（幅を取りすぎない）
    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cbar.ax.tick_params(labelsize=TICK_FONTSIZE)

    # 余白を削って保存（予稿向け）
    plt.savefig(out_png, dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"[OK] saved: {out_png}")


def main():
    paths = collect_paths()
    if not paths:
        print("[ERROR] 対象ファイルがありません。ROOT_GLOBS/USE_NUMSを確認してください。")
        return
    
    os.makedirs(OUT_DIR, exist_ok=True)

    # (1) まず全文字列カウント（経験分布の基礎）
    counts_true_trans = {lab: Counter() for lab in LABELS}  # true_label -> Counter(tr_norm)
    total_true = Counter()

    # (2) 大分類カウント（5行×7列）
    buckets = ["sakana", "shakana", "thakana", "tyakana", "takana",
               "fish_unexpected", "out_of_context"]
    group_mat = np.zeros((len(LABELS), len(buckets)), dtype=int)

    # (3) 詳細用（魚系/文脈外の “真ラベル×具体文字列”）
    fish_detail = {lab: Counter() for lab in LABELS}
    ooc_detail  = {lab: Counter() for lab in LABELS}

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
                    # 抽出できない行は無反応扱いにしたいならここを bucket=no_response にする等
                    continue

                tr_norm = normalize_text(tr_raw)
                counts_true_trans[true_label][tr_norm] += 1
                total_true[true_label] += 1

                bucket, fine = classify_bucket(tr_norm)
                j = buckets.index(bucket)
                i = LABELS.index(true_label)
                group_mat[i, j] += 1

                if bucket == "fish_unexpected":
                    fish_detail[true_label][fine] += 1
                elif bucket == "out_of_context":
                    ooc_detail[true_label][fine] += 1

    # ---- 画像(A): 大分類行列 ----
    plot_matrix(
        group_mat,
        row_labels=LABELS,
        col_labels=buckets,
        out_png=OUT_MAIN_PNG,
        title="Whisper grouped matrix",
        show_text=True
    )

    # ---- 画像(B): 魚系詳細（上位K列）----
    fish_cols = []
    seen = set()
    for lab in LABELS:
        for t, _ in fish_detail[lab].most_common(TOPK_DETAIL):
            if t not in seen:
                seen.add(t)
                fish_cols.append(t)

    if fish_cols:
        mat = np.zeros((len(LABELS), len(fish_cols)), dtype=int)
        for i, lab in enumerate(LABELS):
            for j, t in enumerate(fish_cols):
                mat[i, j] = fish_detail[lab][t]
        plot_matrix(mat, LABELS, fish_cols, OUT_FISH_DETAIL_PNG,
                    "Fish-unexpected detail", show_text=True)
    else:
        print("[i] fish_unexpected detail is empty -> skip image")

    # ---- 画像(C): 文脈外詳細（上位K列）----
    ooc_cols = []
    seen = set()
    for lab in LABELS:
        for t, _ in ooc_detail[lab].most_common(TOPK_DETAIL):
            if t not in seen:
                seen.add(t)
                ooc_cols.append(t)

    if ooc_cols:
        mat = np.zeros((len(LABELS), len(ooc_cols)), dtype=int)
        for i, lab in enumerate(LABELS):
            for j, t in enumerate(ooc_cols):
                mat[i, j] = ooc_detail[lab][t]
        plot_matrix(mat, LABELS, ooc_cols, OUT_OOC_DETAIL_PNG,
                    "Out-of-context detail", show_text=True)
    else:
        print("[i] out_of_context detail is empty -> skip image")

    # ---- 規則(JSON): P(label|transcript) の経験分布を保存 ----
    # transcriptごとに各ラベルの出現回数
    trans_to_label_counts = defaultdict(lambda: Counter())
    for lab in LABELS:
        for t, c in counts_true_trans[lab].items():
            trans_to_label_counts[t][lab] += c

    rule = {
        "labels": LABELS,
        "correct_text_norm": CORRECT_NORM,
        "transcript_to_label_counts": {t: dict(cnt) for t, cnt in trans_to_label_counts.items()},
        "true_label_totals": dict(total_true),
        "note": {
            "fish_unexpected_rule": "endswith('かな') OR remove 'ー' matches one of 5 correct strings; others are out_of_context;"
        }
    }

    with open(OUT_RULE_JSON, "w", encoding="utf-8") as f:
        json.dump(rule, f, ensure_ascii=False, indent=2)

    print(f"[OK] saved: {OUT_RULE_JSON}")

if __name__ == "__main__":
    main()
