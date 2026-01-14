import glob
import os
import re
import unicodedata
from collections import Counter

# =========================================================
# ここを編集して使ってください（コマンドライン指定不要）
# =========================================================
TARGET = "たかな"  # 正しい文字起こし（例: "さかな" / "しゃかな"）
INPUT_GLOBS = [
    "word_Ex1/10times_Ex1_A/takana1/takana1_segmented_pydub.txt",
    "word_Ex1/10times_Ex1_A/takana2/takana2_segmented_pydub.txt",
    "word_Ex1/10times_Ex1_A/takana3/takana3_segmented_pydub.txt",
    "word_Ex1/10times_Ex1_A/takana4/takana4_segmented_pydub.txt",
    "word_Ex1/10times_Ex1_A/takana5/takana5_segmented_pydub.txt",
    "word_Ex1/10times_Ex1_A/takana6/takana6_segmented_pydub.txt",
    "word_Ex1/10times_Ex1_B/takana1/takana1_segmented_pydub.txt",
    "word_Ex1/10times_Ex1_B/takana2/takana2_segmented_pydub.txt",
    "word_Ex1/10times_Ex1_B/takana3/takana3_segmented_pydub.txt",
    "word_Ex1/10times_Ex1_B/takana4/takana4_segmented_pydub.txt",
    "word_Ex1/10times_Ex1_B/takana5/takana5_segmented_pydub.txt",
    "word_Ex1/10times_Ex1_B/takana6/takana6_segmented_pydub.txt",
    "word_Ex1/10times_Ex1_C/takana1/takana1_segmented_pydub.txt",
    "word_Ex1/10times_Ex1_C/takana2/takana2_segmented_pydub.txt",
    "word_Ex1/10times_Ex1_C/takana3/takana3_segmented_pydub.txt",
    "word_Ex1/10times_Ex1_C/takana4/takana4_segmented_pydub.txt",
    "word_Ex1/10times_Ex1_C/takana5/takana5_segmented_pydub.txt",
    "word_Ex1/10times_Ex1_C/takana6/takana6_segmented_pydub.txt",
    "word_Ex1/10times_Ex1_D/takana1/takana1_segmented_pydub.txt",
    "word_Ex1/10times_Ex1_D/takana2/takana2_segmented_pydub.txt",
    "word_Ex1/10times_Ex1_D/takana3/takana3_segmented_pydub.txt",
    "word_Ex1/10times_Ex1_D/takana4/takana4_segmented_pydub.txt",
    "word_Ex1/10times_Ex1_D/takana5/takana5_segmented_pydub.txt",
    "word_Ex1/10times_Ex1_D/takana6/takana6_segmented_pydub.txt",
]

# 表記ゆれの吸収設定（必要に応じて True/False を切り替え）
NORMALIZE_NFKC = True
STRIP_TRAILING_PUNCT = True         # 末尾の「。」「!」など除去
STRIP_SURROUNDING_QUOTES = True     # 「」や " " の両端除去
UNIFY_KANA = True                   # カタカナ→ひらがな
COUNT_WRONG_BY = "normalized"       # "normalized" or "raw"
#   normalized: 表記ゆれをまとめて集計（推奨）
#   raw: 生の文字列そのままで集計（表記ゆれも別パターン扱い）

# 末尾につきがちな句読点・記号
TRAILING_PUNCT_RE = re.compile(r"[。．\.!,！？\s]+$")
# 両端に付きがちな引用符など
SURROUNDING_QUOTES_RE = re.compile(r'^[「『"“”]+|[」』"“”]+$')
# 例: "01\t1.49-2.09\tさかな" / "01  1.49-2.09  さかな" を想定
SEGMENT_LINE_RE = re.compile(
    r"^\s*(\d{1,3})\s+(\d+(?:\.\d+)?)-(\d+(?:\.\d+)?)\s+(.+?)\s*$"
)

def kata_to_hira(s: str) -> str:
    """カタカナをひらがなに寄せる"""
    res = []
    for ch in s:
        code = ord(ch)
        # カタカナ(ァ=0x30A1 .. ヴ=0x30F4) → ひらがな(ぁ=0x3041 .. ゔ=0x3094)
        if 0x30A1 <= code <= 0x30F4:
            res.append(chr(code - 0x60))
        else:
            res.append(ch)
    return "".join(res)


def normalize_text(text: str) -> str:
    """比較・集計用の正規化"""
    s = text.strip()
    if NORMALIZE_NFKC:
        s = unicodedata.normalize("NFKC", s)

    if STRIP_SURROUNDING_QUOTES:
        # 両端の引用符を複数回剥がす
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


def extract_transcript_from_line(line: str) -> str:
    """
    セグメント行から文字起こし部分だけ抜き出す。
    例:
      "01\t1.49-2.09\tさかな" -> "さかな"
      "07  10.66-11.23  かな" -> "かな"
    """
    m = SEGMENT_LINE_RE.match(line.replace("\t", " "))
    if not m:
        return ""
    transcript = m.group(4).strip()
    return transcript

def count_one_file(path: str, target_norm: str):
    """
    [segments] ブロック内のセグメント行だけ数える
    """
    total = 0
    correct = 0
    wrong = Counter()

    in_segments = False

    with open(path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.rstrip("\n")

            # [segments] 開始を検知
            if line.strip() == "[segments]":
                in_segments = True
                continue

            if not in_segments:
                # segmentsより前のメタ情報は無視
                continue

            if not line.strip():
                # 空行は無視
                continue

            # segmentsブロック内でも、セグメント行以外は無視
            transcript_raw = extract_transcript_from_line(line)
            if not transcript_raw:
                continue

            total += 1
            transcript_norm = normalize_text(transcript_raw)

            if transcript_norm == target_norm:
                correct += 1
            else:
                key = transcript_norm if COUNT_WRONG_BY == "normalized" else transcript_raw.strip()
                wrong[key] += 1

    return total, correct, wrong

def main():
    # 入力ファイル展開
    paths = []
    for g in INPUT_GLOBS:
        paths.extend(glob.glob(g))
    paths = sorted(set([p for p in paths if os.path.isfile(p)]))

    if not paths:
        print("入力ファイルが見つかりませんでした。INPUT_GLOBS を確認してください。")
        return

    target_norm = normalize_text(TARGET)

    total_all = 0
    correct_all = 0
    wrong_all = Counter()

    per_file_results = []

    for p in paths:
        total, correct, wrong = count_one_file(p, target_norm)
        per_file_results.append((p, total, correct, wrong))
        total_all += total
        correct_all += correct
        wrong_all.update(wrong)

    # ===== 出力 =====
    print("=== 集計設定 ===")
    print(f"TARGET (raw): {TARGET}")
    print(f"TARGET (normalized): {target_norm}")
    print(f"COUNT_WRONG_BY: {COUNT_WRONG_BY}")
    print()

    print("=== 全体集計 ===")
    print(f"総数: {total_all}")
    print(f"正解数: {correct_all}")
    print(f"誤り数: {total_all - correct_all}")
    print()

    print("=== ファイル別（正解/総数） ===")
    for p, total, correct, _ in per_file_results:
        print(f"- {os.path.basename(p)}: 正解 {correct} / 総数 {total}（誤り {total - correct}）")
    print()

    print("=== 誤りパターン別（全体） ===")
    if wrong_all:
        for text, cnt in wrong_all.most_common():
            print(f"- {text}: {cnt}")
    else:
        print("誤りはありませんでした。")
    print()

    print("=== 誤りパターン別（ファイル別） ===")
    for p, total, correct, wrong in per_file_results:
        print(f"[{os.path.basename(p)}] 誤り {total - correct}")
        if wrong:
            for text, cnt in wrong.most_common():
                print(f"  - {text}: {cnt}")
        else:
            print("  (誤りなし)")
        print()


if __name__ == "__main__":
    main()
