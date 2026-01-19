"""
- 推論用音声を wordWhisper3.py と同様の処理で分割+Whisper文字起こし
- 生成された *_segmented_pydub.txt から transcript を抽出・正規化
- whisper_rule.json（経験分布規則）で「何を発音したか」を推定
"""

import os, re, json, math, unicodedata
from collections import Counter, defaultdict

# ============ 設定（スクリプト内で指定） ============
RULE_JSON = "whisperGrid_output/whisper_rule.json"   # build側の出力
# 推論したい音声（10回発音が入っているwav等）
INFER_AUDIO_PATHS = [
    r"word\10times_pre\pre\cleaned_audio.wav",
]

# wordWhisper3.py が同じフォルダにある前提（importできるように）
WORDWHISPER3_MODULE = "wordWhisper3"

# 平滑化（経験分布で0割りを避ける）
ALPHA = 0.5
# ================================================

# ---- whisperGrid/wordWhisper の正規化（build側と同一にしておく）----
NORMALIZE_NFKC = True
STRIP_TRAILING_PUNCT = True
STRIP_SURROUNDING_QUOTES = True
UNIFY_KANA = True

TRAILING_PUNCT_RE = re.compile(r"[。．\.!,！？\s]+$")
SURROUNDING_QUOTES_RE = re.compile(r'^[「『"“”]+|[」』"“”]+$')

SEGMENT_LINE_RE = re.compile(
    r"^\s*(\d{1,3})\s+(\d+(?:\.\d+)?)-(\d+(?:\.\d+)?)\s+(.+?)\s*$"
)

def kata_to_hira(s: str) -> str:
    res = []
    for ch in s:
        code = ord(ch)
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

def extract_transcript_from_segment_line(line: str) -> str:
    line2 = line.replace("\t", " ").strip()
    m = SEGMENT_LINE_RE.match(line2)
    if not m:
        return ""
    return m.group(4).strip()

def load_transcripts_from_txt(txt_path: str):
    """wordWhisper3.py が出力した *_segmented_pydub.txt から transcript のみ抽出"""
    outs = []
    in_segments = False
    with open(txt_path, "r", encoding="utf-8") as f:
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
            outs.append(normalize_text(tr_raw))
    return outs

def bucketize(tr_norm: str, correct_norm: dict):
    # build側と同じルール
    if tr_norm == "":
        return "no_response"
    for lab, corr in correct_norm.items():
        if tr_norm == corr:
            return lab
    if tr_norm.endswith("かな"):
        return "fish_unexpected"
    tr_no_bar = tr_norm.replace("ー", "")
    if tr_no_bar != tr_norm and tr_no_bar in set(correct_norm.values()):
        return "fish_unexpected"
    return "out_of_context"

def infer_label(transcripts, rule):
    """
    経験分布に基づく推定:
      score[label] = Σ log P(label | transcript_i)
    P(label|t) は学習データでのカウントから推定（ALPHAで平滑化）
    """
    labels = rule["labels"]
    t2lc = rule["transcript_to_label_counts"]
    L = len(labels)

    scores = {lab: 0.0 for lab in labels}

    for t in transcripts:
        cnts = t2lc.get(t, {})
        total = sum(cnts.get(lab, 0) for lab in labels)

        for lab in labels:
            c = cnts.get(lab, 0)
            p = (c + ALPHA) / (total + ALPHA * L) if (total + ALPHA * L) > 0 else (1.0 / L)
            scores[lab] += math.log(p)

    # 最大スコアのラベルを採用
    best = max(scores.items(), key=lambda x: x[1])[0]
    return best, scores

def main():
    # 1) 規則読み込み
    with open(RULE_JSON, "r", encoding="utf-8") as f:
        rule = json.load(f)
    correct_norm = rule["correct_text_norm"]

    # 2) wordWhisper3.py を使って、推論用音声→txt生成
    mod = __import__(WORDWHISPER3_MODULE)

    # wordWhisper3.py は AUDIO_PATHS を回して process_one を呼ぶ構造なので、
    # INFER_AUDIO_PATHS を代入して main() を呼ぶ（そのまま流用）
    mod.AUDIO_PATHS = INFER_AUDIO_PATHS
    mod.main()  # *_segmented_pydub.txt が生成される想定 :contentReference[oaicite:2]{index=2}

    # 3) 生成されたtxtパスは wordWhisper3.py の resolve_output_path 規則に従う
    #    ここでは同じ関数を呼んで特定する
    for audio_path in INFER_AUDIO_PATHS:
        txt_path = mod.resolve_output_path(audio_path)
        if not os.path.exists(txt_path):
            print(f"[ERROR] txt not found: {txt_path}")
            continue

        transcripts = load_transcripts_from_txt(txt_path)

        # 4) 大分類の内訳（参考表示）
        buckets = Counter(bucketize(t, correct_norm) for t in transcripts)

        # 5) 5ラベル推定（経験分布）
        pred, scores = infer_label(transcripts, rule)

        print("=== Inference ===")
        print(f"[audio] {audio_path}")
        print(f"[txt]   {txt_path}")
        print(f"[segments_used] {len(transcripts)}")
        print("[bucket_counts]", dict(buckets))
        print("[transcripts_head]", transcripts[:20])
        print("[scores(log P sum)]", {k: round(v, 4) for k, v in scores.items()})
        print("[predicted_label]", pred)
        print()

if __name__ == "__main__":
    main()
