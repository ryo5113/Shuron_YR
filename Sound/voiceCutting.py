from pathlib import Path
from pydub import AudioSegment
from pydub.silence import detect_nonsilent
import numpy as np

# ====== 入力 ======
INPUT_FILES = [
    Path(r"word_Ex1/10times_Ex1_D/sakana1/cleaned_audio.wav"),
    Path(r"word_Ex1/10times_Ex1_D/sakana2/cleaned_audio.wav"),
    Path(r"word_Ex1/10times_Ex1_D/sakana3/cleaned_audio.wav"),
    Path(r"word_Ex1/10times_Ex1_D/sakana4/cleaned_audio.wav"),
    Path(r"word_Ex1/10times_Ex1_D/sakana5/cleaned_audio.wav"),
    Path(r"word_Ex1/10times_Ex1_D/sakana6/cleaned_audio.wav"),
    Path(r"word_Ex1/10times_Ex1_D/shakana1/cleaned_audio.wav"),
    Path(r"word_Ex1/10times_Ex1_D/shakana2/cleaned_audio.wav"),
    Path(r"word_Ex1/10times_Ex1_D/shakana3/cleaned_audio.wav"),
    Path(r"word_Ex1/10times_Ex1_D/shakana4/cleaned_audio.wav"),
    Path(r"word_Ex1/10times_Ex1_D/shakana5/cleaned_audio.wav"),
    Path(r"word_Ex1/10times_Ex1_D/shakana6/cleaned_audio.wav"),
    Path(r"word_Ex1/10times_Ex1_D/takana1/cleaned_audio.wav"),
    Path(r"word_Ex1/10times_Ex1_D/takana2/cleaned_audio.wav"),
    Path(r"word_Ex1/10times_Ex1_D/takana3/cleaned_audio.wav"),
    Path(r"word_Ex1/10times_Ex1_D/takana4/cleaned_audio.wav"),
    Path(r"word_Ex1/10times_Ex1_D/takana5/cleaned_audio.wav"),
    Path(r"word_Ex1/10times_Ex1_D/takana6/cleaned_audio.wav"),
    Path(r"word_Ex1/10times_Ex1_D/thakana1/cleaned_audio.wav"),
    Path(r"word_Ex1/10times_Ex1_D/thakana2/cleaned_audio.wav"),
    Path(r"word_Ex1/10times_Ex1_D/thakana3/cleaned_audio.wav"),
    Path(r"word_Ex1/10times_Ex1_D/thakana4/cleaned_audio.wav"),
    Path(r"word_Ex1/10times_Ex1_D/thakana5/cleaned_audio.wav"),
    Path(r"word_Ex1/10times_Ex1_D/thakana6/cleaned_audio.wav"),
    Path(r"word_Ex1/10times_Ex1_D/tyakana1/cleaned_audio.wav"),
    Path(r"word_Ex1/10times_Ex1_D/tyakana2/cleaned_audio.wav"),
    Path(r"word_Ex1/10times_Ex1_D/tyakana3/cleaned_audio.wav"),
    Path(r"word_Ex1/10times_Ex1_D/tyakana4/cleaned_audio.wav"),
    Path(r"word_Ex1/10times_Ex1_D/tyakana5/cleaned_audio.wav"),
    Path(r"word_Ex1/10times_Ex1_D/tyakana6/cleaned_audio.wav"),
]
# ===============

# 無音判定パラメータ（ここは固定のまま）
MIN_SILENCE_LEN_MS = 250
SILENCE_THRESH_DBFS = -62.0
KEEP_SILENCE_MS = 100

# 後処理パラメータ（「閾値」ではなく後処理ルール）
TARGET_COUNT = 10          # 10回発音が前提
FRAME_MS = 5               # エネルギー計算フレーム
TRIM_REL_DB = 30.0         # 区間内ピークから -25dB までを「有音」とみなして前後を詰める
MAX_CHUNK_MS = 1000     # これより長いchunkは「2回分結合の疑い」で分割を試す
MIN_CHUNK_MS = 120         # これより短いchunkはノイズ疑い（今回は削除せずそのまま残す）
VALLEY_DROP_DB = 20.0      # 分割点候補：ピークより -20dB 以下の谷を優先
MIN_GAP_MS = 80            # 分割点の前後に最低これだけ余白がある谷を採用
MIN_AVG_DBFS = -35.0         # 平均音量が小さすぎるものは捨てる
MIN_PEAK_DBFS = -25.0        # ピークが小さすぎるものは捨てる（ノイズだけの区間対策）
SAVE_REJECTED = False   # False: 除外chunkは保存しない / True: rejectedフォルダに保存

def is_valid_chunk(chunk: AudioSegment) -> bool:
    """音声が含まれていない（ノイズのみ等）chunkを弾くための簡易判定"""
    if len(chunk) < MIN_CHUNK_MS:
        return False
    # dBFS / max_dBFS は AudioSegment が提供する指標
    if chunk.dBFS < MIN_AVG_DBFS:
        return False
    if chunk.max_dBFS < MIN_PEAK_DBFS:
        return False
    return True

def _frame_rms_dbfs(seg: AudioSegment, frame_ms: int) -> np.ndarray:
    # フレームごとのRMS(dBFS相当)を計算
    sr = seg.frame_rate
    samples = np.array(seg.get_array_of_samples()).astype(np.float32)
    if seg.channels > 1:
        samples = samples.reshape((-1, seg.channels)).mean(axis=1)
    # [-1,1] 近似
    max_amp = float(seg.max_possible_amplitude)
    if max_amp > 0:
        samples /= max_amp

    hop = int(sr * frame_ms / 1000.0)
    hop = max(hop, 1)
    n = len(samples)
    rms = []
    for i in range(0, n, hop):
        x = samples[i:i+hop]
        if len(x) == 0:
            continue
        v = np.sqrt(np.mean(x * x) + 1e-12)
        rms.append(20.0 * np.log10(v + 1e-12))
    return np.array(rms, dtype=np.float32)

def tighten_bounds(audio: AudioSegment, s_ms: int, e_ms: int) -> tuple[int, int]:
    seg = audio[s_ms:e_ms]
    rms_db = _frame_rms_dbfs(seg, FRAME_MS)
    if len(rms_db) == 0:
        return s_ms, e_ms

    peak = float(np.max(rms_db))
    thr = peak - TRIM_REL_DB

    idx = np.where(rms_db >= thr)[0]
    if len(idx) == 0:
        return s_ms, e_ms

    hop = FRAME_MS
    s2 = s_ms + int(idx[0] * hop)
    e2 = s_ms + int((idx[-1] + 1) * hop)
    s2 = max(0, s2)
    e2 = min(len(audio), e2)
    if e2 <= s2:
        return s_ms, e_ms
    return s2, e2

def split_if_too_long(audio: AudioSegment, s_ms: int, e_ms: int) -> list[tuple[int, int]]:
    dur = e_ms - s_ms
    if dur <= MAX_CHUNK_MS:
        return [(s_ms, e_ms)]

    seg = audio[s_ms:e_ms]
    rms_db = _frame_rms_dbfs(seg, FRAME_MS)
    if len(rms_db) == 0:
        return [(s_ms, e_ms)]

    peak = float(np.max(rms_db))
    valley_thr = peak - VALLEY_DROP_DB

    # 谷候補（十分低いところ）
    cand = np.where(rms_db <= valley_thr)[0]
    if len(cand) == 0:
        return [(s_ms, e_ms)]

    # 分割点は中央付近の「一番低い谷」を優先
    hop = FRAME_MS
    mid = len(rms_db) // 2
    # ギャップ条件（前後に最低MIN_GAP_MS）
    min_gap_frames = max(int(MIN_GAP_MS / hop), 1)
    valid = [i for i in cand if (i > min_gap_frames) and (i < len(rms_db) - min_gap_frames)]
    if not valid:
        return [(s_ms, e_ms)]

    best = min(valid, key=lambda i: (abs(i - mid), rms_db[i]))
    cut_ms = s_ms + int(best * hop)

    # 2分割して、さらに長ければ再帰的に分割
    left = split_if_too_long(audio, s_ms, cut_ms)
    right = split_if_too_long(audio, cut_ms, e_ms)
    return left + right

def postprocess_ranges(audio: AudioSegment, ranges: list[list[int]]) -> list[tuple[int, int]]:
    out = []
    for s, e in ranges:
        s2, e2 = tighten_bounds(audio, s, e)
        parts = split_if_too_long(audio, s2, e2)
        for ps, pe in parts:
            # もう一度だけタイト化（分割後の端を詰める）
            ps2, pe2 = tighten_bounds(audio, ps, pe)
            out.append((ps2, pe2))

    # 目標10個に近づけたい場合：長い順に追加分割して数を増やす
    # （閾値は変えず、MAX_CHUNK_MS判定だけで分割）
    guard = 0
    while len(out) < TARGET_COUNT and guard < 50:
        guard += 1
        # 最長を再分割
        i = int(np.argmax([e - s for s, e in out]))
        s, e = out.pop(i)
        parts = split_if_too_long(audio, s, e)
        if len(parts) == 1:
            out.append((s, e))
            break
        out.extend(parts)

    # 時系列順に整列
    out.sort(key=lambda x: x[0])
    return out

def process_one(in_path: Path) -> None:
    if not in_path.exists():
        print(f"[SKIP] not found: {in_path}")
        return

    audio = AudioSegment.from_file(in_path)

    # 1) まずは固定閾値で検出（ここは現状のまま）
    ranges = detect_nonsilent(
        audio,
        min_silence_len=MIN_SILENCE_LEN_MS,
        silence_thresh=SILENCE_THRESH_DBFS,
    )  # [[start_ms, end_ms], ...] :contentReference[oaicite:3]{index=3}

    base_dir = in_path.parent / f"{in_path.stem}_chunks"
    voiced_dir = base_dir / "voiced"
    rejected_dir = base_dir / "rejected"

    voiced_dir.mkdir(parents=True, exist_ok=True)
    if SAVE_REJECTED:
        rejected_dir.mkdir(parents=True, exist_ok=True)

    if not ranges:
        print(f"[NG] no voiced segments: {in_path.name}")
        return
    
    voiced_idx = 0
    rejected_idx = 0

    # 2) 後処理（トリム＋長い区間の分割）
    refined = postprocess_ranges(audio, ranges)

    for i, (start_ms, end_ms) in enumerate(refined, start=1):
        s = max(0, start_ms - KEEP_SILENCE_MS)
        e = min(len(audio), end_ms + KEEP_SILENCE_MS)
        chunk = audio[s:e]
        if is_valid_chunk(chunk):
            voiced_idx += 1
            out_path = voiced_dir / f"chunkD_{voiced_idx:02d}.wav"
            chunk.export(out_path, format="wav")
        else:
            # 除外chunkは保存しない／保存する（選択式）
            if SAVE_REJECTED:
                rejected_idx += 1
                out_path = rejected_dir / f"chunkD_{rejected_idx:02d}.wav"
                chunk.export(out_path, format="wav")

    print(f"[OK] {in_path.name} -> detected={len(ranges)} voiced={voiced_idx} rejected={rejected_idx}")

def main():
    for f in INPUT_FILES:
        process_one(f)

if __name__ == "__main__":
    main()
