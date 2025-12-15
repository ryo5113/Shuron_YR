import os
import csv
import torch
import numpy as np
import torchaudio
import torchaudio.functional as F
from transformers import AutoProcessor, Wav2Vec2ForCTC

# =========================
# 設定
# =========================
AUDIO_PATH = "takana.wav"          # 入力WAV
TRANSCRIPT_HIRA = "たかな"         # ひらがな（Whisper結果）
OUT_CSV = "mora_timestamps_t.csv"    # 出力CSV
OUT_DIR = "mora_segments_takana" 

MODEL_NAME = "reazon-research/japanese-wav2vec2-base-rs35kh"
MODEL_SR = 16000

# forced_alignの前後に入れるパディング（秒）
PAD_SEC = 0.5

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_audio_mono_16k(path: str, target_sr: int = 16000):
    wav, sr = torchaudio.load(path)  # [ch, time]
    if wav.size(0) > 1:
        wav = torch.mean(wav, dim=0, keepdim=True)  # stereo -> mono
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)
        sr = target_sr
    return wav, sr


def build_char_targets(tokenizer, text: str) -> torch.Tensor:
    """
    文字単位で token id を作る（「かな」が1トークンになる問題を回避）
    戻り: targets [B, L]
    """
    ids = []
    for ch in text:
        enc = tokenizer(ch, add_special_tokens=False)
        ids.extend(enc["input_ids"])
    return torch.tensor([ids], dtype=torch.long)


def path_to_spans_by_ids(path_1d: torch.Tensor, token_ids: list[int], blank_id: int):
    """
    path_1d: [T]  forced_alignが返す「各フレームのラベルID」
    token_ids: 例 [43, 14, 13]（さ/か/な）
    blank_id: CTC blank

    戻り: [(start_frame, end_frame)] を len(token_ids) 個（endは含まない）
    - 「次のトークンの開始フレーム」までを、そのトークンの区間とする（途中のblankも含める）。
    """
    T = int(path_1d.numel())
    starts = []
    prev = 0

    # まず各トークンの「最初に出たフレーム」を見つける
    for tid in token_ids:
        idx_all = (path_1d == tid).nonzero(as_tuple=True)[0]
        idx = idx_all[idx_all >= prev]
        if idx.numel() == 0:
            starts.append(None)
        else:
            s = int(idx.min().item())
            starts.append(s)
            prev = s + 1

    # start が取れないトークンがあるなら、以降の区間化ができないので None を返す
    if any(s is None for s in starts):
        return [(None, None) for _ in token_ids]

    # 次トークン開始までを end にする（最後はTまで）
    spans = []
    for i in range(len(token_ids)):
        s = starts[i]
        e = starts[i + 1] if i + 1 < len(token_ids) else T
        spans.append((s, e))
    return spans

def frames_to_seconds(spans_frames, num_audio_samples: int, num_emission_frames: int, sr: int, pad_sec: float):
    """
    フレーム(start,end) -> 秒(start,end)
    pad_sec（前後に足した分）を差し引き、0未満は0に丸める
    """
    ratio = num_audio_samples / num_emission_frames / sr
    out = []
    for sf, ef in spans_frames:
        if sf is None:
            out.append((None, None))
            continue
        s = sf * ratio - pad_sec
        e = ef * ratio - pad_sec
        out.append((max(0.0, float(s)), max(0.0, float(e))))
    return out


def main():
    if not os.path.isfile(AUDIO_PATH):
        raise FileNotFoundError(AUDIO_PATH)

    # モデル
    processor = AutoProcessor.from_pretrained(MODEL_NAME)
    model = Wav2Vec2ForCTC.from_pretrained(MODEL_NAME).to(DEVICE).eval()

    # ★あなたの環境では forced_align が blank_id 引数を受け取れないため、
    #   blank を 0 として扱う前提に揃える（ここを先に実行）
    processor.tokenizer.pad_token_id = 0
    model.config.pad_token_id = 0
    blank_id = 0

    # 音声読み込み（mono/16k）
    wav, sr = load_audio_mono_16k(AUDIO_PATH, MODEL_SR)  # wav: [1, N]

    # PAD_SEC だけ前後にゼロパディングしてからモデルに入れる
    pad = int(PAD_SEC * sr)
    wav_padded = torch.nn.functional.pad(wav, (pad, pad))  # [1, N + 2*pad]

    # emission（log_probs）を作る
    audio_np = wav_padded.squeeze(0).cpu().numpy()
    inputs = processor(audio_np, sampling_rate=sr, return_tensors="pt")
    input_values = inputs.input_values.to(DEVICE)

    with torch.inference_mode():
        logits = model(input_values).logits  # [B, T, V]
    log_probs = torch.log_softmax(logits, dim=-1).cpu()    # [B, T, V]

    # targets を文字単位で作る（さ/か/な）
    targets = build_char_targets(processor.tokenizer, TRANSCRIPT_HIRA).cpu()  # [B, L]
    token_ids = targets[0].tolist()
    L = targets.size(1)

    print("token_ids:", token_ids, "L=", L)
    print("blank_id:", blank_id)

    # forced_align：返り値は tuple(Tensor path, Tensor scores) :contentReference[oaicite:2]{index=2}
    path, scores = F.forced_align(log_probs, targets)
    print("path type:", type(path))
    print("path shape:", path.shape)

    # path: [B, T] から tokenごとの(start,end)フレームを作る
    path_1d = path[0]  # [T]
    
    spans_frames = path_to_spans_by_ids(path_1d, token_ids, blank_id)

    # 秒へ変換（※wav_paddedを入れてるのでサンプル数はwav_padded基準、最後にPAD_SECを引く）
    T = log_probs.size(1)  # emission frames
    spans_sec = frames_to_seconds(
        spans_frames,
        num_audio_samples=wav_padded.size(1),
        num_emission_frames=T,
        sr=sr,
        pad_sec=PAD_SEC
    )

    # 未割り当てがある場合は詳細を出して停止
    if any(s is None or e is None for (s, e) in spans_sec):
        raise RuntimeError(f"一部トークンにアライメントが割り当てられていません: {spans_sec}")

    # CSV出力（モーラ=1文字前提）
    moras = list(TRANSCRIPT_HIRA)
    if len(moras) != L:
        # 今回は「さかな」なので通常ここには来ないはず
        raise RuntimeError(f"モーラ数({len(moras)})とtargets長({L})が一致しません。")

    with open(OUT_CSV, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["mora", "token_id", "start_sec", "end_sec"])
        for mora, tid, (s, e) in zip(moras, token_ids, spans_sec):
            w.writerow([mora, tid, s, e])

    # ===== 各モーラ区間で音声を切り出して保存 =====
    os.makedirs(OUT_DIR, exist_ok=True)

    for i, (mora, (s, e)) in enumerate(zip(moras, spans_sec), start=1):
        # 秒→サンプル index（wav は PAD していない元の音声）
        start_sample = int(round(s * sr))
        end_sample   = int(round(e * sr))

        # 範囲チェック（念のため）
        start_sample = max(0, min(start_sample, wav.size(1)))
        end_sample   = max(0, min(end_sample, wav.size(1)))
        if end_sample <= start_sample:
            print(f"[warn] skip {mora}: invalid range {s:.3f}-{e:.3f}")
            continue

        seg = wav[:, start_sample:end_sample]  # [1, segment_len]

        out_path = os.path.join(OUT_DIR, f"{i:02d}_{mora}_{s:.3f}-{e:.3f}.wav")
        torchaudio.save(out_path, seg.cpu(), sr)
        print(f"[i] saved: {out_path}")

    print(f"OK: wrote {OUT_CSV}")
    for mora, tid, (s, e) in zip(moras, token_ids, spans_sec):
        print(f"{mora} (id={tid}) : {s:.3f} - {e:.3f} sec")


if __name__ == "__main__":
    main()
