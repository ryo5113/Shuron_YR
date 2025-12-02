# 使い方: 同じフォルダに soundAnalysis.py / soundWhisper.py を置き、
# 実行してください。録画開始後、Enter で停止します。

import subprocess
import sys
import re
import shlex
from pathlib import Path
from datetime import datetime

# 出力動画ファイル名（soundAnalysis.py の TARGET_VIDEO='testA.mp4' に合わせる）
OUTPUT_VIDEO = "testv2.mp4"

# 録画条件
WIDTH, HEIGHT = 1280, 720        # 720p
FRAMERATE = 30
AUDIO_SR = 16000                 # 16kHz
AUDIO_CH = 1                     # mono

def pick_default_dshow_devices():
    """
    ffmpeg の dshow デバイス一覧を取得し、
    最初に見つかった Video / Audio を返す（名前文字列）。
    """
    # デバイス一覧（stderrに出る）
    proc = subprocess.run(
        ["ffmpeg", "-hide_banner", "-list_devices", "true", "-f", "dshow", "-i", "dummy"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding="utf-8", errors="ignore"
    )
    s = proc.stderr

    # 1行ずつ見て、直前のヘッダ行(=表示名行)にだけ Alternative name を結びつける
    head_re = re.compile(r'^\[dshow\s*@\s*[^\]]+\]\s*"([^"]+)"\s*\((video|audio|none)\)\s*$')
    alt_re  = re.compile(r'^\[dshow\s*@\s*[^\]]+\]\s*Alternative name\s*"([^"]+)"\s*$')

    video_display = video_alt = None
    audio_display = audio_alt = None

    pending_role = None  # "video" or "audio"（none は無視）

    for line in s.splitlines():
        m = head_re.match(line)
        if m:
            name, kind = m.group(1), m.group(2).lower()
            pending_role = kind if kind in ("video", "audio") else None
            if kind == "video" and video_display is None:
                video_display = name
            elif kind == "audio" and audio_display is None:
                audio_display = name
            continue

        a = alt_re.match(line)
        if a and pending_role:
            alt = a.group(1)
            if pending_role == "video" and video_alt is None:
                video_alt = alt
            elif pending_role == "audio" and audio_alt is None:
                audio_alt = alt
            pending_role = None  # この見出しへの結びつけは完了

    return (video_display, video_alt), (audio_display, audio_alt)

def resolve_dshow_id(kind: str, display: str | None, alt: str | None) -> str:
    """
    kind: "video" or "audio"
    Alternative name を優先し、開けなければ Display name を試す。
    成功した方の「値部分」（@device_... または "表示名"）を返す。
    """

    # dshow -i に渡す候補（値部分）
    candidates = []
    if alt:
        candidates.append(alt)                 # 例: @device_pnp_...
    if display:
        candidates.append(f'"{display}"')      # 例: "HD Webcam"

    for val in candidates:
        # -t 1 で短時間だけ開いて破棄（録画はしない）
        cmd = [
            "ffmpeg", "-hide_banner",
            "-f", "dshow", "-t", "1",
            "-i", f"{kind}={val}",
            "-f", "null", "-"
        ]
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding="utf-8", errors="ignore")
        # dshow は成功でも 0 以外の終了コードを返す場合があるため、文言でも補助判定
        if proc.returncode == 0 or "Press [q] to stop" in proc.stderr or "Immediate exit requested" in proc.stderr:
            return val

    raise RuntimeError(f"dshow {kind} デバイスが開けませんでした（display={display}, alt={alt}）")



def record_with_ffmpeg(out_path: Path, video_name: str, audio_name: str):
    """
    ffmpeg を起動して録画し、Enter 押下で 'q' を送って停止させる。
    stderr を逐次表示して、失敗理由を可視化する。
    """
    import threading, sys

    dshow_input = f'video={video_name}:audio={audio_name}'

    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        # DirectShow 入力
        "-f", "dshow",
        # 取りこぼし回避のためのバッファ
        "-rtbufsize", "512M",
        "-thread_queue_size", "512",
        # 映像条件
        "-video_size", f"{WIDTH}x{HEIGHT}",
        "-framerate", str(FRAMERATE),
        "-i", dshow_input,
        # エンコード設定
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-crf", "23",
        "-pix_fmt", "yuv420p",
        # 音声設定（16kHz, mono）
        "-c:a", "aac",
        "-ar", str(AUDIO_SR),
        "-ac", str(AUDIO_CH),
        # 出力
        str(out_path),
    ]

    print("▶ 録画を開始します。停止するには Enter を押してください。")
    print(f"  Video Device: {video_name}")
    print(f"  Audio Device: {audio_name}")
    print(f"  出力: {out_path.resolve()}")
    print("  実行コマンド:", " ".join(cmd))

    # 逐次ログ出力のため PIPE で受けて別スレッドで標準エラーを都度表示
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    def pump_stderr():
        for line in proc.stderr:
            # ffmpegは進行状況をstderrへ出す
            sys.stderr.write(line)
            sys.stderr.flush()

    t = threading.Thread(target=pump_stderr, daemon=True)
    t.start()

    try:
        input()  # Enter 待ち
        # 'q' + newline を送って優雅に停止
        try:
            proc.stdin.write('q\n')
            proc.stdin.flush()
        except Exception:
            pass
    finally:
        proc.wait(timeout=None)  # 終了待ち

    # 正常に終了していても、ゼロ長ファイルだと失敗扱いにする
    if not out_path.exists() or out_path.stat().st_size == 0:
        raise RuntimeError("録画ファイルが生成されていません。上の ffmpeg ログを確認してください。")


def run_analysis_and_whisper():
    """
    soundAnalysis.py → soundWhisper.py を順に実行。
    いずれも同じフォルダにある想定。
    """
    # 周波数解析＆デノイズ
    print("▶ soundAnalysis.py 実行中...")
    subprocess.run([sys.executable, "soundAnalysis.py"], check=True) #sys.executableはこのスクリプトが保存されているpython実行ファイルのパス（仮想環境対応）

    # Whisper 文字起こし
    print("▶ soundWhisper.py 実行中...")
    subprocess.run([sys.executable, "soundWhisper.py"], check=True)

def main():
    # 前提チェック
    for name in ["soundAnalysis.py", "soundWhisper.py"]:
        if not Path(name).exists():
            raise FileNotFoundError(f"{name} が見つかりません。スクリプトと同じフォルダに置いてください。")

    # デバイス自動選択
    (video_display, video_alt), (audio_display, audio_alt) = pick_default_dshow_devices()
    vname = resolve_dshow_id("video", video_display, video_alt)
    aname = resolve_dshow_id("audio", audio_display, audio_alt)
    if not vname or not aname:
        raise RuntimeError("DirectShow のデフォルト Video/Audio デバイスを自動取得できませんでした。"
                           " ffmpeg -list_devices true -f dshow -i dummy の出力をご確認ください。")

    # 出力ファイル名（soundAnalysis.py 既定）
    out_path = Path(OUTPUT_VIDEO)

    # 録画
    record_with_ffmpeg(out_path, vname, aname)

    # 後処理（解析→文字起こし）
    run_analysis_and_whisper()

    print("一連の処理が完了しました。")

if __name__ == "__main__":
    main()
