import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ==== 設定 ====
CSV_PATH = "session_20251117_152802/logs/lip_landmarks_norm.csv"  # ここは環境に合わせて変更可
TIMER_INTERVAL_MS = 100  # 自動再生時のフレーム間隔（ミリ秒）

# ==== データ読み込み ====
df = pd.read_csv(CSV_PATH)

# 利用可能なフレーム一覧
frame_list = sorted(df["frame_idx"].unique())
if len(frame_list) == 0:
    raise ValueError("frame_idx のデータがありません。")

state = {
    "frame_idx_pos": 0,   # frame_list の中で何番目か（0〜len-1）
    "auto_play": False    # 自動再生中かどうか
}

# ==== プロット準備 ====
fig, ax = plt.subplots()
scat = ax.scatter([], [], s=15)  # とりあえず空で作っておく

ax.set_aspect("equal", adjustable="box")
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_xlabel("x_norm")
ax.set_ylabel("1 - y_norm")

info_text = ax.text(
    0.02, 0.95, "",
    transform=ax.transAxes,
    va="top", ha="left"
)

def update_plot():
    """現在の state["frame_idx_pos"] に対応するフレームを再描画"""
    idx_pos = state["frame_idx_pos"]
    frame_val = frame_list[idx_pos]

    f = df[df["frame_idx"] == frame_val]

    # x, y 座標をセット（y は上下反転）
    xy = np.column_stack([f["x_norm"].values, 1.0 - f["y_norm"].values])
    scat.set_offsets(xy)

    ax.set_title(f"Lips (normalized) @ frame_idx={frame_val}")
    info_text.set_text(
        f"frame list index: {idx_pos+1}/{len(frame_list)}\n"
        f"auto_play: {state['auto_play']}"
    )

    fig.canvas.draw_idle()

def next_frame(step=1):
    """フレームを step 分だけ進める（負の値で戻す）"""
    state["frame_idx_pos"] = (state["frame_idx_pos"] + step) % len(frame_list)
    update_plot()

def on_key(event):
    """キー操作による制御"""
    key = event.key
    if key == "right":          # →キーで1フレーム進む
        state["auto_play"] = False
        next_frame(+1)
    elif key == "left":         # ←キーで1フレーム戻る
        state["auto_play"] = False
        next_frame(-1)
    elif key == " ":            # スペースキーで自動再生 ON/OFF
        state["auto_play"] = not state["auto_play"]
        update_plot()

fig.canvas.mpl_connect("key_press_event", on_key)

# ==== タイマーで自動再生 ====
timer = fig.canvas.new_timer(interval=TIMER_INTERVAL_MS)

def timer_callback():
    if state["auto_play"]:
        next_frame(+1)

timer.add_callback(timer_callback)
timer.start()

# 初期フレームを表示
update_plot()
plt.tight_layout()
plt.show()
