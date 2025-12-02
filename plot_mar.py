# plot_mar.py
import pandas as pd
import matplotlib.pyplot as plt

CSV_PATH = "lip_landmarks_norm.csv"

df = pd.read_csv(CSV_PATH)

# 同一フレームで重複するので、1フレーム1行に集約（MAR, mouth_open の代表値）
agg = (df.groupby(["timestamp_sec","frame_idx"], as_index=False)
         .agg(MAR=("MAR","mean"), mouth_open=("mouth_open","max")))

plt.figure()
plt.plot(agg["timestamp_sec"], agg["MAR"], label="MAR")
# 開口時に背景色を薄く
open_mask = agg["mouth_open"]==1
plt.fill_between(agg["timestamp_sec"], 0, agg["MAR"].max(), where=open_mask,
                 alpha=0.15, step="pre", label="OPEN")
plt.xlabel("time [s]"); plt.ylabel("MAR")
plt.legend(); plt.tight_layout()
plt.show()
