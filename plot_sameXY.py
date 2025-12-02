# plot_xy.py
import pandas as pd
import matplotlib.pyplot as plt

CSV_PATH = "session_20251115_180727/logs/lip_landmarks_norm.csv"
TARGET_IDS = [61, 13, 14, 291]   # 左口角/上内側/下内側/右口角

df = pd.read_csv(CSV_PATH)
sub = df[df["landmark_id"].isin(TARGET_IDS)].copy()

plt.figure()
for lid in TARGET_IDS:
    s = sub[sub["landmark_id"]==lid]
    plt.plot(s["timestamp_sec"], s["x_norm"], label=f"id{lid} x")
plt.title("x_norm over time"); plt.xlabel("time [s]"); plt.ylabel("x_norm [0-1]")
plt.legend(); plt.tight_layout(); plt.show()

plt.figure()
for lid in TARGET_IDS:
    s = sub[sub["landmark_id"]==lid]
    plt.plot(s["timestamp_sec"], s["y_norm"], label=f"id{lid} y")
plt.title("y_norm over time"); plt.xlabel("time [s]"); plt.ylabel("y_norm [0-1]")
plt.legend(); plt.tight_layout(); plt.show()
