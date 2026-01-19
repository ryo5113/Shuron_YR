import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ===== ここだけ手で編集（ファイル名をリストで直書き） =====
CSV_LIST = [
    r"C:\Users\PC_User\Documents\GitHub\Shuron_YR\Sound\word_Ex1\10times_Ex1_A\svm_wav_dataset_A\trained_A_svm_model_band\sum73\band_sweep_cv_results.csv",
    r"C:\Users\PC_User\Documents\GitHub\Shuron_YR\Sound\word_Ex1\10times_Ex1_B\svm_wav_dataset_B\trained_B_svm_model_band\band_sweep_cv_results.csv",
    r"C:\Users\PC_User\Documents\GitHub\Shuron_YR\Sound\word_Ex1\10times_Ex1_C\svm_wav_dataset_C\trained_C_svm_model_band\band_sweep_cv_results.csv",
    r"C:\Users\PC_User\Documents\GitHub\Shuron_YR\Sound\word_Ex1\10times_Ex1_D\svm_wav_dataset_D\trained_D_svm_model_band\band_sweep_cv_results.csv",
    r"C:\Users\PC_User\Documents\GitHub\Shuron_YR\Sound\word\trained_Y_svm_model_band\band_sweep_cv_results.csv",
    r"C:\Users\PC_User\Documents\GitHub\Shuron_YR\Sound\word_Ex1\trained_all_svm_model_band\sum73\band_sweep_cv_results.csv",
]

# 最後だけ "all"、それ以外は A,B,C,... を自動付与
labels = []
for i in range(len(CSV_LIST)):
    if i == len(CSV_LIST) - 1:
        labels.append("all")
    else:
        labels.append(chr(ord("A") + i))

# ===== 描画 =====
plt.figure()

for label, csv_path in zip(labels, CSV_LIST):
    csv_path = csv_path.replace("￥", "\\")  # 念のため全角￥を吸収
    path = Path(csv_path)
    if not path.exists():
        print(f"[WARN] not found: {label} -> {path}")
        continue

def plot_range(title: str, out_png: str, lo: float, hi: float | None):
    """
    lo <= band_hz <= hi でプロット（hi=Noneなら band_hz > lo をプロット）
    """
    plt.figure(figsize=(20,6))
    plt.rcParams["font.size"] = 25
    any_plotted = False

    for label, csv_path in zip(labels, CSV_LIST):
        csv_path = csv_path.replace("￥", "\\")  # 念のため全角￥を吸収
        path = Path(csv_path)
        if not path.exists():
            print(f"[WARN] not found: {label} -> {path}")
            continue

        df = pd.read_csv(path).sort_values("band_hz")

        if hi is None:
            dff = df[df["band_hz"] > lo]
        else:
            dff = df[(df["band_hz"] >= lo) & (df["band_hz"] <= hi)]

        if len(dff) == 0:
            print(f"[WARN] no data in range for {label}: {lo}..{hi}")
            continue

        plt.plot(dff["band_hz"], dff["test_accuracy"] * 100.0, marker="o", label=label)
        any_plotted = True

    plt.xlabel("band_hz [Hz]")
    plt.ylabel("test_accuracy [%]")
    plt.title(title)
    plt.grid(True)
    #plt.legend()
    plt.tight_layout()

    if any_plotted:
        out = Path(out_png)
        plt.savefig(out, dpi=200)
        print("saved:", out.resolve())
    else:
        print(f"[WARN] nothing plotted -> skip saving: {out_png}")

    plt.close()

# 1) 0〜100Hz
plot_range(
    title="Test accuracy vs band_hz (0-100Hz)",
    out_png="band_test_accuracy_0_100.png",
    lo=0.0,
    hi=100.0,
)

# 2) 100Hzより大きい
plot_range(
    title="Test accuracy vs band_hz (>100Hz)",
    out_png="band_test_accuracy_over_100.png",
    lo=100.0,
    hi=None,
)

# ===== 描画 =====
plt.figure(figsize=(20,6))
plt.rcParams["font.size"] = 25

for label, csv_path in zip(labels, CSV_LIST):
    csv_path = csv_path.replace("￥", "\\")  # 念のため全角￥を吸収
    path = Path(csv_path)
    if not path.exists():
        print(f"[WARN] not found: {label} -> {path}")
        continue

    df = pd.read_csv(path)

    # band_hz 順に並べる
    df = df.sort_values("band_hz")

    # test_accuracy を%表示
    plt.plot(df["band_hz"], df["test_accuracy"] * 100.0, marker="o", label=label)

plt.xlabel("band_hz [Hz]")
plt.ylabel("test_accuracy [%]")
plt.title("Test accuracy vs band_hz")
plt.grid(True)
plt.legend()
plt.tight_layout()

out_png = Path("band_test_accuracy_compare.png")
plt.savefig(out_png, dpi=200)
plt.close()

print("saved:", out_png.resolve())