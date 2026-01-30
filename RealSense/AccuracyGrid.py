import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# (label, csv_path)
CSV_LIST = [
    ("Subject1", r"./grid_sweep_results_csv/dens/NN/grid_sweep_accuracy.csv"),
    ("Subject2", r"./grid_sweep_results_csv/dens/NT/grid_sweep_accuracy.csv"),
    ("Subject3", r"./grid_sweep_results_csv/dens/KH/grid_sweep_accuracy.csv"),
    ("Subject4", r"./grid_sweep_results_csv/dens/SR/grid_sweep_accuracy.csv"),
    ("Subject5", r"./grid_sweep_results_csv/dens/YR/grid_sweep_accuracy.csv"),
    ("All",      r"./grid_sweep_results_csv/dens/ALL/grid_sweep_accuracy.csv"),
]

OUT_DIR = Path("./plots_from_multi_csv")
OUT_DIR.mkdir(parents=True, exist_ok=True)

FIG_W, FIG_H = 12, 7
DPI = 200
AS_PERCENT = False  # Trueなら 0.923 -> 92.3 [%]


def read_and_prepare(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    required = {"grid", "test_accuracy"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {csv_path}: {missing}\nColumns: {list(df.columns)}")

    df = df.copy()
    df["grid"] = pd.to_numeric(df["grid"])
    df["test_accuracy"] = pd.to_numeric(df["test_accuracy"])
    df = df.sort_values("grid")
    return df


def save_mean_except_all(mean_df: pd.DataFrame, out_csv: Path):
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    mean_df.to_csv(out_csv, index=False, encoding="utf-8")
    print(f"[mean] saved: {out_csv.resolve()}")


def print_mean_except_all(mean_df: pd.DataFrame):
    print("\n=== Mean test_accuracy per grid (except All) ===")
    # 見やすい表示
    for _, row in mean_df.iterrows():
        g = int(row["grid"])
        acc = float(row["mean_test_accuracy"])
        if AS_PERCENT:
            print(f"grid={g:>3}  mean_test_accuracy={acc:.2f} [%]")
        else:
            print(f"grid={g:>3}  mean_test_accuracy={acc:.4f}")


def compute_mean_except_all(non_all_frames: list[pd.DataFrame]) -> pd.DataFrame:
    df_cat = pd.concat(non_all_frames, ignore_index=True)
    mean_df = (
        df_cat.groupby("grid", as_index=False)["test_accuracy"]
        .mean()
        .rename(columns={"test_accuracy": "mean_test_accuracy"})
        .sort_values("grid")
        .reset_index(drop=True)
    )
    if AS_PERCENT:
        mean_df["mean_test_accuracy"] = mean_df["mean_test_accuracy"] * 100.0
    return mean_df


def plot_mode(mode_name: str, out_png: Path):
    plt.figure(figsize=(FIG_W, FIG_H))
    plt.rcParams["font.size"] = 30

    any_plotted = False
    non_all_frames = []

    for label, csv_path in CSV_LIST:
        path = Path(csv_path.replace("￥", "\\"))
        if not path.exists():
            print(f"[WARN] not found: {label} -> {path}")
            continue

        df = read_and_prepare(path)

        # feature_mode 列がある場合は mode で絞る
        if "feature_mode" in df.columns:
            dff = df[df["feature_mode"].astype(str) == mode_name]
        else:
            dff = df

        if len(dff) == 0:
            print(f"[WARN] no data for mode={mode_name}: {label} -> {path}")
            continue

        y = dff["test_accuracy"].to_numpy()
        if AS_PERCENT:
            y = y * 100.0

        # グラフは各CSV（Subject/All）の線のみ
        plt.plot(dff["grid"], y, marker="o", linewidth=2, label=label)
        any_plotted = True

        # 平均用データ収集（Allを除外：大文字小文字どちらでも除外できるように）
        if label.lower() != "all":
            non_all_frames.append(dff[["grid", "test_accuracy"]].copy())

    # 平均は「描かない」：print と CSV保存だけ
    if len(non_all_frames) > 0:
        mean_df = compute_mean_except_all(non_all_frames)
        print_mean_except_all(mean_df)

        out_mean_csv = OUT_DIR / f"mean_except_all_{mode_name}.csv"
        save_mean_except_all(mean_df, out_mean_csv)
    else:
        print("[WARN] no non-All data -> mean cannot be computed")

    ylab = "test_accuracy [%]" if AS_PERCENT else "test_accuracy"
    plt.xlabel("grid", fontsize=30)
    plt.ylabel(ylab, fontsize=30)
    plt.title(f"Grid vs Test accuracy ({mode_name})", fontsize=30)
    plt.grid(True)
    plt.legend(bbox_to_anchor=(1, 1), loc="upper left", fontsize=18)
    plt.tight_layout()

    if any_plotted:
        plt.savefig(out_png, dpi=DPI)
        print("saved:", out_png.resolve())
    else:
        print(f"[WARN] nothing plotted -> skip saving: {out_png}")

    plt.close()


def detect_modes() -> list[str]:
    modes = set()

    for _, csv_path in CSV_LIST:
        path = Path(csv_path.replace("￥", "\\"))
        if not path.exists():
            continue

        df = pd.read_csv(path)
        if "feature_mode" not in df.columns:
            continue

        for m in df["feature_mode"].astype(str).unique():
            modes.add(m)

    if not modes:
        return ["all"]
    return sorted(modes)


def main():
    modes = detect_modes()

    if modes == ["all"]:
        out_png = OUT_DIR / "grid_test_accuracy_compare.png"
        plot_mode("all", out_png)
    else:
        for m in modes:
            out_png = OUT_DIR / f"grid_test_accuracy_compare_{m}.png"
            plot_mode(m, out_png)


if __name__ == "__main__":
    main()
