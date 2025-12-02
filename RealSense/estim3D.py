import os
import pandas as pd
import numpy as np

# ==== 設定ここから =====================================

CSV_PATH = r"firstmouth/lip_depth_points_3d.csv"  # あなたのCSVパスに変更
REF_CAPTURE_ID = 0  # 基準（正面）の capture_id

# 比較したい capture_id のリスト
TARGET_CAPTURE_IDS = [1, 2, 3, 4]  # 必要に応じて変更

# 出力するテキストファイル
OUTPUT_TXT = os.path.join(os.path.dirname(CSV_PATH), "lip_3d_stats.txt")

# ==== 設定ここまで =====================================


def main():
    df = pd.read_csv(CSV_PATH)

    # txtファイルを新規作成（上書き）
    with open(OUTPUT_TXT, "w", encoding="utf-8") as f:

        def log(msg: str):
            f.write(msg + "\n")

        # 基準データ
        df_ref = df[df["capture_id"] == REF_CAPTURE_ID]
        if df_ref.empty:
            log(f"REF_CAPTURE_ID = {REF_CAPTURE_ID} のデータがありません。")
            return

        log(f"=== 基準 capture_id = {REF_CAPTURE_ID} ===")
        log(f"  点数: {len(df_ref)}")

        # 基準（正面）の X/Y/Z 幅と max/min ランドマーク
        x_ref = df_ref["x_cam_mm"]
        y_ref = df_ref["y_cam_mm"]
        z_ref = df_ref["z_cam_mm"]

        # X
        x_max_ref = x_ref.max()
        x_min_ref = x_ref.min()
        x_width_ref = x_max_ref - x_min_ref
        idx_x_max_ref = x_ref.idxmax()
        idx_x_min_ref = x_ref.idxmin()
        lm_x_max_ref = df_ref.loc[idx_x_max_ref, "landmark_id"]
        lm_x_min_ref = df_ref.loc[idx_x_min_ref, "landmark_id"]

        # Y
        y_max_ref = y_ref.max()
        y_min_ref = y_ref.min()
        y_width_ref = y_max_ref - y_min_ref
        idx_y_max_ref = y_ref.idxmax()
        idx_y_min_ref = y_ref.idxmin()
        lm_y_max_ref = df_ref.loc[idx_y_max_ref, "landmark_id"]
        lm_y_min_ref = df_ref.loc[idx_y_min_ref, "landmark_id"]

        # Z
        z_max_ref = z_ref.max()
        z_min_ref = z_ref.min()
        z_width_ref = z_max_ref - z_min_ref
        idx_z_max_ref = z_ref.idxmax()
        idx_z_min_ref = z_ref.idxmin()
        lm_z_max_ref = df_ref.loc[idx_z_max_ref, "landmark_id"]
        lm_z_min_ref = df_ref.loc[idx_z_min_ref, "landmark_id"]

        log("  --- 基準キャプチャの3D座標の幅 ---")
        log(f"    X 幅: {x_width_ref:.2f} [mm] "
            f"(min={x_min_ref:.2f} @ landmark {lm_x_min_ref}, "
            f"max={x_max_ref:.2f} @ landmark {lm_x_max_ref})")
        log(f"    Y 幅: {y_width_ref:.2f} [mm] "
            f"(min={y_min_ref:.2f} @ landmark {lm_y_min_ref}, "
            f"max={y_max_ref:.2f} @ landmark {lm_y_max_ref})")
        log(f"    Z 幅: {z_width_ref:.2f} [mm] "
            f"(min={z_min_ref:.2f} @ landmark {lm_z_min_ref}, "
            f"max={z_max_ref:.2f} @ landmark {lm_z_max_ref})")
        log("")

        # 各ターゲット capture_id について
        for cid in TARGET_CAPTURE_IDS:
            df_tgt = df[df["capture_id"] == cid]
            if df_tgt.empty:
                log(f"capture_id = {cid} のデータがありません。スキップします。")
                log("")
                continue

            # 同じ landmark_id だけを使って対応を取る
            merged = pd.merge(df_ref, df_tgt,
                              on="landmark_id",
                              suffixes=("_ref", "_tgt"))

            if merged.empty:
                log(f"capture_id = {cid}: 共通の landmark_id がありません。スキップします。")
                log("")
                continue

            # 3D座標差分（mm）
            dx = merged["x_cam_mm_tgt"] - merged["x_cam_mm_ref"]
            dy = merged["y_cam_mm_tgt"] - merged["y_cam_mm_ref"]
            dz = merged["z_cam_mm_tgt"] - merged["z_cam_mm_ref"]

            # 各ランドマークごとの移動距離 d_i（mm）
            disp = np.sqrt(dx**2 + dy**2 + dz**2)

            mean_disp = disp.mean()          # 移動距離の平均
            std_disp = disp.std()            # 移動距離のばらつき（任意）

            # 平均変位ベクトル（方向を見るため）
            mean_dx = dx.mean()
            mean_dy = dy.mean()
            mean_dz = dz.mean()

            # XZ平面での角度（左右方向の変化イメージ）
            yaw_rad = np.arctan2(mean_dx, mean_dz)
            yaw_deg = np.degrees(yaw_rad)

            # YZ平面での角度（上下方向の変化イメージ）
            pitch_rad = np.arctan2(mean_dy, mean_dz)
            pitch_deg = np.degrees(pitch_rad)

            log(f"=== capture_id = {cid} ===")
            log(f"  共通ランドマーク数: {len(merged)}")
            log(f"  各ランドマーク移動距離の平均: {mean_disp:.2f} [mm]")
            log(f"  各ランドマーク移動距離の標準偏差: {std_disp:.2f} [mm]")
            log(f"  平均変位ベクトル: "
                f"dx={mean_dx:.2f} mm, dy={mean_dy:.2f} mm, dz={mean_dz:.2f} mm")
            log(f"  XZ平面での平均方向 (yaw 的): {yaw_deg:.2f} [deg]")
            log(f"  YZ平面での平均方向 (pitch 的): {pitch_deg:.2f} [deg]")

            # ===== 各姿勢での X/Y/Z 幅とその最大・最小ランドマーク（target側座標） =====
            x_tgt = merged["x_cam_mm_tgt"]
            y_tgt = merged["y_cam_mm_tgt"]
            z_tgt = merged["z_cam_mm_tgt"]

            # X
            x_max = x_tgt.max()
            x_min = x_tgt.min()
            x_width = x_max - x_min
            idx_x_max = x_tgt.idxmax()
            idx_x_min = x_tgt.idxmin()
            lm_x_max = merged.loc[idx_x_max, "landmark_id"]
            lm_x_min = merged.loc[idx_x_min, "landmark_id"]

            # Y
            y_max = y_tgt.max()
            y_min = y_tgt.min()
            y_width = y_max - y_min
            idx_y_max = y_tgt.idxmax()
            idx_y_min = y_tgt.idxmin()
            lm_y_max = merged.loc[idx_y_max, "landmark_id"]
            lm_y_min = merged.loc[idx_y_min, "landmark_id"]

            # Z
            z_max = z_tgt.max()
            z_min = z_tgt.min()
            z_width = z_max - z_min
            idx_z_max = z_tgt.idxmax()
            idx_z_min = z_tgt.idxmin()
            lm_z_max = merged.loc[idx_z_max, "landmark_id"]
            lm_z_min = merged.loc[idx_z_min, "landmark_id"]

            log("  --- 3D座標の幅（target側座標） ---")
            log(f"    X 幅: {x_width:.2f} [mm] "
                f"(min={x_min:.2f} @ landmark {lm_x_min}, "
                f"max={x_max:.2f} @ landmark {lm_x_max})")
            log(f"    Y 幅: {y_width:.2f} [mm] "
                f"(min={y_min:.2f} @ landmark {lm_y_min}, "
                f"max={y_max:.2f} @ landmark {lm_y_max})")
            log(f"    Z 幅: {z_width:.2f} [mm] "
                f"(min={z_min:.2f} @ landmark {lm_z_min}, "
                f"max={z_max:.2f} @ landmark {lm_z_max})")
            log("")

    # 関数外で一度だけ終了メッセージを出すならここ
    print(f"結果を {OUTPUT_TXT} に保存しました。")


if __name__ == "__main__":
    main()
