import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# ==== 設定ここから =====================================

# 計測で保存された CSV ファイルのパス
CSV_PATH = r"firstmouth/lip_depth_points_3d.csv"  # 実際のパスに変更してください

# 基準（正面）の capture_id
REF_CAPTURE_ID = 0

# 比較したい capture_id のリスト
TARGET_CAPTURE_IDS = [1, 2, 3, 4]  # 必要に応じて変更

# 結果を書き出すテキストファイル
OUTPUT_TXT = os.path.join(os.path.dirname(CSV_PATH), "kabsch_results.txt")

# ==== 設定ここまで =====================================


def kabsch_rotation_translation(src: np.ndarray, tgt: np.ndarray):
    """
    Kabsch 法による剛体変換推定
    src: (N,3) 基準側の点群 (P)
    tgt: (N,3) 比較側の点群 (Q)
    tgt ≒ R @ src + t となる R(3x3), t(3,) を求める
    """
    # 重心
    src_mean = src.mean(axis=0)
    tgt_mean = tgt.mean(axis=0)

    # 中心化
    src_centered = src - src_mean
    tgt_centered = tgt - tgt_mean

    # 共分散行列
    H = src_centered.T @ tgt_centered  # (3,3)

    # SVD
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # 反転（鏡映）を防ぐ
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    # 平行移動 t
    t = tgt_mean - R @ src_mean

    return R, t


def rotation_angle_deg(R: np.ndarray) -> float:
    """
    回転行列 R から全体回転角（度）を計算
    θ = acos((trace(R) - 1) / 2)
    """
    tr = np.trace(R)
    # 数値誤差対策
    cos_theta = (tr - 1.0) / 2.0
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    theta = np.arccos(cos_theta)
    return np.degrees(theta)


def plot_overlay(P: np.ndarray, Q_aligned: np.ndarray, landmark_ids: np.ndarray, cid: int):
    """
    正面 P と、Kabsch で整列した Q_aligned を重ねてプロット
    2×2 サブプロット: XY, XZ, ZY, 3D
    """
    xP, yP, zP = P[:, 0], P[:, 1], P[:, 2]
    xQ, yQ, zQ = Q_aligned[:, 0], Q_aligned[:, 1], Q_aligned[:, 2]

    # スケールそろえ（両者をまとめた範囲）
    all_x = np.concatenate([xP, xQ])
    all_y = np.concatenate([yP, yQ])
    all_z = np.concatenate([zP, zQ])

    max_range = max(
        all_x.max() - all_x.min(),
        all_y.max() - all_y.min(),
        all_z.max() - all_z.min(),
    )
    if max_range == 0:
        max_range = 1.0

    x_mid = (all_x.max() + all_x.min()) / 2.0
    y_mid = (all_y.max() + all_y.min()) / 2.0
    z_mid = (all_z.max() + all_z.min()) / 2.0

    x_lim = (x_mid - max_range / 2, x_mid + max_range / 2)
    y_lim = (y_mid - max_range / 2, y_mid + max_range / 2)
    z_lim = (z_mid - max_range / 2, z_mid + max_range / 2)

    fig = plt.figure(figsize=(10, 10))
    plt.rcParams["font.size"] = 16
    fig.suptitle(f"capture_id={cid}: ref vs aligned (Kabsch)")

    # 1: XY
    ax_xy = fig.add_subplot(2, 2, 1)
    ax_xy.scatter(xP, yP, s=20, label="ref (capture 0)")
    ax_xy.scatter(xQ, yQ, s=20, marker="^", label=f"aligned capture {cid}")
    # for xi, yi, lid in zip(xP, yP, landmark_ids):
    #     ax_xy.text(xi, yi, str(lid), fontsize=8)
    # for xi, yi, lid in zip(xQ, yQ, landmark_ids):
    #     ax_xy.text(xi, yi, str(lid), fontsize=8)
    ax_xy.set_xlabel("X [mm]")
    ax_xy.set_ylabel("Y [mm]")
    ax_xy.set_title("XY view")
    ax_xy.set_xlim(x_lim)
    ax_xy.set_ylim(y_lim)
    ax_xy.set_aspect("equal", adjustable="box")
    ax_xy.legend(bbox_to_anchor=(0.5, -0.2), loc='upper right', fontsize=12)

    # 2: XZ
    ax_xz = fig.add_subplot(2, 2, 2)
    ax_xz.scatter(xP, zP, s=20, label="ref (capture 0)")
    ax_xz.scatter(xQ, zQ, s=20, marker="^", label=f"aligned capture {cid}")
    # for xi, zi, lid in zip(xP, zP, landmark_ids):
    #     ax_xz.text(xi, zi, str(lid), fontsize=8)
    # for xi, zi, lid in zip(xQ, zQ, landmark_ids):
    #     ax_xz.text(xi, zi, str(lid), fontsize=8)
    ax_xz.set_xlabel("X [mm]")
    ax_xz.set_ylabel("Z [mm]")
    ax_xz.set_title("XZ view")
    ax_xz.set_xlim(x_lim)
    ax_xz.set_ylim(z_lim)
    ax_xz.set_aspect("equal", adjustable="box")
    ax_xz.legend(bbox_to_anchor=(0.5, -0.2), loc='upper right', fontsize=12)

    # 3: ZY
    ax_zy = fig.add_subplot(2, 2, 3)
    ax_zy.scatter(zP, yP, s=20, label="ref (capture 0)")
    ax_zy.scatter(zQ, yQ, s=20, marker="^", label=f"aligned capture {cid}")
    # for zi, yi, lid in zip(zP, yP, landmark_ids):
    #     ax_zy.text(zi, yi, str(lid), fontsize=8)
    # for zi, yi, lid in zip(zQ, yQ, landmark_ids):
    #     ax_zy.text(zi, yi, str(lid), fontsize=8)
    ax_zy.set_xlabel("Z [mm]")
    ax_zy.set_ylabel("Y [mm]")
    ax_zy.set_title("ZY view")
    ax_zy.set_xlim(z_lim)
    ax_zy.set_ylim(y_lim)
    ax_zy.set_aspect("equal", adjustable="box")
    ax_zy.legend(bbox_to_anchor=(0.5, -0.2), loc='upper right', fontsize=12)

    # 4: 3D
    ax_3d = fig.add_subplot(2, 2, 4, projection="3d")
    ax_3d.scatter(xP, yP, zP, s=20, label="ref (capture 0)")
    ax_3d.scatter(xQ, yQ, zQ, s=20, marker="^", label=f"aligned capture {cid}")
    # for xi, yi, zi, lid in zip(xP, yP, zP, landmark_ids):
    #     ax_3d.text(xi, yi, zi, str(lid), fontsize=8)
    # for xi, yi, zi, lid in zip(xQ, yQ, zQ, landmark_ids):
    #     ax_3d.text(xi, yi, zi, str(lid), fontsize=8)
    ax_3d.set_xlabel("X [mm]")
    ax_3d.set_ylabel("Y [mm]")
    ax_3d.set_zlabel("Z [mm]")
    ax_3d.set_title("3D view")
    ax_3d.set_xlim(x_lim)
    ax_3d.set_ylim(y_lim)
    ax_3d.set_zlim(z_lim)
    ax_3d.legend(bbox_to_anchor=(0.5, -0.2), loc='upper right', fontsize=12)

    plt.tight_layout()
    plt.show()


def main():
    df = pd.read_csv(CSV_PATH)

    # txtファイルを新規作成（上書き）
    with open(OUTPUT_TXT, "w", encoding="utf-8") as f:

        def log(msg: str):
            f.write(msg + "\n")

        # 基準（正面）データ
        df_ref = df[df["capture_id"] == REF_CAPTURE_ID]
        if df_ref.empty:
            log(f"REF_CAPTURE_ID = {REF_CAPTURE_ID} のデータがありません。")
            return

        log(f"基準 capture_id = {REF_CAPTURE_ID} の点数: {len(df_ref)}")
        log("")

        for cid in TARGET_CAPTURE_IDS:
            df_tgt = df[df["capture_id"] == cid]
            if df_tgt.empty:
                log(f"capture_id = {cid} のデータがありません。スキップします。")
                log("")
                continue

            # landmark_id で対応付け
            merged = pd.merge(
                df_ref,
                df_tgt,
                on="landmark_id",
                suffixes=("_ref", "_tgt"),
            )

            if merged.empty:
                log(f"capture_id = {cid}: 共通の landmark_id がありません。スキップします。")
                log("")
                continue

            # 基準側 P（src）と比較側 Q（tgt）の3D座標（mm）
            P = merged[["x_cam_mm_ref", "y_cam_mm_ref", "z_cam_mm_ref"]].to_numpy()
            Q = merged[["x_cam_mm_tgt", "y_cam_mm_tgt", "z_cam_mm_tgt"]].to_numpy()

            # Kabsch 法で P → Q の R, t を推定
            R, t = kabsch_rotation_translation(P, Q)

            # 回転角
            angle_deg = rotation_angle_deg(R)

            # 平行移動ノルム（mm）
            trans_norm = np.linalg.norm(t)

            # 整列後の誤差（RMS）: Q_est = R @ P + t
            Q_est = (R @ P.T).T + t  # (N,3)
            diff = Q - Q_est
            rms = np.sqrt((diff**2).sum(axis=1).mean())

            log(f"=== capture_id = {cid} ===")
            log(f"  共通ランドマーク数: {P.shape[0]}")
            log(f"  回転角: {angle_deg:.2f} [deg]")
            log(f"  平行移動量の大きさ: {trans_norm:.2f} [mm]")
            log(f"  整列後のRMS誤差: {rms:.2f} [mm]")
            log("  回転行列 R:")
            log(f"   {R[0]}")
            log(f"   {R[1]}")
            log(f"   {R[2]}")
            log(f"  平行移動ベクトル t: {t}")

            # ===== ここから追加: Kabsch で整列した座標の計算 =====
            # Q = R P + t なので、Q を P 側に戻すには Q_aligned = R^T (Q - t)
            Q_aligned = (R.T @ (Q - t).T).T  # (N,3), P と同じ座標系に整列

            # 正面 P と整列後 Q_aligned の各軸の幅を比較
            for axis, idx in zip(["X", "Y", "Z"], [0, 1, 2]):
                ref_vals = P[:, idx]
                ali_vals = Q_aligned[:, idx]

                ref_min, ref_max = ref_vals.min(), ref_vals.max()
                ali_min, ali_max = ali_vals.min(), ali_vals.max()

                ref_width = ref_max - ref_min
                ali_width = ali_max - ali_min

                log(f"  {axis}軸 幅比較 (基準 vs 整列後 capture {cid}):")
                log(f"    ref width = {ref_width:.2f} [mm] "
                    f"(min={ref_min:.2f}, max={ref_max:.2f})")
                log(f"    aligned width = {ali_width:.2f} [mm] "
                    f"(min={ali_min:.2f}, max={ali_max:.2f})")

            log("")

            # ===== ここから追加: プロット（重ね合わせ） =====
            # 必要に応じてコメントアウトしてください
            plot_overlay(P, Q_aligned, merged["landmark_id"].to_numpy(), cid)

    print(f"結果を {OUTPUT_TXT} に保存しました。")


if __name__ == "__main__":
    main()
