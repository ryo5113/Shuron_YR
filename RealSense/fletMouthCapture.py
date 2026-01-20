# fletMouthCapture.py
# 要件:
#  - 文字列入力で親フォルダ作成（fletSound.py同様）
#  - Start/Stopボタンで「撮影～口形状保存」プロトコルを開始/終了
#  - 撮影タイミングは OpenCVウィンドウ上で 'c' キー（3Cam_ARrot_MP_ML4.py同様）
#  - ARマーカー（AprilTag）検出が必須（見えていない場合は c を押しても撮影しない）
#  - 保存先は親フォルダ直下に:
#      raw_ply/  : 生のPLY（各カメラraw、統合plyなど）
#      mouth_ply/: 口切り出し後のPLY（機械学習用）
#      mediapipe_img/: MediaPipe描画画像（口元検出画像）
#
# ※ 3Cam_ARrot_MP_ML4.py の処理を「importして流用」し、保存パスだけ親フォルダ配下に変更します。

from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime
import traceback
import base64

import flet as ft
import cv2
import numpy as np
import pyrealsense2 as rs
import open3d as o3d

# 元スクリプトをimport（mainは __name__ == "__main__" のときだけ動くのでimportしても実行されません）
import captureMouth as core

LABELS = ["A", "I", "U", "E", "O"]# 口形状ラベル一覧

# -------------------------
# GUI 状態
# -------------------------
@dataclass
class AppState:
    subject_dir: Path | None = None
    raw_dir: Path | None = None
    mouth_dir: Path | None = None
    mpimg_dir: Path | None = None
    current_label: str | None = None

    is_running: bool = False
    stop_event: threading.Event | None = None
    worker_thread: threading.Thread | None = None


def safe_subject_name(name: str) -> str:
    bad = ["\\", "/", ":", "*", "?", '"', "<", ">", "|"]
    for b in bad:
        name = name.replace(b, "_")
    return name.strip()


def now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def get_next_index(out_dir: Path, prefix: str) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    max_n = 0
    for p in out_dir.glob(f"{prefix}_*.ply"):
        # 例: prefix_12.ply → "12" を読む
        last = p.stem.rsplit("_", 1)[-1]
        try:
            n = int(last)
            if n > max_n:
                max_n = n
        except ValueError:
            pass
    return max_n + 1

# -------------------------
# 3Cam処理（保存先を親フォルダ配下に変更した版）
# -------------------------
def capture_and_process_3cams_to_dirs(
    pipelines,
    profiles,
    pitch_label_deg: float,
    tag_R,
    tag_t,
    raw_dir: Path,
    mouth_dir: Path,
    mpimg_dir: Path,
    subject_prefix: str,
):
    """
    core.capture_and_process_3cams() をベースに、
    主要な出力を raw_dir / mouth_dir / mpimg_dir に保存する版。
    """
    raw_dir.mkdir(parents=True, exist_ok=True)
    mouth_dir.mkdir(parents=True, exist_ok=True)
    mpimg_dir.mkdir(parents=True, exist_ok=True)

    color_frames = [None] * len(pipelines)
    depth_frames = [None] * len(pipelines)
    aligns = [rs.align(rs.stream.color) for _ in pipelines]

    def grab_one(i):
        return pipelines[i].wait_for_frames()

    def make_T_from_Rt(R, t):
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = np.asarray(R, dtype=np.float64)
        T[:3, 3]  = np.asarray(t, dtype=np.float64).reshape(3)
        return T

    def transform_xyz(xyz, T):
        xyz = np.asarray(xyz, dtype=np.float64).reshape(3)
        p = np.ones(4, dtype=np.float64)
        p[:3] = xyz
        q = T @ p
        return q[:3]

    # NUM_FRAMES 回まわして最後のフレームを採用（元スクリプト踏襲）
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=len(pipelines)) as ex:
        for _ in range(core.NUM_FRAMES):
            futures = [ex.submit(grab_one, i) for i in range(len(pipelines))]
            framesets = [f.result() for f in futures]
            for i, fs in enumerate(framesets):
                aligned = aligns[i].process(fs)
                depth = aligned.get_depth_frame()
                color = aligned.get_color_frame()
                if not depth or not color:
                    raise RuntimeError("フレーム取得に失敗しました")
                depth_frames[i] = depth
                color_frames[i] = color

    timestamp = now_stamp()

    # 点群生成（Y反転済みは元スクリプト踏襲）
    pcds = []
    raw_pcds = []
    for i in range(len(core.SERIALS)):
        pcd, pcd_raw = core.frames_to_pointcloud(
            color_frames[i], depth_frames[i], profiles[i],
            apply_flip=True, return_raw=True
        )
        pcds.append(pcd)
        raw_pcds.append(pcd_raw)

    # 各カメラ raw PLY 保存（座標変換なし）
    for i, pcd_raw in enumerate(raw_pcds):
        raw_path = raw_dir / f"face_cam{i}_raw_{int(pitch_label_deg)}deg_{timestamp}.ply"
        o3d.io.write_point_cloud(str(raw_path), pcd_raw)

    # ICPでcam1/cam2をcam0へ（元スクリプト踏襲）
    base_pcd = pcds[0]
    T_1_to_0_icp = core.icp_to_cam0(pcds[1], base_pcd, core.T_1_to_0, source_cam_index=1)
    T_2_to_0_icp = core.icp_to_cam0(pcds[2], base_pcd, core.T_2_to_0, source_cam_index=2)

    pcd0_aligned = base_pcd
    pcd1_aligned = core.copy.deepcopy(pcds[1]); pcd1_aligned.transform(T_1_to_0_icp)
    pcd2_aligned = core.copy.deepcopy(pcds[2]); pcd2_aligned.transform(T_2_to_0_icp)

    merged_pcd = o3d.geometry.PointCloud()
    merged_pcd += pcd0_aligned
    merged_pcd += pcd1_aligned
    merged_pcd += pcd2_aligned

    merged_path = raw_dir / f"face_3cams_geom_merged_{int(pitch_label_deg)}deg_{timestamp}.ply"
    o3d.io.write_point_cloud(str(merged_path), merged_pcd)

    # 口切り出し（MediaPipe）
    lip_results = []
    for cam_idx in range(len(core.SERIALS)):
        if cam_idx == 0:
            T_cam_to_0 = np.eye(4, dtype=np.float64)
        elif cam_idx == 1:
            T_cam_to_0 = T_1_to_0_icp
        else:
            T_cam_to_0 = T_2_to_0_icp

        res = core.detect_lip_3d_for_camera(
            color_frames[cam_idx],
            depth_frames[cam_idx],
            profiles[cam_idx],
            T_cam_to_0,
            cam_index=cam_idx
        )
        lip_results.append(res)

    # 元スクリプトの優先ルール踏襲（変数名そのまま）
    if 0 <= pitch_label_deg <= 21.0:
        camera_priority = [0, 2]
    elif pitch_label_deg > 21.0:
        camera_priority = [2, 0]
    if -21.0 <= pitch_label_deg < 0:
        camera_priority = [0, 1]
    elif pitch_label_deg < -21.0:
        camera_priority = [1, 0]

    selected = None
    for idx in camera_priority:
        if idx < len(lip_results) and lip_results[idx].get("ok"):
            selected = lip_results[idx]
            break

    if selected is None or not selected.get("ok"):
        # 口検出失敗：保存はしない（要件どおり最低限）
        return

    pts = selected["points_cam0"]

    # Cam0->Tag 変換（元スクリプト踏襲）
    T_cam0_to_tag_raw = make_T_from_Rt(tag_R, tag_t)
    T_cam0_to_tag = T_cam0_to_tag_raw @ core.T_FLIP

    # Tag座標系へ（4点）
    pts_tag = {k: transform_xyz(v, T_cam0_to_tag) for k, v in pts.items()}
    lip_center_tag = (pts_tag["upper"] + pts_tag["lower"] + pts_tag["left"] + pts_tag["right"]) / 4.0

    face_landmarks = selected["face_landmarks"]
    cam_index = selected["camera_index"]

    # annotated画像（MediaPipe描画）保存（要件）
    annotated = selected.get("annotated_image", None)
    if annotated is not None:
        img_path = mpimg_dir / f"lip_cam{cam_index}_{timestamp}.png"
        cv2.imwrite(str(img_path), annotated)

    # 唇外周ポリゴン作成
    h, w, _ = np.asanyarray(color_frames[cam_index].get_data()).shape
    lip_poly = core.build_outer_lip_polygon(face_landmarks, w, h)

    # intrinsics & T_cam_to_cam0
    color_intr = color_frames[cam_index].profile.as_video_stream_profile().get_intrinsics()
    if cam_index == 0:
        T_cam_to_cam0 = np.eye(4, dtype=np.float64)
    elif cam_index == 1:
        T_cam_to_cam0 = T_1_to_0_icp
    else:
        T_cam_to_cam0 = T_2_to_0_icp

    mouth_pcd = core.crop_pcd_by_lip_polygon_project(
        merged_pcd=merged_pcd,
        lip_poly_px=lip_poly,
        color_intrinsics=color_intr,
        T_cam_to_cam0=T_cam_to_cam0,
        depth_frame=depth_frames[cam_index],
        depth_tol_m=0.01,
        mask_dilate_px=0,
    )

    if mouth_pcd is None or len(mouth_pcd.points) == 0:
        return

    mouth_pcd = core.keep_largest_cluster_dbscan(mouth_pcd, eps=0.006, min_points=30)
    if mouth_pcd is None or len(mouth_pcd.points) == 0:
        return

    def transform_pcd_points(pcd, T):
        pts0 = np.asarray(pcd.points)
        pts_h = np.hstack([pts0, np.ones((len(pts0), 1), dtype=np.float64)])
        pts2 = (T @ pts_h.T).T[:, :3]
        pcd2 = o3d.geometry.PointCloud()
        pcd2.points = o3d.utility.Vector3dVector(pts2)
        if pcd.has_colors():
            pcd2.colors = pcd.colors
        return pcd2

    mouth_pcd_tag = transform_pcd_points(mouth_pcd, T_cam0_to_tag)

    # 唇中心を原点にし、口ローカル軸で整列（元スクリプト踏襲）
    pts_m = np.asarray(mouth_pcd_tag.points)
    pts_centered = pts_m - lip_center_tag.reshape(1, 3)

    x_axis = pts_tag["right"] - pts_tag["left"]
    x_axis = x_axis / (np.linalg.norm(x_axis) + 1e-9)

    y_axis = pts_tag["upper"] - pts_tag["lower"]
    y_axis = y_axis - np.dot(y_axis, x_axis) * x_axis
    y_axis = y_axis / (np.linalg.norm(y_axis) + 1e-9)

    z_axis = np.cross(x_axis, y_axis)
    z_axis = z_axis / (np.linalg.norm(z_axis) + 1e-9)

    R_mouth = np.stack([x_axis, y_axis, z_axis], axis=1)
    pts_local = (R_mouth.T @ pts_centered.T).T

    mouth_out = o3d.geometry.PointCloud()
    mouth_out.points = o3d.utility.Vector3dVector(pts_local)
    if mouth_pcd_tag.has_colors():
        mouth_out.colors = mouth_pcd_tag.colors

    idx = get_next_index(mouth_dir, subject_prefix)
    mouth_path = mouth_dir / f"{subject_prefix}_{idx}.ply"
    o3d.io.write_point_cloud(str(mouth_path), mouth_out)

# -------------------------
# プロトコル本体（別スレッド）
# -------------------------
def protocol_worker(page, state, set_status_threadsafe, preview, capture_event, quit_event):
    pipelines = []
    profiles = []
    detector = core.create_detector()
    set_status_threadsafe("RealSenseカメラを初期化中…")
    pitch_hist = core.deque(maxlen=10)

    try:
        # 3台起動
        for serial in core.SERIALS:
            pipeline, profile = core.create_pipeline(serial)
            set_status_threadsafe(f"RealSenseカメラ {serial} 起動完了")
            pipelines.append(pipeline)
            profiles.append(profile)

        camera_params = core.get_color_intrinsics_from_profile(profiles[0])

        label = state.current_label
        if not label:
            raise RuntimeError("current_label が未設定です。")

        raw_dir_label   = state.raw_dir   / label
        mouth_dir_label = state.mouth_dir / label
        mpimg_dir_label = state.mpimg_dir / label

        raw_dir_label.mkdir(parents=True, exist_ok=True)
        mouth_dir_label.mkdir(parents=True, exist_ok=True)
        mpimg_dir_label.mkdir(parents=True, exist_ok=True)

        is_processing = False
        set_status_threadsafe("プロトコル実行中：OpenCVウィンドウで 'c' を押すと撮影します（ARマーカー必須）")

        while not state.stop_event.is_set() and not quit_event.is_set():
            frames0 = pipelines[0].wait_for_frames()
            color0 = frames0.get_color_frame()
            if not color0:
                continue

            color_image0 = np.asanyarray(color0.get_data())
            gray0 = cv2.cvtColor(color_image0, cv2.COLOR_BGR2GRAY)

            results = detector.detect(
                gray0,
                estimate_tag_pose=True,
                camera_params=camera_params,
                tag_size=core.TAG_SIZE_M
            )

            matched_any = False
            frame_vis = color_image0.copy()

            # Tag姿勢（検出できた場合のみ有効）
            R_tag = None
            t_tag = None
            pitch_deg_smooth = 0.0
            roll_deg = pitch_deg = yaw_deg = 0.0

            for r in results:
                R_tag = r.pose_R
                t_tag = r.pose_t

                roll, pitch, yaw = core.rotation_matrix_to_euler(R_tag)
                pitch = -pitch  # 元スクリプト踏襲

                roll_deg = float(np.degrees(roll))
                pitch_deg = float(np.degrees(pitch))
                yaw_deg = float(np.degrees(yaw))

                pitch_hist.append(pitch_deg)
                pitch_deg_smooth = sum(pitch_hist) / len(pitch_hist)

                matched_any = True
                break

            capture_ready = matched_any  # 要件：ARマーカー必須

            status = "READY" if capture_ready else "NG"
            cv2.putText(frame_vis, f"CAPTURE: {status}",
                        (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (0, 255, 0) if capture_ready else (0, 0, 255), 2)

            if is_processing:
                cv2.putText(frame_vis, "PROCESSING... DO NOT MOVE",
                            (30, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            if matched_any:
                cv2.putText(frame_vis, f"R:{roll_deg:+.1f}  P:{pitch_deg_smooth:+.1f}  Y:{yaw_deg:+.1f} [deg]",
                            (30, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            else:
                cv2.putText(frame_vis, "R:--  P:--  Y:-- [deg]",
                            (30, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            # 1) 毎フレーム、Fletへ表示
            ok, buf = cv2.imencode(".jpg", frame_vis, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            if ok:
                b64 = base64.b64encode(buf).decode("ascii")
                page.run_thread(lambda b64=b64: setattr(preview, "src_base64", b64))
                page.run_thread(page.update)

            # 2) 撮影トリガ（Flet側で c が押されたら capture_event が立つ）
            if capture_event.is_set():
                capture_event.clear()
                if capture_ready and (not is_processing) and (R_tag is not None) and (t_tag is not None):
                    is_processing = True
                    try:
                        capture_and_process_3cams_to_dirs(
                            pipelines, profiles,
                            pitch_label_deg=pitch_deg_smooth,
                            tag_R=R_tag, tag_t=t_tag,
                            raw_dir=raw_dir_label,
                            mouth_dir=mouth_dir_label,
                            mpimg_dir=mpimg_dir_label,
                            subject_prefix=state.subject_dir.name,
                        )
                        set_status_threadsafe(f"保存しました（pitch={pitch_deg_smooth:.2f}deg）")
                    except Exception as e:
                        set_status_threadsafe(f"保存処理でエラー: {e}")
                    finally:
                        is_processing = False
                else:
                    set_status_threadsafe("c入力：ARマーカー未検出のため撮影しません（NG）")

            # 3) quit_event が立ったら終了（q）
            if quit_event.is_set():
                state.stop_event.set()
                break

    except Exception:
        set_status_threadsafe("プロトコルで例外:\n" + traceback.format_exc())
    finally:
        for p in pipelines:
            try:
                p.stop()
            except Exception:
                pass
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass


# -------------------------
# Flet UI
# -------------------------
def main(page: ft.Page):
    page.title = "口周辺点群PLY 収録GUI"
    page.window_width = 820
    page.window_height = 520
    print(hasattr(page, "call_from_thread"))

    state = AppState()

    subject_name = ft.TextField(label="親フォルダ名（被験者名など）", width=520)
    status = ft.Text(value="未作成", selectable=True)
    paths_view = ft.Text(value="", selectable=True)

    DUMMY_PNG_B64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMB/ax0f9kAAAAASUVORK5CYII="

    preview = ft.Image(src_base64=DUMMY_PNG_B64, width=640, height=360)
    page.add(preview)

    # 共有イベント（workerと共有）
    capture_event = threading.Event()
    quit_event = threading.Event()

    def on_key(e: ft.KeyboardEvent):
        if e.key.lower() == "c":
            capture_event.set()   # 1回分の撮影トリガ
        elif e.key.lower() == "q":
            quit_event.set()      # 停止

    page.on_keyboard_event = on_key

    def set_status(msg: str):
        status.value = msg
        page.update()

    def set_status_threadsafe(msg: str):
        # 別スレッドからUI更新するため
        page.run_thread(lambda: set_status(msg))

    def set_paths():
        if state.subject_dir is None:
            paths_view.value = ""
        else:
            paths_view.value = (
                f"親フォルダ: {state.subject_dir}\n"
                f"raw_ply: {state.raw_dir}\n"
                f"mouth_ply: {state.mouth_dir}\n"
                f"mediapipe_img: {state.mpimg_dir}\n"
                f"実行中: {state.is_running}"
            )
        page.update()

    def on_create_folder(_):
        name = safe_subject_name(subject_name.value or "")
        if not name:
            set_status("親フォルダ名が空です。")
            return

        base = Path.cwd()
        subject_dir = base / name
        raw_dir = subject_dir / "raw_ply"
        mouth_dir = subject_dir / "mouth_ply"
        mpimg_dir = subject_dir / "mediapipe_img"

        subject_dir.mkdir(parents=True, exist_ok=True)
        raw_dir.mkdir(parents=True, exist_ok=True)
        mouth_dir.mkdir(parents=True, exist_ok=True)
        mpimg_dir.mkdir(parents=True, exist_ok=True)

        state.subject_dir = subject_dir
        state.raw_dir = raw_dir
        state.mouth_dir = mouth_dir
        state.mpimg_dir = mpimg_dir

        set_status(f"作成しました: {subject_dir}")
        set_paths()

    def on_start_for_label(label: str):
        if state.subject_dir is None or state.raw_dir is None or state.mouth_dir is None or state.mpimg_dir is None:
            set_status("先に親フォルダを作成してください。")
            return
        if state.is_running:
            set_status("すでにプロトコル実行中です。")
            return
        
        capture_event.clear()
        quit_event.clear()

        state.current_label = label
        state.stop_event = threading.Event()
        state.is_running = True

        # プロトコルは別スレッドで実行（fletSound.py同様にUIを固めない）
        t = threading.Thread(
            target=protocol_worker,
            args=(page, state, set_status_threadsafe, preview, capture_event, quit_event),
            daemon=True
        )
        state.worker_thread = t
        t.start()

        set_status(f"{label}の撮影開始しました。OpenCVウィンドウで 'c' を押すと撮影します。停止はStopボタンまたはOpenCVで'q'。")
        set_paths()

    def on_stop(_):
        if not state.is_running or state.stop_event is None:
            set_status("プロトコルは実行中ではありません。")
            return
        if state.current_label is None:
            set_status("ラベルが未選択です（ラベルの録音Startボタンから開始してください）。")
            return
        state.stop_event.set()
        state.is_running = False
        set_status("停止要求を出しました（OpenCVウィンドウが閉じるまで待ちます）。")
        set_paths()

    btn_create = ft.ElevatedButton(text="親フォルダ作成", on_click=on_create_folder)
    btn_stop = ft.ElevatedButton(text="プロトコル停止", on_click=on_stop)
    # ラベル別ボタン生成
    label_buttons = [
        ft.ElevatedButton(text=f"{lbl} プロトコル開始",
                        on_click=lambda e, l=lbl: on_start_for_label(l))
        for lbl in LABELS
    ]

    page.add(
        ft.Column(
            [
                subject_name,
                ft.Row([btn_create, btn_stop]),
                ft.Row(label_buttons),
                ft.Divider(),
                status,
                ft.Divider(),
                paths_view,
                ft.Text("※撮影タイミングは OpenCVウィンドウ上で 'c'（ARマーカー検出時のみ）"),
            ],
            spacing=10,
        )
    )

if __name__ == "__main__":
    ft.app(target=main)
