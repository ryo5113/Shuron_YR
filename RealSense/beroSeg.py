import argparse, time, sys
from collections import deque
import cv2, numpy as np
from ultralytics import YOLO
import torch
from collections import deque
import time

def parse_args():
    ap = argparse.ArgumentParser(description="Realtime tongue segmentation (mask only)")
    ap.add_argument("--model", required=True, help="path to best.pt or exported model dir")
    ap.add_argument("--source", default=0, help="camera index or video file/URL")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou",  type=float, default=0.50)
    ap.add_argument("--max_det", type=int, default=1, help="max instances per frame")
    ap.add_argument("--device", default=None, help="None -> auto, or 'cpu' or '0' (GPU)")
    ap.add_argument("--ema", type=float, default=0.4, help="EMA smoothing for metrics (0..1)")
    ap.add_argument("--show", action="store_true", help="cv2.imshow window")
    ap.add_argument("--save", type=str, default=None, help="save video to this path (.mp4)")
    ap.add_argument("--warmup-sec", type=float, default=1.0, help="seconds to measure capture FPS before recording")
    return ap.parse_args()

def pick_largest_mask(masks):
    # masks: (N, H', W'), uint8{0,1}
    areas = masks.reshape(masks.shape[0], -1).sum(axis=1)
    idx = int(np.argmax(areas))
    return masks[idx]

def overlay_mask(bgr, mask_bin, color=(180,200,255), alpha=0.55, draw_contour=True):
    h, w = bgr.shape[:2]
    m = (mask_bin*255).astype(np.uint8)
    m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
    out = bgr.copy()
    out[m>0] = out[m>0]*(1-alpha) + np.array(color, np.float32)*alpha
    out = out.astype(np.uint8)
    if draw_contour:
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, cnts, -1, (255,255,255), 2, cv2.LINE_AA)
    return out, m

def metrics_from_mask(mask_bin):
    ys, xs = np.where(mask_bin>0)
    if xs.size == 0:
        return dict(expose=0.0, forward=0.0)
    H, W = mask_bin.shape[:2]
    expose = xs.size / float(W*H)             # 画面面積比（簡易）
    forward = xs.max() / float(W)             # “右方向”への突出（0..1）
    return dict(expose=expose, forward=forward)

def ema(prev, val, a):
    return val if prev is None else prev*(1-a) + val*a

def main():
    args = parse_args()
    device = args.device if args.device is not None else ("0" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] device={device}")

    model = YOLO(args.model)
    task = getattr(model.model, "task", None)
    if task != "segment":
        print(f"[ERR] model task is '{task}', expected 'segment'. Wrong weights?")
        sys.exit(1)
    names = model.names  # {0:'class_name', ...}

    cap = cv2.VideoCapture(args.source if str(args.source).isdigit() else str(args.source))
    if not cap.isOpened():
        print(f"[ERR] failed to open source: {args.source}")
        sys.exit(1)

    # 保存：writerは後で作る（FPS確定後）
    writer = None
    VIDEO_FOURCC = "mp4v"
    rec_t0 = None
    measured_fps = None

    ema_expose = None
    ema_forward = None
    tbuf = deque(maxlen=60)     # 実効FPS計測用

    while True:
        ok, frame = cap.read()
        if not ok: break

        r = model.predict(
            source=frame, imgsz=args.imgsz, conf=args.conf, iou=args.iou,
            device=device, verbose=False, max_det=args.max_det
        )[0]

        vis = r.plot(conf=True, boxes=True, labels=True, masks=True)
        H, W = vis.shape[:2]

        if r.masks is not None and len(r.masks.data)>0:
            masks = r.masks.data.cpu().numpy().astype(np.uint8)  # (N,H',W')
            m = pick_largest_mask(masks)
            vis, m_full = overlay_mask(vis, m, alpha=0.55, draw_contour=True)

            # 簡易指標（口幅での正規化は未実装：FaceMesh併用時に置換）
            met = metrics_from_mask(m_full)
            ema_expose = ema(ema_expose, met["expose"], args.ema)
            ema_forward = ema(ema_forward, met["forward"], args.ema)

            txt = f"Expose(EMA): {ema_expose:.3f}  Forward(EMA): {ema_forward:.3f}"
            #cv2.putText(vis, txt, (10,28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 3, cv2.LINE_AA)
            #cv2.putText(vis, txt, (10,28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 1, cv2.LINE_AA)
        #else:
            #cv2.putText(vis, "NO MASK", (10,28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2, cv2.LINE_AA)

        # FPS
        tbuf.append(time.time())
        if len(tbuf) >= 2:
            fps = (len(tbuf)-1) / (tbuf[-1]-tbuf[0])
        else:
            fps = 0.0
        #cv2.putText(vis, f"FPS: {fps:.1f}", (10,58), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 3, cv2.LINE_AA)
        #cv2.putText(vis, f"FPS: {fps:.1f}", (10,58), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 1, cv2.LINE_AA)

        # 実効FPSの更新（処理込み）
        tbuf.append(time.perf_counter())
        if len(tbuf) >= 2:
            measured_fps = (len(tbuf) - 1) / (tbuf[-1] - tbuf[0])
            #cv2.putText(vis, f"FPS: {measured_fps:.1f}", (10, 58),
                        #cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3, cv2.LINE_AA)
            #cv2.putText(vis, f"FPS: {measured_fps:.1f}", (10, 58),
                        #cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)

        # ====== 録画：保存FPSを“撮影FPS”に合わせる ======
        if args.save:
            now = time.perf_counter()

            # Start measuring effective FPS from the first frame
            if writer is None:
                if rec_t0 is None:
                    rec_t0 = now
                    frames_for_fps = 0
                frames_for_fps += 1

                # After warmup-sec, compute effective FPS and open VideoWriter with that FPS
                if (now - rec_t0) >= max(0.2, float(args.warmup_sec)):
                    eff_fps = max(1.0, frames_for_fps / (now - rec_t0))
                    h_out, w_out = vis.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*VIDEO_FOURCC)
                    writer = cv2.VideoWriter(args.save, fourcc, eff_fps, (w_out, h_out))
                    print(f"[Video] init writer @ eff_fps ≈ {eff_fps:.2f}  -> {args.save}")
                    # 初期フレームは破棄（RStest5と同様の思想：Writer生成後のフレームから記録）
            else:
                writer.write(vis)

        if args.show or writer is None:
            cv2.imshow("Tongue Seg (ESC to quit)", vis)
            if cv2.waitKey(1) & 0xFF == 27:
                break

    cap.release()
    if writer is not None: 
        writer.release()
    cv2.destroyAllWindows()
    print("[INFO] done.")

if __name__ == "__main__":
    main()

# 実行例）python beroSeg.py --model .\runs\segment\tongue_seg_cpu\weights\best.pt --source 0 --imgsz 640 --conf 0.35 --show --save out.mp4