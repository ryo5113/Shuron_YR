import argparse, cv2, numpy as np
from ultralytics import YOLO

def overlay_mask(img, mask, alpha=0.5):
    h, w = img.shape[:2]
    if mask.dtype != np.uint8:
        mask = (mask*255).astype(np.uint8)
    mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
    out = img.copy()
    out[mask>0] = out[mask>0]* (1-alpha) + np.array([180,200,255], dtype=np.float32)*alpha
    out = out.astype(np.uint8)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, cnts, -1, (255,255,255), 2, cv2.LINE_AA)
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--image", required=True)
    ap.add_argument("--out", default="out.jpg")
    ap.add_argument("--imgsz", type=int, default=512)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.50)
    args = ap.parse_args()

    model = YOLO(args.model)
    print("[INFO] model task:", getattr(model.model, "task", "unknown"))

    im = cv2.imread(args.image)
    assert im is not None, f"image not found: {args.image}"

    r = model.predict(im, imgsz=args.imgsz, conf=args.conf, iou=args.iou, verbose=False, max_det=1)[0]
    print("[INFO] boxes:", 0 if r.boxes is None else len(r.boxes))
    print("[INFO] masks:", 0 if r.masks is None else len(r.masks.data))

    vis = im.copy()
    if r.masks is not None and len(r.masks.data) > 0:
        # 最大面積のマスクを採用
        masks = r.masks.data.cpu().numpy().astype(np.uint8)  # (N,H',W')
        areas = masks.reshape(masks.shape[0], -1).sum(axis=1)
        m = masks[int(areas.argmax())]
        vis = overlay_mask(vis, m, alpha=0.5)
    else:
        cv2.putText(vis, "NO MASK", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 2, cv2.LINE_AA)

    cv2.imwrite(args.out, vis)
    print("[INFO] saved:", args.out)

if __name__ == "__main__":
    main()

# 実行例　python .\test_image_seg.py --model .\runs\segment\tongue_seg_cpu3\weights\best.pt ^
# --image .\dataset\image\val\test001.jpg --out .\out_test.jpg --imgsz 512 --conf 0.25