import os, glob, json, cv2
import numpy as np

# ==== あなたの環境に合わせる部分 =========================
DATASET_ROOT = "dataset"         # ルート
IMG_DIR_TRAIN = os.path.join(DATASET_ROOT, "images", "train")
#IMG_DIR_VAL   = os.path.join(DATASET_ROOT, "images", "val")    # ないなら後でスキップ
JSON_DIR      = os.path.join(DATASET_ROOT, "annos_labelme")    # JSON を置いた場所
OUT_LABEL_DIR = os.path.join(DATASET_ROOT, "labels")           # 出力先 (train/val 下に .txt)
# Labelme の実ラベル名に合わせる（厳密一致）:
CLASS_MAP = {
    "bero": 0,   # 例: "舌":0 とか "Tongue":0 など実際の名前で
}
# ========================================================

def find_image_for_base(base):
    for ext in (".jpg", ".jpeg", ".png", ".JPG", ".PNG"):
        p = os.path.join(IMG_DIR_TRAIN, base + ext)
        if os.path.exists(p): return p, "train"
        p = os.path.join(IMG_DIR_VAL, base + ext)
        if os.path.exists(p): return p, "val"
    return None, None

def rect_to_polygon(pts):
    # rectangle: points = [ (x1,y1), (x2,y2) ] (対角) → 4点矩形へ
    (x1,y1),(x2,y2) = pts
    x_min, x_max = min(x1,x2), max(x1,x2)
    y_min, y_max = min(y1,y2), max(y1,y2)
    return [(x_min,y_min),(x_max,y_min),(x_max,y_max),(x_min,y_max)]

def poly_to_bbox_norm(poly_xy, W, H):
    xs = poly_xy[:,0]; ys = poly_xy[:,1]
    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()
    xc = ((x_min + x_max) / 2.0) / W
    yc = ((y_min + y_max) / 2.0) / H
    w  = (x_max - x_min) / W
    h  = (y_max - y_min) / H
    return xc, yc, w, h

def write_lines(txt_path, lines):
    os.makedirs(os.path.dirname(txt_path), exist_ok=True)
    with open(txt_path, "w", encoding="utf-8") as f:
        for ln in lines:
            f.write(ln + "\n")

def convert_split(json_dir):
    json_list = sorted(glob.glob(os.path.join(json_dir, "*.json")))
    total, pos, neg, skipped = 0,0,0,0
    for jp in json_list:
        total += 1
        base = os.path.splitext(os.path.basename(jp))[0]

        img_path, split = find_image_for_base(base)
        if img_path is None:
            print(f"[WARN] image not found for {base} (json={jp})")
            skipped += 1
            continue

        im = cv2.imread(img_path)
        if im is None:
            print(f"[WARN] failed to read image: {img_path}")
            skipped += 1
            continue
        H, W = im.shape[:2]

        try:
            data = json.load(open(jp, "r", encoding="utf-8"))
        except Exception as e:
            print(f"[WARN] failed to load json: {jp} ({e})")
            skipped += 1
            continue

        lines = []
        found = 0
        for s in data.get("shapes", []):
            label = s.get("label")
            if label not in CLASS_MAP:
                # ラベル名が一致しない → 変換されない
                continue

            st = s.get("shape_type", "polygon")
            pts = s.get("points", [])
            if not pts or len(pts) < 2:
                continue

            if st == "rectangle" and len(pts) == 2:
                pts = rect_to_polygon(pts)
                st = "polygon"

            if st != "polygon":
                # line/point 等はスキップ
                continue

            pts = np.array(pts, dtype=np.float32)
            # 画像範囲外の点はクリップ
            pts[:,0] = np.clip(pts[:,0], 0, W-1)
            pts[:,1] = np.clip(pts[:,1], 0, H-1)

            # YOLO 形式: class xc yc w h px1 py1 px2 py2 ...
            bbox = poly_to_bbox_norm(pts, W, H)
            pts_norm = np.stack([pts[:,0]/W, pts[:,1]/H], axis=1).reshape(-1)
            seg_str = " ".join([f"{v:.6f}" for v in pts_norm])
            cls_id = CLASS_MAP[label]
            line = f"{cls_id} {bbox[0]:.6f} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f} {seg_str}"
            lines.append(line)
            found += 1

        out_txt = os.path.join(OUT_LABEL_DIR, split, base + ".txt")
        write_lines(out_txt, lines)

        if found > 0:
            pos += 1
        else:
            neg += 1
            # 理由をヒント表示
            # 1) ラベル不一致 2) shape_type 不一致 の可能性が高い
            print(f"[INFO] no segments written for {base}.txt  (labels match? shape_type polygon/rectangle?)")

    print(f"[DONE] total JSON: {total}, positive: {pos}, negative(empty): {neg}, skipped(no image/err): {skipped}")

if __name__ == "__main__":
    convert_split(JSON_DIR)
