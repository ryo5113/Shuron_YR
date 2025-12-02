from ultralytics import YOLO
m = YOLO(r".\runs\segment\tongue_seg_cpu3\weights\best.pt")
print("task:", getattr(m.model, "task", None))  # 'segment' ならセグモデル
