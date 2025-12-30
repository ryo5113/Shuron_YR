from ultralytics import YOLO

# 例：YOLOv8の分類用事前学習モデル
model = YOLO("yolov8n-cls.pt")

# data には、train/ と val/ を含む「データセットのルート」を渡す
# 例: ML_spectrogram_dataset_yolo_cls
results = model.train(
    data="spectrogram03_normmax_by_label_yolo_cls",  # データセットのルートフォルダ
    epochs=100,
    imgsz=1024,   # 必須項目なので指定例（変更可）
)
