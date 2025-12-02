import cv2
import numpy as np        # ← これが必要！
from pupil_apriltags import Detector

# Detector 準備
detector = Detector(
    families="tag36h11",
    nthreads=4,
    quad_decimate=1.0,
    quad_sigma=0.0,
    refine_edges=True,
    decode_sharpening=0.25,
    debug=False
)

cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)

if not cap.isOpened():
    print("Camera not found")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # タグ検出
    results = detector.detect(gray)

    for r in results:
        tag_id = r.tag_id

        # 角の座標
        corners = np.int32(r.corners)

        # 角線を描画
        cv2.polylines(frame, [corners], True, (0, 255, 0), 2)

        # 中心点
        cX, cY = int(r.center[0]), int(r.center[1])
        cv2.circle(frame, (cX, cY), 5, (0, 0, 255), -1)

        cv2.putText(frame, f"ID: {tag_id}", (cX-20, cY-20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    cv2.imshow("AprilTag Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
