from ultralytics import YOLO
import cv2
import numpy as np


model = YOLO("../runs/detect/train/weights/best.pt")  

def predict_image(image_bytes):
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    results = model(img)

    detections = []
    for r in results[0].boxes:
        detections.append({
            "class": int(r.cls),
            "confidence": float(r.conf),
            "bbox": r.xyxy[0].tolist()
        })

    return detections

def predict_and_draw(image_bytes):
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    results = model(img)
    
    for r in results[0].boxes:
        x1, y1, x2, y2 = map(int, r.xyxy[0].tolist())
        cls = int(r.cls)
        conf = float(r.conf)

        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, f"{model.names[cls]} {conf:.2f}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Encode ảnh ra bytes để trả về
    _, buffer = cv2.imencode(".jpg", img)
    return buffer.tobytes()