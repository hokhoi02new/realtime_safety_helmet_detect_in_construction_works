import cv2
import argparse
from ultralytics import YOLO

# Load model đã train
model = YOLO("runs/detect/train/weights/best.pt")

CLASS_NAMES = ["helmet", "no_helmet"]
COLORS = {"helmet": (0, 255, 0), "no_helmet": (0, 0, 255)}

def run_inference(video_path):
    cap = cv2.VideoCapture(video_path)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, verbose=False)
        no_helmet_count = 0

        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                xyxy = box.xyxy[0].tolist()
                label = CLASS_NAMES[cls_id]
                color = COLORS[label]

                if label == "no_helmet":
                    no_helmet_count += 1

                x1, y1, x2, y2 = map(int, xyxy)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Hiển thị số người không đội mũ
        cv2.putText(frame, f"Warning No Helmet: {no_helmet_count}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

        cv2.imshow("Helmet Detection - Video", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Helmet Detection on Video")
    parser.add_argument("--video", type=str, required=True, help="Path to input video")
    args = parser.parse_args()

    run_inference(args.video)
