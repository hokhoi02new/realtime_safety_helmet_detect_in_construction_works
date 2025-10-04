import cv2
import argparse
from ultralytics import YOLO

model = YOLO("runs/detect/train/weights/best.pt")

CLASS_NAMES = ["helmet", "no_helmet"]
COLORS = {"helmet": (0, 255, 0), "no_helmet": (0, 0, 255)}

def run_inference(image_path='test/image/image1.jpg'):
    image = cv2.imread(image_path)
    results = model(image, verbose=False)

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
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            cv2.putText(image, f"{label} {conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    cv2.putText(image, f"Warning no helmet: {no_helmet_count}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

    cv2.imshow("Helmet Detection - Image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Helmet Detection on Image")
    parser.add_argument("--image", type=str, help="Path to input image")
    args = parser.parse_args()

    run_inference(args.image)
