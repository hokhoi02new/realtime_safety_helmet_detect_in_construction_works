import uvicorn
import cv2
import shutil
import os
from ultralytics import YOLO
from fastapi import FastAPI, UploadFile, File, Response, HTTPException
from fastapi.responses import FileResponse
from app.utils import predict_image, predict_and_draw  

app = FastAPI()

MODEL_PATH = "runs/detect/train/weights/best.pt"
CLASS_NAMES = ["helmet", "no_helmet"]
COLORS = {"helmet": (0, 255, 0), "no_helmet": (0, 0, 255)}

model = None 


def load_model(model_path=MODEL_PATH):
    global model
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")
    try:
        model = YOLO(model_path)
        print("YOLO model loaded successfully")
    except Exception as e:
        raise RuntimeError(f"Error loading model: {e}")
    
@app.on_event("startup")
async def startup_event():
    load_model(MODEL_PATH)

@app.get("/")
async def root():
    return {"message": "Safety helmet detection API is running"}

@app.get("/health")
async def health_check():   
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "ok", "model_loaded": True, "message": "API is running healthy"}


# API nhận ảnh trả về data JSON
@app.post("/detect/image")
async def detect(file: UploadFile = File(...)):
    image_bytes = await file.read()
    detections = predict_image(image_bytes)
    return {"detections": detections}


# aPI nhận ảnh trả về ảnh có bounding box
@app.post("/detect/image_bbox/")
async def detect_image(file: UploadFile = File(...)):
    image_bytes = await file.read()
    output_bytes = predict_and_draw(image_bytes)
    return Response(content=output_bytes, media_type="image/jpeg")


@app.post("/detect-video")
async def detect_video(file: UploadFile = File(...)):
    input_path = f"temp_{file.filename}"
    output_path = f"output_{file.filename}"
    with open(input_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    cap = cv2.VideoCapture(input_path)
    fourcc = cv2.VideoWriter_fourcc(*"avc1")  
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, verbose=False)

        for r in results:
            for box in r.boxes:
                conf = float(box.conf[0])
                if conf < 0.5:
                    continue

                cls_id = int(box.cls[0])
                xyxy = box.xyxy[0].tolist()
                label = CLASS_NAMES[cls_id]
                color = COLORS[label]

                x1, y1, x2, y2 = map(int, xyxy)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        out.write(frame)

    cap.release()
    out.release()
    os.remove(input_path)

    return FileResponse(output_path, media_type="video/mp4", filename=output_path)


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)