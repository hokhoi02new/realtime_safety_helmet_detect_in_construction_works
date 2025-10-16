# 🪖 Real-Time Safety Helmet Detection using YOLOv8 & Faster R-CNN

## Introduction
This project implements a real-time safety helmet detection system to identify workers not wearing helmets in construction works, using two popular object detection approaches:  
- **YOLOv8 (one-stage detector)** → optimized for speed and real-time applications.  
- **Faster R-CNN (two-stage detector)** → optimized for higher accuracy.   

Achieved 92% mAP, can detect real-time with ~25 FPS on camera CCTV for construction site safety monitoring. 

---

## 🚀 Features
- Detect worker with **helmet** (green box) and **no_helmet** (red box).  
- **Warning** and **count** the number of workers without helmets.  
- Works on images, videos, and camera
- Provides API (FastAPI) + UI (Streamlit)

---

## 📌 Example Demo
Helmet ✅ (green) | No Helmet ❌ (red)

![gifdemo](https://github.com/user-attachments/assets/7c5e84a8-0450-49ec-9790-fd6cd22f07da)






---



## 📈 Results

| Model          | mAP (IoU=0.5)   | FPS (GTX 1660Ti) | Best For        |
|----------------|-----------------|------------------|-----------------|
| YOLOv8         | ~89.3%          | 25–30 FPS        | Real-time usage |
| Faster R-CNN   | ~91.1%          | 5–8 FPS          | High accuracy   |

---
## 📂 Project Structure
```
detect_workers_without_helmets_in_construction_site/
│── src/ 		  				# training source code
│   ├── train_yolo.ipynb        
│   └── train_faster_rcnn.py     
│── inference_image.py 		    #  detect on images
│── inference_video.py  		#  detect on video
│── inference_camera.py 		#  detect on camera
│── dataset/ 
│── runs/  		  				# YOLO training results
│── faster_rcnn/ 				# Faster R-CNN training results
│── test_video/ 
│── requirements.txt 
│── README.md
```
⚠️ Note: The `dataset/` folders and `faster_rcnn/fasterrcnn.pth` saved models are empty in this repository because the files are too large for GitHub.  
You can download them from the following links:

- **Dataset**: https://drive.google.com/drive/folders/1KT41S-1LHofH9k4G0flQd6Ne4B-HUIKz?usp=drive_link
- **faster_rcnn/fasterrcnn.pth**: https://drive.google.com/drive/folders/1qzCErR2joFjOoJnSe8atX-rKtPtFZg2U?usp=drive_link

---

## ⚙️ Usage
```bash
git clone https://github.com/hokhoi02new/realtime_safety_helmet_detect_in_construction_works.git
cd realtime_safety_helmet_detect_in_construction_works
pip install -r requirements.txt
```

#### 🌐 API (FastAPI):
- Run backend:
uvicorn app:app --host 0.0.0.0 --port 8000 --reload

API available at: http://127.0.0.1:8000/docs

#### 💻 UI (Streamlit):
- Run UI:
streamlit run app/app_ui.py

---

#### Inference

- Image:
```bash
python inference_image.py --image test/image/image1.jpg
```
- Video:
```bash
python inference_video.py --video test/video/vid1.mp4
```
- Camera:
```bash
python inference_camera.py --camera 0
```

