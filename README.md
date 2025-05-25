# 🧠 Brain Tumor Detection and Segmentation with YOLOv11 and SAM2

This project demonstrates the use of **YOLO11 ** and **SAM2 (Segment Anything Model v2)** for detecting and segmenting brain tumors from MRI images.

🩺 The goal is to provide a fast and interpretable AI-assisted tool for radiologists to analyze tumor regions in MRI scans.

## 📂 Dataset

- MRI brain images
- Annotations in YOLO format (`data.yaml`)
- Divided into training and testing sets
- Supports three tumor types (e.g., glioma, meningioma, etc.)

## 🛠️ Technologies Used

- Python
- [Ultralytics YOLO11](https://github.com/ultralytics/ultralytics)
- [Ultralytics SAM2](https://github.com/ultralytics/ultralytics)
- Matplotlib
- Kaggle environment (GPU enabled)

## 🚀 Project Workflow

1. **Training YOLO11 Model**  
   Fine-tuned YOLOv8n (`yolo11n.pt`) on MRI brain tumor dataset using:
   - 20 epochs
   - image size: 640
   - GPU: `device=0`

2. **Detection**  
   Run object detection using the trained YOLOv11 model on test images.

3. **Segmentation with SAM2**  
   Detected bounding boxes are passed to **SAM2** to extract fine-grained tumor regions.

## 🧠 Model Code Highlights

```python
from ultralytics import YOLO, SAM

# Load trained YOLOv11 model
model = YOLO('yolo11n.pt')
results = model('test_image.jpg')

# Load SAM2 model
sam = SAM('sam2_b.pt')
sam_output = sam(results[0].orig_img, bboxes=results[0].boxes.xyxy, save=True)
