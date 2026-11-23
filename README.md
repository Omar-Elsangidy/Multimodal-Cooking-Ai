# Multimodal Cooking Assistant — Smart Kitchen EDGE-AI (YOLOv5s)

Standalone 30-class kitchen object detection pipeline (`YOLOv5s` -> static `opset 12` ONNX for Hailo-8) with a live desktop viewer and Flask phone/tablet web dashboard.

## Project Structure

- [`kitchen_data.yaml`](kitchen_data.yaml) — 30-class EPIC-KITCHENS-aligned schema (`hand`, `pan`, `pot`, `lid`, `hob`, `tap`, `sink`, `knife`, `spatula`, `spoon`, `fork`, `cutting_board`, `bowl`, `plate`, `cup`, `bottle`, `jar`, `appliance`, `onion`, `tomato`, `potato`, `carrot`, `garlic`, `pepper`, `egg`, `cheese`, `bread`, `meat`, `vegetable`, `food`).
- [`notebooks/train_kitchen_yolov5_colab.ipynb`](notebooks/train_kitchen_yolov5_colab.ipynb) — Google Colab GPU notebook (`2 FPS` extraction, `yolov8l-worldv2.pt` auto-labeling, `100`-epoch `yolov5s` training, static ONNX export).
- [`models/kitchen_yolov5s_best.pt`](models/kitchen_yolov5s_best.pt) — Trained 30-class PyTorch weights.
- [`exports/kitchen_yolov5s.onnx`](exports/kitchen_yolov5s.onnx) — Verified Hailo-8-ready static ONNX model (`[1, 3, 640, 640]` -> `[1, 25200, 35]`, opset 12).
- [`scripts/extract_frames.py`](scripts/extract_frames.py) — Samples evenly spaced `.jpg` frames from `.mp4`/`.mov` videos.
- [`scripts/extract_bbox_features.py`](scripts/extract_bbox_features.py) — Extracts `(T, C * 10)` normalized bounding-box coordinate arrays (`.npy`) and optional `--save-vis` frames.
- [`scripts/run_live_detection.py`](scripts/run_live_detection.py) — Desktop OpenCV live detection viewer (`--start`, `--duration`, `--stride`, `--save-video`).
- [`prototype/web_dashboard.py`](prototype/web_dashboard.py) — Standalone Flask web dashboard streaming live detections and controls to any phone/tablet browser.

## Usage

### 1. Phone / Tablet Web Dashboard
```powershell
python -m prototype.web_dashboard --source "C:\Users\ASUS\Desktop\P02_10.MP4" --stride 3 --host 0.0.0.0 --port 5000
```
Open `http://<your-PC-WiFi-IP>:5000` on your phone (same Wi-Fi network).

### 2. Desktop OpenCV Live Viewer
```powershell
python scripts/run_live_detection.py --source "C:\Users\ASUS\Desktop\P02_10.MP4" --start 1:30 --stride 4
```

### 3. Extract Bounding-Box Features
```powershell
python scripts/extract_bbox_features.py --source path/to/video_or_frames --output exports/bbox_features.npy --save-vis exports/vis_preview
```
