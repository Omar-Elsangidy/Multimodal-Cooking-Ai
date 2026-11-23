# Multimodal Cooking Assistant — First Prototype

Unified senior project prototype connecting all three modules into a real-time cooking assistant pipeline:

```text
RGB Video / Webcam + Food Probe Temp
   ├─► 1. YOLO Object Detector (prototype/vision_detector.py)
   │      • 18-class kitchen schema (fixes 5-class pan-handle->knife & stove->pan errors)
   │      • Supports YOLO-World (18+ classes out-of-the-box) OR Hailo-ready YOLOv5s ONNX
   ├─► 2. TAS Action Segmentation (prototype/action_recognizer.py -> TAS/model.py)
   │      • 5 FPS ResNet-18 (512-D) + 2-Stage MS-TCN (TAS/Weights/mstcn_cooking.pth)
   │      • Predicts: Idle | Chopping | Stirring | Adding
   ├─► 3. Action-to-Step Bridge + Food Probe Monitor (prototype/action_step_bridge.py, temperature_monitor.py)
   │      • Maps coarse TAS actions + active YOLO objects -> CaptainCook4D recipe steps
   │      • Evaluates live Food Probe Thermometer thresholds & safety alerts
   └─► 4. CaptainCook4D DAG Recipe Engine (captaincook4d_project/src/recipe_engine.py)
          • Ranks 24 recipes, locks onto active recipe, validates preconditions & flags Order Errors
```

## Key Improvements in This Prototype

- **Expanded 30-Class EPIC-KITCHENS YOLO Schema** ([`kitchen_data.yaml`](kitchen_data.yaml) & [`notebooks/train_kitchen_yolov5_colab.ipynb`](notebooks/train_kitchen_yolov5_colab.ipynb)):
  - Expanded to 30 recipe-ready classes: `hand`, `pan`, `pot`, `lid`, `hob`, `tap`, `sink`, `knife`, `spatula`, `spoon`, `fork`, `cutting_board`, `bowl`, `plate`, `cup`, `bottle`, `jar`, `appliance`, `onion`, `tomato`, `potato`, `carrot`, `garlic`, `pepper`, `egg`, `cheese`, `bread`, `meat`, `vegetable`, `food`.
  - Colab v2 training uses `yolov8l-worldv2.pt` auto-labeling at `2 FPS` (tiny-box noise filtering), `100` epochs with `--cos-lr` and `--patience 20`, and saves `results.png`, `confusion_matrix.png`, and `PR_curve.png` to Google Drive.
  - Legacy 5-class [`exports/kitchen_yolov5s.onnx`](exports/kitchen_yolov5s.onnx) remains supported with automatic class detection from ONNX metadata.

## Running the First Prototype

### 1. Instant Integration Verification (Simulation)
```powershell
python -m prototype.run_prototype --demo-sim --detector onnx
```

### 2. Full Live Pipeline (OpenCV Split-Screen HUD)
```powershell
# Using YOLO-World (18+ kitchen classes immediately)
python -m prototype.run_prototype --source "C:\Users\ASUS\Desktop\P02_10.MP4" --detector world --stride 3

# Using your compiled YOLOv5s ONNX model (with 5-class confusion filter)
python -m prototype.run_prototype --source "C:\Users\ASUS\Desktop\P02_10.MP4" --detector onnx --stride 3

# Webcam live mode with pinned recipe
python -m prototype.run_prototype --source 0 --recipe scrambledeggs
```
Controls: `Space` pause/resume, `+`/`-` (or `W`/`S`) adjust simulated Food Probe temperature, `R` reset, `Q` quit.

### 3. Phone / Tablet Front-End Web UI (Flask)
```powershell
python -m prototype.web_dashboard --source "C:\Users\ASUS\Desktop\P02_10.MP4" --stride 3 --host 0.0.0.0 --port 5000
```
Open `http://localhost:5000` (or `http://<your-PC-IP>:5000` on a phone/tablet on the same Wi-Fi).

## Retraining YOLOv5s with All 30 Classes (Colab -> Hailo-8 ONNX)

1. Open [`notebooks/train_kitchen_yolov5_colab.ipynb`](notebooks/train_kitchen_yolov5_colab.ipynb) in Google Colab (**L4**, **A100**, or **T4** GPU).
2. Run all cells to extract `2 FPS` frames, auto-label with `yolov8l-worldv2.pt` across all 30 classes, train `yolov5s` (`100` epochs), save training charts, and export static ONNX (`[1, 3, 640, 640]`, opset 12).
3. Replace [`exports/kitchen_yolov5s.onnx`](exports/kitchen_yolov5s.onnx) and [`models/kitchen_yolov5s_best.pt`](models/kitchen_yolov5s_best.pt) — [`prototype/vision_detector.py`](prototype/vision_detector.py), [`scripts/extract_bbox_features.py`](scripts/extract_bbox_features.py), and [`scripts/run_live_detection.py`](scripts/run_live_detection.py) automatically adapt from ONNX metadata.
