"""Live real-time YOLOv5s ONNX object detection viewer for kitchen videos or webcam.

Reuses letterbox preprocessing, NMS postprocessing, and drawing helpers from
`extract_bbox_features.py`. Press `q` or `Esc` to quit, `Space` to pause/resume.

Examples:
    python scripts/run_live_detection.py --source path/to/P01_01.MP4 --start 2:15
    python scripts/run_live_detection.py --source path/to/P01_01.MP4 --start 120 --duration 30
    python scripts/run_live_detection.py --source 0
    python scripts/run_live_detection.py --source path/to/P01_01.MP4 --start 1:30 --save-video exports/demo_annotated.mp4
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import onnxruntime as ort

# Ensure sibling import works regardless of working directory
sys.path.insert(0, str(Path(__file__).resolve().parent))
from extract_bbox_features import (  # noqa: E402
    draw_detections,
    postprocess_predictions,
    preprocess_frame,
    resolve_model_classes,
    resolve_path,
)


def open_capture(source: str, workspace: Path) -> cv2.VideoCapture:
    """Open a webcam index (e.g. '0') or video file path."""
    if source.isdigit():
        cap = cv2.VideoCapture(int(source))
    else:
        video_path = resolve_path(workspace, source)
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")
        cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video source: {source}")
    return cap


def parse_timestamp(value: str) -> float:
    """Parse seconds ('90', '90.5') or 'MM:SS' / 'HH:MM:SS' into total seconds."""
    parts = value.strip().split(":")
    if len(parts) == 1:
        return float(parts[0])
    if len(parts) == 2:
        return int(parts[0]) * 60.0 + float(parts[1])
    if len(parts) == 3:
        return int(parts[0]) * 3600.0 + int(parts[1]) * 60.0 + float(parts[2])
    raise ValueError(f"Invalid timestamp format: {value!r} (expected seconds, MM:SS, or HH:MM:SS)")


def draw_hud(
    frame,
    fps: float,
    pos_sec: float,
    dets: list[tuple[int, float, float, float, float, float]],
    class_names: tuple[str, ...],
) -> None:
    """Overlay timestamp, FPS, and active detected object names in the top-left corner."""
    mins, secs = divmod(int(pos_sec), 60)
    seen = sorted({class_names[d[0]] if d[0] < len(class_names) else f"class_{d[0]}" for d in dets})
    status = f"{mins:02d}:{secs:02d} | FPS: {fps:.1f} | Objects: {', '.join(seen) if seen else 'none'}"
    cv2.rectangle(frame, (8, 8), (min(frame.shape[1] - 8, 18 + len(status) * 9), 38), (20, 20, 20), -1)
    cv2.putText(frame, status, (14, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)


def main() -> None:
    parser = argparse.ArgumentParser(description="Real-time YOLOv5s ONNX detection viewer.")
    parser.add_argument("--source", type=str, required=True, help="Video file path (.mp4/.mov) or webcam index ('0').")
    parser.add_argument("--onnx", type=str, default="exports/kitchen_yolov5s.onnx", help="ONNX model path.")
    parser.add_argument("--start", type=str, default="0", help="Start timestamp (seconds, MM:SS, or HH:MM:SS).")
    parser.add_argument("--duration", type=str, default=None, help="Optional clip length to process from --start.")
    parser.add_argument("--imgsz", type=int, default=640, help="ONNX static input resolution.")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold.")
    parser.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold.")
    parser.add_argument("--stride", type=int, default=1, help="Run inference every Nth frame (use 2-3 on 50 FPS video).")
    parser.add_argument("--save-video", type=str, default=None, help="Optional output .mp4 path for annotated video.")
    parser.add_argument("--no-display", action="store_true", help="Process without opening a GUI window.")
    args = parser.parse_args()

    workspace = Path(__file__).resolve().parent.parent
    onnx_path = resolve_path(workspace, args.onnx)
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    active_classes = resolve_model_classes(session)

    cap = open_capture(args.source, workspace)
    start_sec = parse_timestamp(args.start)
    duration_sec = parse_timestamp(args.duration) if args.duration is not None else None
    if start_sec > 0 and not args.source.isdigit():
        cap.set(cv2.CAP_PROP_POS_MSEC, start_sec * 1000.0)

    writer = None
    if args.save_video:
        out_video = resolve_path(workspace, args.save_video)
        out_video.parent.mkdir(parents=True, exist_ok=True)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
        fps_src = cap.get(cv2.CAP_PROP_FPS) or 25.0
        writer = cv2.VideoWriter(str(out_video), cv2.VideoWriter_fourcc(*"mp4v"), fps_src, (w, h))

    frame_idx = 0
    last_dets: list[tuple[int, float, float, float, float, float]] = []
    paused = False
    window_name = "Multimodal Cooking Assistant - YOLOv5s ONNX"

    try:
        while True:
            if not paused:
                ok, frame = cap.read()
                if not ok:
                    break
                pos_sec = (cap.get(cv2.CAP_PROP_POS_MSEC) or 0.0) / 1000.0
                if duration_sec is not None and (pos_sec - start_sec) > duration_sec:
                    break
                t0 = time.perf_counter()
                if frame_idx % max(1, args.stride) == 0:
                    tensor, meta = preprocess_frame(frame, args.imgsz)
                    raw_out = session.run(None, {input_name: tensor})[0]
                    last_dets = postprocess_predictions(raw_out, meta, args.conf, args.iou)
                dt = max(1e-6, time.perf_counter() - t0)
                vis = draw_detections(frame, last_dets, active_classes)
                draw_hud(vis, 1.0 / dt, pos_sec, last_dets, active_classes)
                if writer is not None:
                    writer.write(vis)
                frame_idx += 1

            if not args.no_display:
                cv2.imshow(window_name, vis)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):  # 'q' or Esc
                    break
                if key == ord(" "):
                    paused = not paused
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if not args.no_display:
            cv2.destroyAllWindows()

    print(f"Processed {frame_idx} frames.")


if __name__ == "__main__":
    main()
