"""CLI runner for the unified Multimodal Cooking Assistant First Prototype.

Examples:
    # 1. Run full live pipeline (YOLO + TAS MS-TCN + Probe + DAG) on a video clip
    python -m prototype.run_prototype --source "C:/Users/ASUS/Desktop/P02_10.MP4" --stride 3

    # 2. Run on webcam with specific recipe pinned and save annotated demo video
    python -m prototype.run_prototype --source 0 --recipe scrambledeggs --save-video exports/prototype_demo.mp4

    # 3. Use existing 5-class ONNX model with geometric pan-handle/stove filter
    python -m prototype.run_prototype --source "C:/Users/ASUS/Desktop/P02_10.MP4" --detector onnx --stride 3

    # 4. Run instant end-to-end integration simulation (no camera needed)
    python -m prototype.run_prototype --demo-sim
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from prototype.pipeline import MultimodalCookingPipeline  # noqa: E402


def parse_timestamp(value: str) -> float:
    """Parse seconds ('90', '90.5') or 'MM:SS' / 'HH:MM:SS' into total seconds."""
    parts = value.strip().split(":")
    if len(parts) == 1:
        return float(parts[0])
    if len(parts) == 2:
        return int(parts[0]) * 60.0 + float(parts[1])
    if len(parts) == 3:
        return int(parts[0]) * 3600.0 + int(parts[1]) * 60.0 + float(parts[2])
    raise ValueError(f"Invalid timestamp format: {value!r}")


def open_capture(source: str) -> cv2.VideoCapture:
    if source.isdigit():
        cap = cv2.VideoCapture(int(source))
    else:
        path = Path(source)
        if not path.is_absolute():
            path = (WORKSPACE_ROOT / path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"Video source not found: {path}")
        cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video source: {source}")
    return cap


def run_demo_simulation(pipeline: MultimodalCookingPipeline) -> None:
    """Verify full YOLO -> TAS -> Probe -> DAG integration with a realistic cooking sequence."""
    print("\n=== Multimodal Cooking Assistant — End-to-End Integration Simulation ===")
    pipeline.reset()

    # (timestamp_sec, yolo_dag_objects, tas_action, tas_conf, raw_labels, probe_override_c)
    script = [
        (0.0, {"pan", "egg", "bowl", "knife", "cutting board"}, "Idle", 0.92, ["pan", "egg", "bowl", "knife", "cutting_board"], 22.0),
        (1.0, {"knife", "cutting board", "onion"}, "Chopping", 0.88, ["hand", "knife", "cutting_board", "onion"], None),
        (1.4, {"knife", "cutting board", "onion"}, "Chopping", 0.90, ["hand", "knife", "cutting_board", "onion"], None),
        (7.0, {"knife", "cutting board", "tomato"}, "Chopping", 0.87, ["hand", "knife", "cutting_board", "tomato"], None),
        (7.4, {"knife", "cutting board", "tomato"}, "Chopping", 0.89, ["hand", "knife", "cutting_board", "tomato"], None),
        (13.0, {"pan", "spatula", "stove"}, "Stirring", 0.84, ["hand", "pan", "spatula", "stove"], 48.0),
        (13.4, {"pan", "spatula", "stove"}, "Stirring", 0.86, ["hand", "pan", "spatula", "stove"], 48.0),
        (19.0, {"egg", "bowl"}, "Adding", 0.85, ["hand", "egg", "bowl"], 135.0),
        (19.4, {"egg", "bowl"}, "Adding", 0.88, ["hand", "egg", "bowl"], 135.0),
        (25.0, {"bowl", "whisk", "egg"}, "Stirring", 0.91, ["hand", "bowl", "whisk", "egg"], 150.0),
        (25.4, {"bowl", "whisk", "egg"}, "Stirring", 0.92, ["hand", "bowl", "whisk", "egg"], 150.0),
    ]

    for t, dag_objs, action, conf, labels, temp_override in script:
        if temp_override is not None:
            pipeline.temp_monitor.set_temp(temp_override)
        probs = {k: (conf if k == action else round((1.0 - conf) / 3.0, 3)) for k in ("Idle", "Chopping", "Stirring", "Adding")}
        state = pipeline.step_signals(
            timestamp_sec=t,
            dag_objects=dag_objs,
            tas_action=action,
            tas_conf=conf,
            tas_probs=probs,
            detected_labels=labels,
            commit_immediately=True,
        )
        lock_str = " [LOCKED]" if state.recipe_locked else ""
        print(
            f"[t={t:4.1f}s] TAS={action:<8}({conf:.0%}) | YOLO={','.join(sorted(labels)):<36} | "
            f"Recipe={state.selected_recipe_name}{lock_str} ({state.recipe_confidence:.2f}) | "
            f"Probe={state.probe_temp_c:.1f}C | Progress={state.completion_ratio:.0%}"
        )
        if state.resolved_step:
            print(f"   -> Matched Step : {state.resolved_step}")
        if state.mistake_this_frame:
            print(f"   !! ORDER ERROR  : {state.mistake_this_frame}")
        if state.temp_alert:
            print(f"   !! TEMP ALERT   : {state.temp_alert}")

    print("\nSimulation Summary:")
    print(f"  Final Recipe     : {state.selected_recipe_name} (locked={state.recipe_locked})")
    print(f"  Completed Steps  : {len(state.completed_steps)} -> {[s[0] for s in state.completed_steps]}")
    print(f"  Mistakes Flagged : {len(state.detected_mistakes)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Multimodal Cooking Assistant — Unified First Prototype")
    parser.add_argument("--source", type=str, default=None, help="Video file path (.mp4/.mov) or webcam index ('0').")
    parser.add_argument("--detector", type=str, default="auto", choices=["auto", "world", "onnx"], help="YOLO backend.")
    parser.add_argument("--onnx", type=str, default="exports/kitchen_yolov5s.onnx", help="ONNX model path.")
    parser.add_argument("--tas-weights", type=str, default="TAS/Weights/mstcn_cooking.pth", help="MS-TCN weights path.")
    parser.add_argument("--recipe", type=str, default=None, help="Optional CaptainCook4D recipe ID to lock (e.g. scrambledeggs).")
    parser.add_argument("--start", type=str, default="0", help="Start timestamp (seconds, MM:SS, or HH:MM:SS).")
    parser.add_argument("--duration", type=str, default=None, help="Optional max clip duration from --start.")
    parser.add_argument("--stride", type=int, default=2, help="Read every Nth video frame (use 3-6 on 60 FPS video).")
    parser.add_argument("--conf", type=float, default=0.28, help="YOLO confidence threshold.")
    parser.add_argument("--iou", type=float, default=0.45, help="YOLO NMS IoU threshold.")
    parser.add_argument("--save-video", type=str, default=None, help="Optional .mp4 path for annotated prototype HUD.")
    parser.add_argument("--save-bbox-features", type=str, default=None, help="Optional .npy path for 5 FPS bbox features.")
    parser.add_argument("--no-display", action="store_true", help="Run headless without opening OpenCV window.")
    parser.add_argument("--probe-sim", action="store_true", help="Simulate automatic food probe heating.")
    parser.add_argument("--demo-sim", action="store_true", help="Run scripted end-to-end integration test without video.")
    args = parser.parse_args()

    pipeline = MultimodalCookingPipeline(
        detector_backend=args.detector,
        onnx_path=args.onnx,
        tas_weights=args.tas_weights,
        conf=args.conf,
        iou=args.iou,
        forced_recipe_id=args.recipe,
        temp_mode="auto" if args.probe_sim else "manual",
        detect_every_n=1,
    )
    print(f"Initialized pipeline | Detector: {pipeline.detector.backend} ({pipeline.detector.num_classes} classes) | Recipes: {len(pipeline.recipes)}")

    if args.demo_sim or args.source is None:
        run_demo_simulation(pipeline)
        return

    cap = open_capture(args.source)
    start_sec = parse_timestamp(args.start)
    duration_sec = parse_timestamp(args.duration) if args.duration is not None else None
    if start_sec > 0 and not args.source.isdigit():
        cap.set(cv2.CAP_PROP_POS_MSEC, start_sec * 1000.0)

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    stride = max(1, args.stride)
    writer = None
    window_name = "Multimodal Cooking Assistant — First Prototype"
    paused = False
    raw_idx = 0
    processed_idx = 0
    hud_frame = None
    t_wall0 = time.perf_counter()

    try:
        while True:
            if not paused:
                ok, frame = cap.read()
                if not ok:
                    break
                if raw_idx % stride != 0:
                    raw_idx += 1
                    continue
                raw_idx += 1

                if args.source.isdigit():
                    pos_sec = time.perf_counter() - t_wall0
                else:
                    pos_sec = (cap.get(cv2.CAP_PROP_POS_MSEC) or 0.0) / 1000.0
                if duration_sec is not None and (pos_sec - start_sec) > duration_sec:
                    break

                state, hud_frame = pipeline.process_frame(frame, pos_sec)
                processed_idx += 1

                if args.save_video and writer is None:
                    out_vid = Path(args.save_video)
                    if not out_vid.is_absolute():
                        out_vid = (WORKSPACE_ROOT / out_vid).resolve()
                    out_vid.parent.mkdir(parents=True, exist_ok=True)
                    h_out, w_out = hud_frame.shape[:2]
                    writer = cv2.VideoWriter(
                        str(out_vid),
                        cv2.VideoWriter_fourcc(*"mp4v"),
                        max(5.0, src_fps / stride),
                        (w_out, h_out),
                    )
                if writer is not None:
                    writer.write(hud_frame)

                if processed_idx % 10 == 0:
                    print(
                        f"[{pos_sec:6.1f}s] Action={state.tas_action:<8} ({state.tas_confidence:.0%}) | "
                        f"Recipe={state.selected_recipe_name} ({state.recipe_confidence:.2f}) | "
                        f"Progress={state.completion_ratio:.0%} | Probe={state.probe_temp_c:.1f}C"
                    )

            if not args.no_display and hud_frame is not None:
                cv2.imshow(window_name, hud_frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                if key == ord(" "):
                    paused = not paused
                elif key in (ord("+"), ord("="), ord("w")):
                    pipeline.temp_monitor.adjust(+5.0)
                elif key in (ord("-"), ord("_"), ord("s")):
                    pipeline.temp_monitor.adjust(-5.0)
                elif key == ord("r"):
                    pipeline.reset()
                    print("Pipeline state reset.")
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if not args.no_display:
            cv2.destroyAllWindows()

    if args.save_bbox_features and pipeline.bbox_sequence:
        feat_path = Path(args.save_bbox_features)
        if not feat_path.is_absolute():
            feat_path = (WORKSPACE_ROOT / feat_path).resolve()
        feat_path.parent.mkdir(parents=True, exist_ok=True)
        arr = np.stack(pipeline.bbox_sequence, axis=0)
        np.save(feat_path, arr)
        print(f"Saved 5 FPS bbox features {arr.shape} to {feat_path}")

    print(f"Finished processing {processed_idx} frames.")


if __name__ == "__main__":
    main()
