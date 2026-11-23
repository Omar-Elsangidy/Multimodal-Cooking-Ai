"""Unified object detector supporting 70+ prompt YOLO-World and Hailo-ready YOLOv5 ONNX."""

from __future__ import annotations

import sys
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, List, Optional, Set, Tuple

import cv2
import numpy as np
import onnxruntime as ort

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = WORKSPACE_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from extract_bbox_features import (  # noqa: E402
    encode_frame_features,
    postprocess_predictions,
    preprocess_frame,
    resolve_model_classes,
)
from prototype.kitchen_classes import (  # noqa: E402
    CLASS_COLORS,
    KITCHEN_18_CLASSES,
    WORLD_PROMPT_SPECS,
    get_class_names,
    get_label_min_conf,
    get_world_prompts,
    map_label_to_dag_terms,
)


@dataclass
class Detection:
    """Normalized bounding box prediction."""

    cls_id: int
    label: str
    cx: float
    cy: float
    w: float
    h: float
    conf: float


@dataclass
class DetectionResult:
    """Outputs of a single-frame object detection pass."""

    detections: List[Detection]
    dag_objects: Set[str]
    bbox_feature_vec: np.ndarray
    filtered_count: int = 0


def _box_iou(a: Detection, b: Detection) -> float:
    ax1, ay1 = a.cx - a.w / 2.0, a.cy - a.h / 2.0
    ax2, ay2 = a.cx + a.w / 2.0, a.cy + a.h / 2.0
    bx1, by1 = b.cx - b.w / 2.0, b.cy - b.h / 2.0
    bx2, by2 = b.cx + b.w / 2.0, b.cy + b.h / 2.0

    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0.0:
        return 0.0
    area_a = max(1e-6, a.w * a.h)
    area_b = max(1e-6, b.w * b.h)
    return inter / (area_a + area_b - inter)


def _center_dist(a: Detection, b: Detection) -> float:
    return float(np.hypot(a.cx - b.cx, a.cy - b.cy))


def filter_legacy_5class_confusions(detections: List[Detection]) -> Tuple[List[Detection], int]:
    """Suppress common 5-class YOLO false positives (pan-handle->knife, stove-circle->pan)."""
    pans = [d for d in detections if d.label == "pan"]
    boards = [d for d in detections if d.label == "cutting_board"]
    hands = [d for d in detections if d.label == "hand"]

    kept: List[Detection] = []
    suppressed = 0

    for det in detections:
        area = det.w * det.h
        if det.label == "knife" and pans:
            near_pan = any(
                _box_iou(det, p) > 0.03 or _center_dist(det, p) < max(p.w, p.h) * 0.72
                for p in pans
            )
            near_board = any(_box_iou(det, b) > 0.03 for b in boards)
            if near_pan and not near_board and area < 0.045 and det.conf < 0.65:
                suppressed += 1
                continue

        if det.label == "pan":
            aspect = det.w / max(det.h, 1e-4)
            is_small_circle = 0.72 <= aspect <= 1.38 and area < 0.035
            has_hand_near = any(_center_dist(det, h) < 0.25 for h in hands)
            if is_small_circle and not has_hand_near and det.conf < 0.52:
                suppressed += 1
                continue

        kept.append(det)

    return kept, suppressed


class KitchenObjectDetector:
    """Unified kitchen object detector supporting expanded YOLO-World and YOLOv5 ONNX."""

    def __init__(
        self,
        backend: str = "auto",
        onnx_path: str | Path = "exports/kitchen_yolov5s.onnx",
        world_weights: str = "models/yolov8s-worldv2.pt",
        conf: float = 0.26,
        iou: float = 0.45,
        imgsz: int = 640,
        top_k: int = 2,
        enable_confusion_filter: bool = True,
    ) -> None:
        self.conf = conf
        self.iou = iou
        self.imgsz = imgsz
        self.top_k = top_k
        self.enable_confusion_filter = enable_confusion_filter
        self.onnx_path = (
            Path(onnx_path) if Path(onnx_path).is_absolute() else (WORKSPACE_ROOT / onnx_path)
        )
        resolved_world = (
            Path(world_weights) if Path(world_weights).is_absolute() else (WORKSPACE_ROOT / world_weights)
        )
        world_arg = str(resolved_world) if resolved_world.exists() else world_weights

        self._session: Optional[ort.InferenceSession] = None
        self._input_name: str = "images"
        self._world_model = None
        self._prompts: List[str] = []
        self._recent_labels: Deque[Set[str]] = deque(maxlen=4)

        self.backend = self._init_backend(backend, world_arg)
        if self.backend == "onnx" and self._session is not None:
            self.class_names = resolve_model_classes(self._session)
            self.num_classes = len(self.class_names)
        else:
            self.num_classes = self._resolve_num_classes()
            self.class_names = get_class_names(self.num_classes)

    def _init_backend(self, requested: str, world_weights: str) -> str:
        if requested in ("auto", "world"):
            try:
                from ultralytics import YOLO

                model = YOLO(world_weights)
                self._prompts = get_world_prompts()
                model.set_classes(self._prompts)
                self._world_model = model
                return "world"
            except Exception as exc:
                if requested == "world":
                    raise RuntimeError(f"Failed to initialize YOLO-World ({world_weights}): {exc}") from exc

        if not self.onnx_path.exists():
            raise FileNotFoundError(f"ONNX model not found: {self.onnx_path}")
        self._session = ort.InferenceSession(str(self.onnx_path), providers=["CPUExecutionProvider"])
        self._input_name = self._session.get_inputs()[0].name
        return "onnx"

    def _resolve_num_classes(self) -> int:
        if self.backend == "world":
            return len(KITCHEN_18_CLASSES)
        assert self._session is not None
        out_shape = self._session.get_outputs()[0].shape
        if len(out_shape) >= 3 and isinstance(out_shape[-1], int):
            return max(1, out_shape[-1] - 5)
        return 5

    def detect(self, frame_bgr: np.ndarray) -> DetectionResult:
        """Run object detection on one BGR frame and emit boxes, DAG terms, and bbox vector."""
        if self.backend == "world":
            raw_dets = self._detect_world(frame_bgr)
            filtered_count = 0
        else:
            raw_dets = self._detect_onnx(frame_bgr)
            filtered_count = 0
            if self.enable_confusion_filter and self.num_classes == 5:
                raw_dets, filtered_count = filter_legacy_5class_confusions(raw_dets)

        # Temporal persistence filter for low-confidence detections
        current_frame_labels = {d.label for d in raw_dets}
        self._recent_labels.append(current_frame_labels)
        stable_dets: List[Detection] = []
        for det in raw_dets:
            if det.conf >= 0.30 or sum(det.label in s for s in self._recent_labels) >= 2:
                stable_dets.append(det)

        dag_objects: Set[str] = set()
        encoded_tuples: List[Tuple[int, float, float, float, float, float]] = []
        for det in stable_dets:
            dag_objects.update(map_label_to_dag_terms(det.label))
            encoded_tuples.append((det.cls_id, det.cx, det.cy, det.w, det.h, det.conf))

        bbox_vec = encode_frame_features(encoded_tuples, self.num_classes, self.top_k)
        return DetectionResult(
            detections=stable_dets,
            dag_objects=dag_objects,
            bbox_feature_vec=bbox_vec,
            filtered_count=filtered_count,
        )

    def _detect_world(self, frame_bgr: np.ndarray) -> List[Detection]:
        assert self._world_model is not None
        # Query at low threshold (0.18) with class-agnostic NMS, then apply per-class thresholds
        results = self._world_model.predict(
            source=frame_bgr,
            conf=0.18,
            iou=self.iou,
            imgsz=self.imgsz,
            agnostic_nms=True,
            verbose=False,
        )
        dets: List[Detection] = []
        if not results or results[0].boxes is None:
            return dets

        boxes = results[0].boxes
        cls_ids = boxes.cls.cpu().numpy().astype(int)
        xywhn = boxes.xywhn.cpu().numpy()
        confs = boxes.conf.cpu().numpy()

        for prompt_idx, (cx, cy, bw, bh), score in zip(cls_ids, xywhn, confs):
            if 0 <= prompt_idx < len(self._prompts):
                prompt_label = self._prompts[prompt_idx]
                min_c = get_label_min_conf(prompt_label, self.conf)
                if float(score) < min_c:
                    continue
                canonical_id = WORLD_PROMPT_SPECS[prompt_label][0]
                display_label = "hand" if prompt_label == "human hand" else prompt_label
                dets.append(
                    Detection(
                        cls_id=canonical_id,
                        label=display_label,
                        cx=float(cx),
                        cy=float(cy),
                        w=float(bw),
                        h=float(bh),
                        conf=float(score),
                    )
                )
        return dets

    def _detect_onnx(self, frame_bgr: np.ndarray) -> List[Detection]:
        assert self._session is not None
        tensor, meta = preprocess_frame(frame_bgr, self.imgsz)
        raw_out = self._session.run(None, {self._input_name: tensor})[0]
        raw_tuples = postprocess_predictions(raw_out, meta, self.conf, self.iou)
        dets: List[Detection] = []
        for cls_id, cx, cy, bw, bh, score in raw_tuples:
            label = self.class_names[cls_id] if cls_id < len(self.class_names) else f"class_{cls_id}"
            dets.append(Detection(cls_id=cls_id, label=label, cx=cx, cy=cy, w=bw, h=bh, conf=score))
        return dets

    def draw_boxes(self, frame_bgr: np.ndarray, detections: List[Detection]) -> np.ndarray:
        """Render labeled bounding boxes onto a copy of `frame_bgr`."""
        vis = frame_bgr.copy()
        h_img, w_img = vis.shape[:2]
        for det in detections:
            x1 = max(0, int((det.cx - det.w / 2.0) * w_img))
            y1 = max(0, int((det.cy - det.h / 2.0) * h_img))
            x2 = min(w_img - 1, int((det.cx + det.w / 2.0) * w_img))
            y2 = min(h_img - 1, int((det.cy + det.h / 2.0) * h_img))
            color = CLASS_COLORS[det.cls_id % len(CLASS_COLORS)]
            text = f"{det.label} {det.conf:.2f}"
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                vis,
                text,
                (x1, max(18, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.50,
                color,
                2,
                cv2.LINE_AA,
            )
        return vis
