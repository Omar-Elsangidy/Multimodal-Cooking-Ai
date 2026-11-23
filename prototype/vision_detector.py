"""Standalone kitchen object detector supporting YOLOv5s ONNX and optional YOLO-World."""

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
    class_color,
    encode_frame_features,
    postprocess_predictions,
    preprocess_frame,
    resolve_model_classes,
)
from prototype.kitchen_classes import (  # noqa: E402
    KITCHEN_30_CLASSES,
    WORLD_PROMPT_TO_ID,
    get_class_names,
    get_label_min_conf,
    get_world_prompts,
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
    bbox_feature_vec: np.ndarray


class KitchenObjectDetector:
    """Kitchen object detector supporting trained YOLOv5s ONNX and optional YOLO-World."""

    def __init__(
        self,
        backend: str = "onnx",
        onnx_path: str | Path = "exports/kitchen_yolov5s.onnx",
        world_weights: str = "models/yolov8s-worldv2.pt",
        conf: float = 0.25,
        iou: float = 0.45,
        imgsz: int = 640,
        top_k: int = 2,
    ) -> None:
        self.conf = conf
        self.iou = iou
        self.imgsz = imgsz
        self.top_k = top_k
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
            self.num_classes = len(KITCHEN_30_CLASSES)
            self.class_names = get_class_names(self.num_classes)

    def _init_backend(self, requested: str, world_weights: str) -> str:
        if requested == "world":
            from ultralytics import YOLO

            model = YOLO(world_weights)
            self._prompts = get_world_prompts()
            model.set_classes(self._prompts)
            self._world_model = model
            return "world"

        if self.onnx_path.exists():
            self._session = ort.InferenceSession(str(self.onnx_path), providers=["CPUExecutionProvider"])
            self._input_name = self._session.get_inputs()[0].name
            return "onnx"

        if requested == "auto":
            from ultralytics import YOLO

            model = YOLO(world_weights)
            self._prompts = get_world_prompts()
            model.set_classes(self._prompts)
            self._world_model = model
            return "world"

        raise FileNotFoundError(f"ONNX model not found: {self.onnx_path}")

    def detect(self, frame_bgr: np.ndarray) -> DetectionResult:
        """Run object detection on one BGR frame and emit boxes and bbox vector."""
        raw_dets = self._detect_world(frame_bgr) if self.backend == "world" else self._detect_onnx(frame_bgr)

        # Temporal persistence filter suppresses isolated single-frame flicker
        self._recent_labels.append({d.label for d in raw_dets})
        stable_dets = [
            d for d in raw_dets
            if d.conf >= max(self.conf, 0.28) or sum(d.label in s for s in self._recent_labels) >= 2
        ]

        encoded_tuples = [(d.cls_id, d.cx, d.cy, d.w, d.h, d.conf) for d in stable_dets]
        bbox_vec = encode_frame_features(encoded_tuples, self.num_classes, self.top_k)
        return DetectionResult(detections=stable_dets, bbox_feature_vec=bbox_vec)

    def _detect_world(self, frame_bgr: np.ndarray) -> List[Detection]:
        assert self._world_model is not None
        results = self._world_model.predict(
            source=frame_bgr,
            conf=min(0.18, self.conf),
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
                if float(score) < get_label_min_conf(prompt_label, self.conf):
                    continue
                canonical_id = WORLD_PROMPT_TO_ID[prompt_label]
                dets.append(
                    Detection(
                        cls_id=canonical_id,
                        label=prompt_label,
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
        return [
            Detection(
                cls_id=cls_id,
                label=self.class_names[cls_id] if cls_id < len(self.class_names) else f"class_{cls_id}",
                cx=cx,
                cy=cy,
                w=bw,
                h=bh,
                conf=score,
            )
            for cls_id, cx, cy, bw, bh, score in raw_tuples
        ]

    def draw_boxes(self, frame_bgr: np.ndarray, detections: List[Detection]) -> np.ndarray:
        """Render labeled bounding boxes onto a copy of `frame_bgr`."""
        vis = frame_bgr.copy()
        h_img, w_img = vis.shape[:2]
        for det in detections:
            x1 = max(0, int((det.cx - det.w / 2.0) * w_img))
            y1 = max(0, int((det.cy - det.h / 2.0) * h_img))
            x2 = min(w_img - 1, int((det.cx + det.w / 2.0) * w_img))
            y2 = min(h_img - 1, int((det.cy + det.h / 2.0) * h_img))
            color = class_color(det.cls_id)
            text = f"{det.label} {det.conf:.2f}"
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                vis,
                text,
                (x1, max(18, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                2,
                cv2.LINE_AA,
            )
        return vis
