"""Run YOLOv5s ONNX inference to extract (T, C * top_k * 5) bounding-box features for MS-TCN.

Supports an image file, a directory of frames, or a video (.mp4/.avi/.mov) as `--source`.
Automatically reads class names from ONNX metadata (or falls back to 5 / 18 / 30-class
schemas by output tensor width) so it stays compatible across model iterations.
"""

from __future__ import annotations

import argparse
import ast
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

LEGACY_5_CLASSES = ("hand", "pan", "knife", "cutting_board", "ingredients")
KITCHEN_18_CLASSES = (
    "hand", "pan", "pot", "stove", "knife", "spatula", "spoon", "cutting_board",
    "bowl", "plate", "cup", "bottle", "appliance", "egg", "vegetable", "bread",
    "meat", "ingredients",
)
KITCHEN_30_CLASSES = (
    "hand", "pan", "pot", "lid", "hob", "tap", "sink", "knife", "spatula", "spoon",
    "fork", "cutting_board", "bowl", "plate", "cup", "bottle", "jar", "appliance",
    "onion", "tomato", "potato", "carrot", "garlic", "pepper", "egg", "cheese",
    "bread", "meat", "vegetable", "food",
)
CLASS_NAMES = KITCHEN_30_CLASSES
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def class_color(cls_id: int) -> tuple[int, int, int]:
    """Deterministic bright BGR color for any class index."""
    return (
        int((37 * cls_id + 80) % 200 + 55),
        int((97 * cls_id + 140) % 200 + 55),
        int((157 * cls_id + 40) % 200 + 55),
    )


def resolve_model_classes(session: ort.InferenceSession) -> tuple[str, ...]:
    """Read class names from ONNX metadata, falling back to known schema sizes."""
    meta_map = session.get_modelmeta().custom_metadata_map or {}
    raw_names = meta_map.get("names")
    if raw_names:
        try:
            parsed = ast.literal_eval(raw_names)
            if isinstance(parsed, dict) and parsed:
                return tuple(str(parsed[k]) for k in sorted(parsed))
            if isinstance(parsed, (list, tuple)) and parsed:
                return tuple(str(x) for x in parsed)
        except (ValueError, SyntaxError):
            pass

    out_shape = session.get_outputs()[0].shape
    num_cls = int(out_shape[-1]) - 5 if len(out_shape) >= 3 and isinstance(out_shape[-1], int) else 30
    for schema in (LEGACY_5_CLASSES, KITCHEN_18_CLASSES, KITCHEN_30_CLASSES):
        if num_cls == len(schema):
            return schema
    return tuple(f"class_{i}" for i in range(max(1, num_cls)))


def preprocess_frame(
    image_bgr: np.ndarray, imgsz: int
) -> tuple[np.ndarray, tuple[float, float, float, int, int]]:
    """Letterbox BGR uint8 image to square `imgsz` and return `(tensor, meta)`.

    `meta` stores `(scale, pad_x, pad_y, orig_w, orig_h)` so predicted boxes can
    be mapped cleanly back onto the original image aspect ratio.
    """
    orig_h, orig_w = image_bgr.shape[:2]
    scale = min(imgsz / orig_w, imgsz / orig_h)
    new_w = int(round(orig_w * scale))
    new_h = int(round(orig_h * scale))
    pad_x = (imgsz - new_w) / 2.0
    pad_y = (imgsz - new_h) / 2.0

    resized = cv2.resize(image_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((imgsz, imgsz, 3), 114, dtype=np.uint8)
    top, left = int(round(pad_y - 0.1)), int(round(pad_x - 0.1))
    canvas[top : top + new_h, left : left + new_w] = resized

    rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    chw = np.transpose(rgb, (2, 0, 1)).astype(np.float32) / 255.0
    return np.expand_dims(chw, axis=0), (scale, float(left), float(top), orig_w, orig_h)


def postprocess_predictions(
    raw_output: np.ndarray,
    letterbox_meta: tuple[float, float, float, int, int],
    conf_thresh: float,
    iou_thresh: float,
) -> list[tuple[int, float, float, float, float, float]]:
    """Decode YOLOv5 ONNX output [1, N, 5 + C], apply NMS, and un-letterbox coords.

    Returns list of (class_id, cx_norm, cy_norm, w_norm, h_norm, score) in [0, 1].
    """
    preds = raw_output[0]
    obj_conf = preds[:, 4]
    cls_scores = preds[:, 5:]
    cls_ids = np.argmax(cls_scores, axis=1)
    cls_conf = cls_scores[np.arange(len(preds)), cls_ids]
    scores = obj_conf * cls_conf

    mask = scores >= conf_thresh
    if not np.any(mask):
        return []

    filtered = preds[mask]
    f_scores = scores[mask]
    f_cls = cls_ids[mask]

    boxes_xywh = np.column_stack(
        [
            filtered[:, 0] - filtered[:, 2] / 2.0,
            filtered[:, 1] - filtered[:, 3] / 2.0,
            filtered[:, 2],
            filtered[:, 3],
        ]
    ).tolist()

    indices = cv2.dnn.NMSBoxes(boxes_xywh, f_scores.tolist(), conf_thresh, iou_thresh)
    if len(indices) == 0:
        return []

    scale, pad_x, pad_y, orig_w, orig_h = letterbox_meta
    detections: list[tuple[int, float, float, float, float, float]] = []
    for idx in np.asarray(indices).flatten():
        cx_px, cy_px, bw_px, bh_px = filtered[idx, :4]
        cx = ((cx_px - pad_x) / scale) / float(orig_w)
        cy = ((cy_px - pad_y) / scale) / float(orig_h)
        w = (bw_px / scale) / float(orig_w)
        h = (bh_px / scale) / float(orig_h)
        detections.append(
            (
                int(f_cls[idx]),
                float(np.clip(cx, 0.0, 1.0)),
                float(np.clip(cy, 0.0, 1.0)),
                float(np.clip(w, 0.0, 1.0)),
                float(np.clip(h, 0.0, 1.0)),
                float(f_scores[idx]),
            )
        )
    return detections


def encode_frame_features(
    detections: list[tuple[int, float, float, float, float, float]],
    num_classes: int,
    top_k: int,
) -> np.ndarray:
    """Encode detections into a fixed (num_classes * top_k * 5,) float32 feature vector."""
    feat = np.zeros((num_classes, top_k, 5), dtype=np.float32)
    per_class: list[list[tuple[float, float, float, float, float]]] = [[] for _ in range(num_classes)]
    for cls_id, cx, cy, w, h, score in detections:
        if 0 <= cls_id < num_classes:
            per_class[cls_id].append((cx, cy, w, h, score))

    for cls_id, items in enumerate(per_class):
        items.sort(key=lambda item: item[4], reverse=True)
        for slot, vals in enumerate(items[:top_k]):
            feat[cls_id, slot] = vals
    return feat.reshape(-1)


def draw_detections(
    image_bgr: np.ndarray,
    detections: list[tuple[int, float, float, float, float, float]],
    class_names: tuple[str, ...] = CLASS_NAMES,
) -> np.ndarray:
    """Draw predicted bounding boxes and labels onto a copy of `image_bgr`."""
    vis = image_bgr.copy()
    h_img, w_img = vis.shape[:2]
    for cls_id, cx, cy, bw, bh, score in detections:
        x1 = max(0, int((cx - bw / 2.0) * w_img))
        y1 = max(0, int((cy - bh / 2.0) * h_img))
        x2 = min(w_img - 1, int((cx + bw / 2.0) * w_img))
        y2 = min(h_img - 1, int((cy + bh / 2.0) * h_img))
        color = class_color(cls_id)
        name = class_names[cls_id] if cls_id < len(class_names) else f"class_{cls_id}"
        label = f"{name} {score:.2f}"
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        cv2.putText(vis, label, (x1, max(18, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
    return vis


def iter_frames(source: Path):
    """Yield `(frame_name, bgr_image)` pairs from an image, directory, or video."""
    if source.is_file() and source.suffix.lower() in IMAGE_EXTS:
        img = cv2.imread(str(source))
        if img is not None:
            yield source.name, img
        return

    if source.is_file() and source.suffix.lower() in VIDEO_EXTS:
        cap = cv2.VideoCapture(str(source))
        idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            yield f"{source.stem}_f{idx:06d}.jpg", frame
            idx += 1
        cap.release()
        return

    if source.is_dir():
        for img_path in sorted(p for p in source.iterdir() if p.suffix.lower() in IMAGE_EXTS):
            img = cv2.imread(str(img_path))
            if img is not None:
                yield img_path.name, img
        return

    raise FileNotFoundError(f"Unsupported or missing source: {source}")


def resolve_path(workspace: Path, raw_path: str) -> Path:
    """Resolve `raw_path` relative to workspace unless already absolute."""
    p = Path(raw_path)
    return p if p.is_absolute() else (workspace / p).resolve()


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract YOLOv5 ONNX coordinate features for MS-TCN.")
    parser.add_argument("--source", type=str, required=True, help="Image, image folder, or video file.")
    parser.add_argument("--onnx", type=str, default="exports/kitchen_yolov5s.onnx", help="ONNX model path.")
    parser.add_argument("--output", type=str, default="exports/bbox_features.npy", help="Output .npy path.")
    parser.add_argument("--save-vis", type=str, default=None, help="Optional folder to save annotated frames.")
    parser.add_argument("--imgsz", type=int, default=640, help="ONNX static input resolution.")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold.")
    parser.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold.")
    parser.add_argument("--top-k", type=int, default=2, help="Top detections kept per class.")
    args = parser.parse_args()

    workspace = Path(__file__).resolve().parent.parent
    source_path = resolve_path(workspace, args.source)
    onnx_path = resolve_path(workspace, args.onnx)
    out_path = resolve_path(workspace, args.output)
    vis_dir = resolve_path(workspace, args.save_vis) if args.save_vis else None
    if vis_dir:
        vis_dir.mkdir(parents=True, exist_ok=True)

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    active_classes = resolve_model_classes(session)

    frame_vectors = []
    for name, img in iter_frames(source_path):
        tensor, meta = preprocess_frame(img, args.imgsz)
        raw_out = session.run(None, {input_name: tensor})[0]
        dets = postprocess_predictions(raw_out, meta, args.conf, args.iou)
        vec = encode_frame_features(dets, len(active_classes), args.top_k)
        frame_vectors.append(vec)
        if vis_dir:
            cv2.imwrite(str(vis_dir / name), draw_detections(img, dets, active_classes))
        print(f"{name}: {len(dets)} detections")

    if not frame_vectors:
        raise RuntimeError(f"No readable frames found at {source_path}")

    sequence = np.stack(frame_vectors, axis=0)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, sequence)
    print(f"Saved MS-TCN input tensor {sequence.shape} ({sequence.dtype}) to {out_path}")


if __name__ == "__main__":
    main()
