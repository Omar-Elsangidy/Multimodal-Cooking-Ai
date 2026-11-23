"""Unified Multimodal Cooking Assistant pipeline connecting YOLO, TAS MS-TCN, Probe Temp, and CaptainCook4D DAG."""

from __future__ import annotations

import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import cv2
import numpy as np

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
CC4D_SRC = WORKSPACE_ROOT / "captaincook4d_project" / "src"
if str(CC4D_SRC) not in sys.path:
    sys.path.insert(0, str(CC4D_SRC))

from dag_loader import CaptainCookDAGLoader, RecipeDAGState  # noqa: E402
from recipe_engine import CaptainCookRecipeEngine  # noqa: E402

from prototype.action_recognizer import ACTION_COLORS, ActionPrediction, TASActionRecognizer  # noqa: E402
from prototype.action_step_bridge import ActionToStepBridge  # noqa: E402
from prototype.temperature_monitor import FoodProbeMonitor, TempStatus  # noqa: E402
from prototype.vision_detector import DetectionResult, KitchenObjectDetector  # noqa: E402

ANNOT_ROOT = WORKSPACE_ROOT / "captaincook4d_project" / "annotations-main"
EXTRA_RECIPES_DIR = WORKSPACE_ROOT / "prototype" / "extra_recipes"


@dataclass
class PrototypeState:
    """Complete snapshot of all 4 modules at the current frame."""

    timestamp_sec: float
    detector_backend: str
    detected_labels: List[str]
    dag_objects: List[str]
    filtered_fp_count: int
    tas_action: str
    tas_detail: str
    tas_confidence: float
    tas_probs: Dict[str, float]
    tas_timeline: List[Tuple[float, str, float]]
    resolved_step: Optional[str]
    probe_temp_c: float
    probe_target_c: Optional[float]
    probe_mode: str
    probe_label: str
    probe_gate_passed: bool
    temp_alert: Optional[str]
    selected_recipe_id: str
    selected_recipe_name: str
    recipe_confidence: float
    recipe_locked: bool
    top_k_candidates: List[Tuple[str, float]]
    completed_steps: List[Tuple[int, str]]
    next_valid_steps: List[Tuple[int, str]]
    detected_mistakes: List[dict]
    mistake_this_frame: Optional[str]
    completion_ratio: float
    is_finished: bool

    def to_dict(self) -> Dict[str, Any]:
        """Serialize snapshot to JSON-friendly dict for web dashboard / API consumers."""
        return {
            "timestamp_sec": round(self.timestamp_sec, 2),
            "detector_backend": self.detector_backend,
            "detected_labels": self.detected_labels,
            "dag_objects": self.dag_objects,
            "filtered_fp_count": self.filtered_fp_count,
            "tas_action": self.tas_action,
            "tas_detail": self.tas_detail,
            "tas_confidence": round(self.tas_confidence, 3),
            "tas_probs": self.tas_probs,
            "tas_timeline": [[round(t, 1), a, round(c, 2)] for t, a, c in self.tas_timeline[-24:]],
            "resolved_step": self.resolved_step,
            "probe_temp_c": self.probe_temp_c,
            "probe_target_c": self.probe_target_c,
            "probe_mode": self.probe_mode,
            "probe_label": self.probe_label,
            "probe_gate_passed": self.probe_gate_passed,
            "temp_alert": self.temp_alert,
            "selected_recipe_id": self.selected_recipe_id,
            "selected_recipe_name": self.selected_recipe_name,
            "recipe_confidence": round(self.recipe_confidence, 3),
            "recipe_locked": self.recipe_locked,
            "top_k_candidates": [[rid, round(s, 3)] for rid, s in self.top_k_candidates],
            "completed_steps": [[sid, desc] for sid, desc in self.completed_steps],
            "next_valid_steps": [[sid, desc] for sid, desc in self.next_valid_steps[:5]],
            "detected_mistakes": [
                {"type": m.get("type", "Order Error"), "step_id": m.get("step_id"), "step_desc": m.get("step_desc")}
                for m in self.detected_mistakes
            ],
            "mistake_this_frame": self.mistake_this_frame,
            "completion_ratio": round(self.completion_ratio, 3),
            "is_finished": self.is_finished,
        }


def _load_all_recipes() -> Dict[str, RecipeDAGState]:
    recipes = CaptainCookDAGLoader.load_all(
        ANNOT_ROOT / "task_graphs",
        ANNOT_ROOT / "annotation_json" / "complete_step_annotations.json",
    )
    if EXTRA_RECIPES_DIR.exists():
        for json_path in sorted(EXTRA_RECIPES_DIR.glob("*.json")):
            dag = CaptainCookDAGLoader.load_one(json_path)
            if json_path.stem == "pansearedmeat":
                dag.recipe_name = "Pan Seared Meat"
                dag.expected_objects.update({"meat", "pan", "skillet", "oil", "plate", "bowl", "spoon"})
            recipes[json_path.stem] = dag
    return recipes


class MultimodalCookingPipeline:
    """End-to-end orchestrator: YOLO -> TAS MS-TCN + Probe Temp -> CaptainCook4D DAG."""

    def __init__(
        self,
        detector_backend: str = "auto",
        onnx_path: str | Path = "exports/kitchen_yolov5s.onnx",
        tas_weights: str | Path = "TAS/Weights/mstcn_cooking.pth",
        conf: float = 0.28,
        iou: float = 0.45,
        forced_recipe_id: Optional[str] = None,
        initial_temp_c: float = 22.0,
        temp_mode: str = "manual",
        detect_every_n: int = 2,
    ) -> None:
        self.recipes: Dict[str, RecipeDAGState] = _load_all_recipes()
        self.engine = CaptainCookRecipeEngine(self.recipes)
        self.detector = KitchenObjectDetector(
            backend=detector_backend,
            onnx_path=onnx_path,
            conf=conf,
            iou=iou,
        )
        self.recognizer = TASActionRecognizer(weights_path=tas_weights)
        self.temp_monitor = FoodProbeMonitor(initial_temp_c=initial_temp_c, mode=temp_mode)
        self.bridge = ActionToStepBridge()

        self.detect_every_n = max(1, detect_every_n)
        self.forced_recipe_id: Optional[str] = None
        self.set_recipe(forced_recipe_id)

        self._frame_counter = 0
        self._last_timestamp: Optional[float] = None
        self._object_seen_counts: Counter[str] = Counter()
        self._last_det = DetectionResult(detections=[], dag_objects=set(), bbox_feature_vec=np.zeros(1))
        self._last_action = ActionPrediction(
            action="Idle",
            confidence=1.0,
            probs={"Idle": 1.0, "Chopping": 0.0, "Stirring": 0.0, "Adding": 0.0},
            buffer_steps=0,
            sampled_new_step=False,
        )
        self._last_resolved_step: Optional[str] = None
        self._last_engine_res: Dict[str, Any] = self.engine.update(yolo_objects=None, predicted_action=None)
        self._last_mistake_banner: Optional[str] = None
        self._last_mistake_time: float = -999.0
        self.bbox_sequence: List[np.ndarray] = []

    def set_recipe(self, recipe_id: Optional[str]) -> None:
        """Pin a specific recipe ID or pass None for automatic selection."""
        if recipe_id and recipe_id not in self.recipes:
            raise ValueError(f"Unknown recipe '{recipe_id}'. Available: {sorted(self.recipes.keys())}")
        self.forced_recipe_id = recipe_id
        self.engine._locked_recipe_id = recipe_id

    def reset(self) -> None:
        """Reset all pipeline buffers and recipe DAG progress."""
        self.engine.reset()
        if self.forced_recipe_id:
            self.engine._locked_recipe_id = self.forced_recipe_id
        self.recognizer.reset()
        self.temp_monitor.reset()
        self.bridge.reset()
        self._frame_counter = 0
        self._last_timestamp = None
        self._object_seen_counts.clear()
        self._last_resolved_step = None
        self._last_mistake_banner = None
        self._last_mistake_time = -999.0
        self.bbox_sequence.clear()
        self._last_engine_res = self.engine.update(yolo_objects=None, predicted_action=None)

    def _persistent_objects(self, dag_objects: Set[str]) -> Set[str]:
        """Commit a YOLO DAG object only after >= 2 observations so 1 stray detection never sticks."""
        for obj in dag_objects:
            self._object_seen_counts[obj.lower().strip()] += 1
        return {obj for obj, cnt in self._object_seen_counts.items() if cnt >= 2}

    def step_signals(
        self,
        timestamp_sec: float,
        dag_objects: Set[str],
        tas_action: str,
        tas_conf: float,
        tas_probs: Dict[str, float],
        detected_labels: List[str],
        filtered_fp_count: int = 0,
        commit_immediately: bool = False,
        tas_detail: Optional[str] = None,
    ) -> PrototypeState:
        """Run Fusion + Bridge + Probe Safety Gate + DAG update."""
        dt = 0.2 if self._last_timestamp is None else max(0.0, timestamp_sec - self._last_timestamp)
        self._last_timestamp = timestamp_sec

        if commit_immediately:
            committed_objs = {o.lower().strip() for o in dag_objects}
        else:
            committed_objs = self._persistent_objects(dag_objects)
        if committed_objs:
            self.engine.observed_objects.update(committed_objs)

        if self.forced_recipe_id:
            candidate_dags = [self.recipes[self.forced_recipe_id]]
        elif self.engine._locked_recipe_id:
            candidate_dags = [self.recipes[self.engine._locked_recipe_id]]
        else:
            ranked = self.engine._rank()[:5]
            candidate_dags = [self.recipes[rid] for rid, _ in ranked]

        resolved_step = self.bridge.resolve_step(
            tas_action=tas_action,
            tas_conf=tas_conf,
            candidate_dags=candidate_dags,
            current_objects=dag_objects,
            cumulative_objects=self.engine.observed_objects | dag_objects,
            timestamp_sec=timestamp_sec,
        )

        # Evaluate Food Probe safety/doneness gate against the active recipe & step
        prelim_rid = self.forced_recipe_id or self._last_engine_res["selected_recipe_id"]
        next_steps_prev = self._last_engine_res["next_valid_steps"]
        active_step_text = resolved_step or (next_steps_prev[0][1] if next_steps_prev else "")
        temp_status: TempStatus = self.temp_monitor.update(
            dt_sec=dt,
            recipe_id=prelim_rid,
            active_objects=dag_objects,
            tas_action=tas_action,
            active_step_text=active_step_text,
        )

        # If probe has reached target temp and a ready 'check temperature' step exists, auto-satisfy it
        if resolved_step is None and temp_status.gate_passed and temp_status.target_c is not None:
            for sid, desc in next_steps_prev:
                if "temperature" in desc.lower() and "check" in desc.lower():
                    resolved_step = desc
                    break

        if resolved_step is not None:
            self._last_resolved_step = resolved_step

        self._last_engine_res = self.engine.update(
            yolo_objects=committed_objs or None,
            predicted_action=resolved_step,
            action_confidence=max(tas_conf, 0.70) if resolved_step else 1.0,
        )
        if self.forced_recipe_id:
            self.engine._locked_recipe_id = self.forced_recipe_id

        if self._last_engine_res["mistake_this_frame"]:
            self._last_mistake_banner = self._last_engine_res["mistake_this_frame"]
            self._last_mistake_time = timestamp_sec
        elif (timestamp_sec - self._last_mistake_time) > 6.0:
            self._last_mistake_banner = None

        sel_rid = self._last_engine_res["selected_recipe_id"]
        sel_dag = self.recipes[sel_rid]
        completed_steps = [
            (sid, sel_dag.steps[sid])
            for sid in self._last_engine_res["completed_step_ids"]
            if sid in sel_dag.steps
        ]

        return PrototypeState(
            timestamp_sec=timestamp_sec,
            detector_backend=self.detector.backend,
            detected_labels=sorted(set(detected_labels)),
            dag_objects=sorted(dag_objects),
            filtered_fp_count=filtered_fp_count,
            tas_action=tas_action,
            tas_detail=tas_detail or tas_action,
            tas_confidence=tas_conf,
            tas_probs=tas_probs,
            tas_timeline=list(self.recognizer.timeline),
            resolved_step=self._last_resolved_step,
            probe_temp_c=temp_status.temp_c,
            probe_target_c=temp_status.target_c,
            probe_mode=temp_status.mode,
            probe_label=temp_status.label,
            probe_gate_passed=temp_status.gate_passed,
            temp_alert=temp_status.alert,
            selected_recipe_id=sel_rid,
            selected_recipe_name=self._last_engine_res["selected_recipe_name"],
            recipe_confidence=self._last_engine_res["confidence"],
            recipe_locked=self._last_engine_res["locked"],
            top_k_candidates=self._last_engine_res["top_k_candidates"],
            completed_steps=completed_steps,
            next_valid_steps=self._last_engine_res["next_valid_steps"],
            detected_mistakes=self._last_engine_res["detected_mistakes"],
            mistake_this_frame=self._last_mistake_banner,
            completion_ratio=self._last_engine_res["completion_ratio"],
            is_finished=self._last_engine_res["is_finished"],
        )

    def process_frame(self, frame_bgr: np.ndarray, timestamp_sec: float) -> Tuple[PrototypeState, np.ndarray]:
        """Run YOLO + TAS + Probe + DAG on one video frame and return `(state, hud_frame)`."""
        if self._frame_counter % self.detect_every_n == 0:
            self._last_det = self.detector.detect(frame_bgr)
        self._frame_counter += 1

        self._last_action = self.recognizer.step(
            frame_bgr,
            timestamp_sec,
            detections=self._last_det.detections,
        )
        if self._last_action.sampled_new_step:
            self.bbox_sequence.append(self._last_det.bbox_feature_vec)

        if self._last_action.sampled_new_step or self._last_timestamp is None:
            state = self.step_signals(
                timestamp_sec=timestamp_sec,
                dag_objects=self._last_det.dag_objects,
                tas_action=self._last_action.action,
                tas_conf=self._last_action.confidence,
                tas_probs=self._last_action.probs,
                detected_labels=[d.label for d in self._last_det.detections],
                filtered_fp_count=self._last_det.filtered_count,
                tas_detail=self._last_action.action_detail,
            )
            self._cached_state = state
        else:
            state = self._cached_state

        annotated = self.detector.draw_boxes(frame_bgr, self._last_det.detections)
        hud_frame = render_prototype_hud(annotated, state)
        return state, hud_frame


def _wrap_text(text: str, max_chars: int = 44) -> List[str]:
    words = text.split()
    lines: List[str] = []
    cur = ""
    for word in words:
        if len(cur) + len(word) + 1 <= max_chars:
            cur = f"{cur} {word}".strip()
        else:
            if cur:
                lines.append(cur)
            cur = word
    if cur:
        lines.append(cur)
    return lines or [""]


def render_prototype_hud(video_bgr: np.ndarray, state: PrototypeState) -> np.ndarray:
    """Compose a split-screen dashboard: annotated video left + live telemetry panel right."""
    target_h = 640
    h, w = video_bgr.shape[:2]
    scale = target_h / max(1, h)
    video_resized = cv2.resize(video_bgr, (int(w * scale), target_h))

    panel_w = 460
    panel = np.full((target_h, panel_w, 3), (24, 26, 32), dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX

    y = 28
    cv2.putText(panel, "MULTIMODAL COOKING ASSISTANT", (16, y), font, 0.58, (0, 210, 255), 2, cv2.LINE_AA)
    y += 22
    mins, secs = divmod(int(state.timestamp_sec), 60)
    meta = f"Time {mins:02d}:{secs:02d} | Detector: {state.detector_backend.upper()}"
    if state.filtered_fp_count:
        meta += f" (-{state.filtered_fp_count} fp)"
    cv2.putText(panel, meta, (16, y), font, 0.44, (170, 175, 185), 1, cv2.LINE_AA)
    y += 14
    cv2.line(panel, (16, y), (panel_w - 16, y), (60, 65, 75), 1)
    y += 24

    lock_tag = " [LOCKED]" if state.recipe_locked else " [CANDIDATE]"
    cv2.putText(
        panel,
        f"RECIPE: {state.selected_recipe_name}{lock_tag}",
        (16, y),
        font,
        0.52,
        (90, 240, 140) if state.recipe_locked else (255, 220, 100),
        1,
        cv2.LINE_AA,
    )
    y += 20
    bar_w = panel_w - 32
    cv2.rectangle(panel, (16, y), (16 + bar_w, y + 14), (50, 54, 64), -1)
    fill_w = int(bar_w * min(1.0, max(0.0, state.completion_ratio)))
    cv2.rectangle(panel, (16, y), (16 + fill_w, y + 14), (70, 210, 120), -1)
    cv2.putText(
        panel,
        f"Progress: {state.completion_ratio:.0%} ({len(state.completed_steps)} steps)",
        (22, y + 11),
        font,
        0.38,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    y += 28

    cand_txt = " | ".join(f"{rid[:13]}:{s:.2f}" for rid, s in state.top_k_candidates[:3])
    cv2.putText(panel, f"Top-3: {cand_txt}", (16, y), font, 0.39, (185, 190, 200), 1, cv2.LINE_AA)
    y += 24

    act_color = ACTION_COLORS.get(state.tas_action, (200, 200, 200))
    cv2.rectangle(panel, (14, y - 16), (panel_w - 14, y + 28), (36, 40, 50), -1)
    cv2.putText(
        panel,
        f"ACTION: {state.tas_detail[:31].upper()} ({state.tas_confidence:.0%})",
        (22, y + 4),
        font,
        0.50,
        act_color,
        2,
        cv2.LINE_AA,
    )
    prob_line = "  ".join(f"{k[:4]}:{v:.0%}" for k, v in state.tas_probs.items())
    cv2.putText(panel, prob_line, (22, y + 21), font, 0.37, (180, 185, 195), 1, cv2.LINE_AA)
    y += 42

    objs = ", ".join(state.detected_labels[:6]) if state.detected_labels else "none"
    cv2.putText(panel, f"YOLO Objects: {objs[:44]}", (16, y), font, 0.43, (235, 235, 235), 1, cv2.LINE_AA)
    y += 22

    gate_str = (
        f"SAFE >= {state.probe_target_c:.0f}C"
        if state.probe_gate_passed and state.probe_target_c
        else (f"WAIT < {state.probe_target_c:.0f}C" if state.probe_target_c else state.probe_label)
    )
    temp_color = (90, 235, 140) if (state.probe_gate_passed and state.probe_target_c) else (80, 190, 255)
    if state.temp_alert:
        temp_color = (70, 90, 255)
    cv2.putText(
        panel,
        f"Food Probe: {state.probe_temp_c:.1f}C [{gate_str}]",
        (16, y),
        font,
        0.44,
        temp_color,
        1,
        cv2.LINE_AA,
    )
    y += 24

    if state.mistake_this_frame:
        cv2.rectangle(panel, (14, y - 14), (panel_w - 14, y + 36), (35, 35, 130), -1)
        for line in _wrap_text(f"ORDER ERROR: {state.mistake_this_frame}", 48)[:2]:
            cv2.putText(panel, line, (20, y), font, 0.40, (120, 160, 255), 1, cv2.LINE_AA)
            y += 17
        y += 12
    elif state.temp_alert:
        cv2.rectangle(panel, (14, y - 14), (panel_w - 14, y + 24), (35, 70, 130), -1)
        cv2.putText(panel, state.temp_alert[:52], (20, y + 4), font, 0.39, (120, 220, 255), 1, cv2.LINE_AA)
        y += 30

    cv2.line(panel, (16, y), (panel_w - 16, y), (60, 65, 75), 1)
    y += 20
    cv2.putText(panel, "NEXT RECOMMENDED STEPS (DAG):", (16, y), font, 0.45, (140, 220, 255), 1, cv2.LINE_AA)
    y += 20
    if state.is_finished:
        cv2.putText(panel, "Recipe Completed!", (16, y), font, 0.55, (80, 240, 120), 2, cv2.LINE_AA)
        y += 24
    else:
        for sid, desc in state.next_valid_steps[:3]:
            for idx, line in enumerate(_wrap_text(f"[{sid}] {desc}", 50)[:2]):
                prefix_color = (225, 230, 235) if idx == 0 else (180, 185, 195)
                cv2.putText(panel, line, (18, y), font, 0.40, prefix_color, 1, cv2.LINE_AA)
                y += 16
            y += 4

    if state.completed_steps and y < target_h - 65:
        y += 6
        cv2.putText(panel, "COMPLETED:", (16, y), font, 0.43, (120, 210, 130), 1, cv2.LINE_AA)
        y += 18
        for sid, desc in state.completed_steps[-3:]:
            if y > target_h - 26:
                break
            cv2.putText(panel, f"[ok {sid}] {desc[:44]}", (18, y), font, 0.38, (145, 205, 155), 1, cv2.LINE_AA)
            y += 16

    cv2.putText(
        panel,
        "[SPACE] Pause  [+/-] Temp  [R] Reset  [Q] Quit",
        (16, target_h - 12),
        font,
        0.38,
        (135, 140, 150),
        1,
        cv2.LINE_AA,
    )
    return np.hstack([video_resized, panel])
