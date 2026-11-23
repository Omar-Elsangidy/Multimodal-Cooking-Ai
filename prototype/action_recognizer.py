"""Streaming TAS recognizer fusing MS-TCN, ROI motion energy, object velocity, and hand-tool cues."""

from __future__ import annotations

import importlib.util
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as T

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
TAS_MODEL_PY = WORKSPACE_ROOT / "TAS" / "model.py"
DEFAULT_WEIGHTS = WORKSPACE_ROOT / "TAS" / "Weights" / "mstcn_cooking.pth"

ID_TO_ACTION: Dict[int, str] = {
    0: "Idle",
    1: "Chopping",
    2: "Stirring",
    3: "Adding",
}

ACTION_COLORS: Dict[str, Tuple[int, int, int]] = {
    "Idle": (160, 160, 160),
    "Chopping": (0, 165, 255),
    "Stirring": (255, 180, 0),
    "Adding": (50, 220, 50),
}

# Training class priors from TAS/labels.json ({0: 4982, 1: 827, 2: 1031, 3: 1505})
TAS_TRAIN_PRIORS = np.array([0.5970, 0.0991, 0.1235, 0.1803], dtype=np.float32)

HAND_LABELS = {"hand", "human hand"}
VESSEL_LABELS = {"pan", "frying pan", "skillet", "pot", "saucepan", "bowl", "mixing bowl"}
KNIFE_LABELS = {"knife", "kitchen knife", "chef knife"}
BOARD_LABELS = {"cutting_board", "cutting board", "chopping board"}
STIRRER_LABELS = {"spatula", "spoon", "wooden spoon", "metal spoon", "ladle", "fork", "whisk", "tongs"}
CONTAINER_ADD_LABELS = {"bottle", "oil bottle", "jar", "spice jar", "cup", "measuring cup"}
NON_FOOD_LABELS = (
    HAND_LABELS | VESSEL_LABELS | KNIFE_LABELS | BOARD_LABELS | STIRRER_LABELS | CONTAINER_ADD_LABELS
    | {"lid", "hob", "stove", "stove burner", "cooktop", "tap", "sink", "plate", "appliance",
       "microwave", "kettle", "blender", "toaster", "oven", "faucet", "kitchen towel", "sponge"}
)


def _is_addable(det) -> bool:
    """Return True if detection is a pourable container or food/ingredient item."""
    lbl = det.label.lower().strip()
    return lbl in CONTAINER_ADD_LABELS or lbl not in NON_FOOD_LABELS


def _load_mstcn_class():
    """Load MultiStageModel from TAS/model.py without polluting sys.modules['model']."""
    spec = importlib.util.spec_from_file_location("tas_model", str(TAS_MODEL_PY))
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load TAS model from {TAS_MODEL_PY}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.MultiStageModel


def _boxes_near(a, b, pad: float = 0.07) -> bool:
    ax1, ay1 = a.cx - a.w / 2.0 - pad, a.cy - a.h / 2.0 - pad
    ax2, ay2 = a.cx + a.w / 2.0 + pad, a.cy + a.h / 2.0 + pad
    bx1, by1 = b.cx - b.w / 2.0, b.cy - b.h / 2.0
    bx2, by2 = b.cx + b.w / 2.0, b.cy + b.h / 2.0
    return (ax1 <= bx2 and ax2 >= bx1) and (ay1 <= by2 and ay2 >= by1)


def _roi_motion_fraction(diff_map: np.ndarray, boxes: List) -> float:
    """Fraction of moving pixels (>0.07 intensity delta) inside the union of `boxes`."""
    if not boxes:
        return 0.0
    h, w = diff_map.shape
    mask = np.zeros((h, w), dtype=bool)
    for b in boxes:
        x1 = max(0, int((b.cx - b.w / 2.0) * w))
        y1 = max(0, int((b.cy - b.h / 2.0) * h))
        x2 = min(w, int((b.cx + b.w / 2.0) * w))
        y2 = min(h, int((b.cy + b.h / 2.0) * h))
        if x2 > x1 and y2 > y1:
            mask[y1:y2, x1:x2] = True
    if not np.any(mask):
        return 0.0
    active = diff_map[mask] > 0.07
    return float(np.mean(active))


def _max_centroid_shift(curr_boxes: List, prev_boxes: List) -> float:
    """Estimate maximum object centroid shift between consecutive samples."""
    if not curr_boxes or not prev_boxes:
        return 0.0
    best = 0.0
    for c in curr_boxes:
        nearest = min(float(np.hypot(c.cx - p.cx, c.cy - p.cy)) for p in prev_boxes)
        if nearest < 0.25:
            best = max(best, nearest)
    return best


@dataclass
class ActionPrediction:
    """Single-step output of the streaming TAS + motion + interaction recognizer."""

    action: str
    confidence: float
    probs: Dict[str, float]
    buffer_steps: int
    sampled_new_step: bool
    motion_score: float = 0.0
    action_detail: str = "Idle"


class TASActionRecognizer:
    """Real-time 5 FPS ResNet-18 + MS-TCN fused with ROI motion, object velocity, and tool cues."""

    def __init__(
        self,
        weights_path: str | Path = DEFAULT_WEIGHTS,
        sample_fps: float = 5.0,
        max_buffer_steps: int = 150,
        smooth_window: int = 4,
        device: Optional[str] = None,
    ) -> None:
        self.sample_fps = sample_fps
        self.sample_interval_sec = 1.0 / max(0.5, sample_fps)
        self.max_buffer_steps = max_buffer_steps
        self.smooth_window = max(1, smooth_window)

        self.device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        w_path = Path(weights_path) if Path(weights_path).is_absolute() else (WORKSPACE_ROOT / weights_path)
        if not w_path.exists():
            raise FileNotFoundError(f"MS-TCN weights not found: {w_path}")

        backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        backbone.fc = nn.Identity()
        self._backbone = backbone.to(self.device).eval()
        self._transform = T.Compose(
            [
                T.ToPILImage(),
                T.Resize((224, 224)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

        MultiStageModel = _load_mstcn_class()
        self._mstcn = MultiStageModel(
            num_stages=2, num_layers=10, num_f_maps=64, dim=512, num_classes=4
        ).to(self.device)
        self._mstcn.load_state_dict(
            torch.load(str(w_path), map_location=self.device, weights_only=True)
        )
        self._mstcn.eval()

        self._feat_buffer: Deque[torch.Tensor] = deque(maxlen=self.max_buffer_steps)
        self._prob_history: Deque[np.ndarray] = deque(maxlen=self.smooth_window)
        self._prev_gray: Optional[np.ndarray] = None
        self._prev_dets: List = []
        self.timeline: List[Tuple[float, str, float]] = []
        self._last_sample_sec: Optional[float] = None
        self._last_pred = ActionPrediction(
            action="Idle",
            confidence=1.0,
            probs={name: (1.0 if i == 0 else 0.0) for i, name in ID_TO_ACTION.items()},
            buffer_steps=0,
            sampled_new_step=False,
            action_detail="Idle",
        )

    def reset(self) -> None:
        """Clear temporal feature buffer, motion state, and prediction history."""
        self._feat_buffer.clear()
        self._prob_history.clear()
        self._prev_gray = None
        self._prev_dets = []
        self.timeline.clear()
        self._last_sample_sec = None
        self._last_pred = ActionPrediction(
            action="Idle",
            confidence=1.0,
            probs={name: (1.0 if i == 0 else 0.0) for i, name in ID_TO_ACTION.items()},
            buffer_steps=0,
            sampled_new_step=False,
            action_detail="Idle",
        )

    def _compute_motion_map(self, frame_bgr: np.ndarray) -> Tuple[np.ndarray, float]:
        """Return (112x112 float32 abs-diff map, active_motion_fraction in [0, 1])."""
        small = cv2.resize(frame_bgr, (112, 112))
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        if self._prev_gray is None:
            self._prev_gray = gray
            return np.zeros_like(gray), 0.0
        diff = np.abs(gray - self._prev_gray)
        self._prev_gray = gray
        # Fraction of pixels with noticeable change (>0.07) avoids static-background dilution
        active_frac = float(np.mean(diff > 0.07))
        return diff, float(np.clip(active_frac / 0.20, 0.0, 1.0))

    def _compute_interaction_probs(
        self,
        detections: Optional[Iterable],
        diff_map: np.ndarray,
        global_motion: float,
    ) -> Tuple[Optional[np.ndarray], str]:
        """Score [Idle, Chopping, Stirring, Adding] from ROI motion, velocity, and tool cues."""
        if not detections:
            return None, "Idle (No Objects)"

        det_list = list(detections)
        hands = [d for d in det_list if d.label.lower().strip() in HAND_LABELS]
        vessels = [d for d in det_list if d.label.lower().strip() in VESSEL_LABELS]
        knives = [d for d in det_list if d.label.lower().strip() in KNIFE_LABELS]
        boards = [d for d in det_list if d.label.lower().strip() in BOARD_LABELS]
        stirrers = [d for d in det_list if d.label.lower().strip() in STIRRER_LABELS]
        addables = [d for d in det_list if _is_addable(d)]

        prev_hands = [d for d in self._prev_dets if d.label.lower().strip() in HAND_LABELS]
        prev_stirrers = [d for d in self._prev_dets if d.label.lower().strip() in STIRRER_LABELS]
        prev_addables = [d for d in self._prev_dets if _is_addable(d)]
        self._prev_dets = det_list

        vessel_motion = _roi_motion_fraction(diff_map, vessels)
        board_motion = _roi_motion_fraction(diff_map, boards)
        hand_motion = _roi_motion_fraction(diff_map, hands)
        hand_shift = _max_centroid_shift(hands, prev_hands)
        utensil_shift = _max_centroid_shift(stirrers, prev_stirrers)
        addable_shift = _max_centroid_shift(addables, prev_addables)

        scores = np.array([0.08, 0.02, 0.02, 0.02], dtype=np.float32)
        detail = "Idle (Stationary)"

        # 1. Chopping: knife + board/hand or active cutting-board motion
        chop_cue = any(_boxes_near(h, k, 0.08) for h in hands for k in knives) or any(
            _boxes_near(k, b, 0.06) for k in knives for b in boards
        )
        if chop_cue and (hand_motion > 0.05 or board_motion > 0.05 or global_motion > 0.12):
            scores[1] += 0.85 + 0.35 * max(board_motion, hand_motion)
            detail = "Chopping (Knife on Board)"
        elif boards and board_motion > 0.12 and hands:
            scores[1] += 0.65 + 0.30 * board_motion
            detail = "Chopping (Board Activity)"

        # 2. Adding: an addable item (bottle/spice/food/cup) is moving or held near a pan/bowl
        moving_addables = [
            a for a in addables
            if any(_boxes_near(h, a, 0.07) for h in hands) or addable_shift > 0.02
        ]
        adding_over_vessel = any(
            _boxes_near(a, v, 0.07) for a in moving_addables for v in vessels
        )
        # Also detect hand carrying/placing food into vessel without a stirring utensil
        hand_only_to_vessel = (
            any(_boxes_near(h, v, 0.06) for h in hands for v in vessels)
            and not any(_boxes_near(h, u, 0.07) for h in hands for u in stirrers)
            and bool(moving_addables)
        )
        if adding_over_vessel:
            scores[3] += 0.90 + 0.30 * max(vessel_motion, addable_shift * 4.0)
            detail = "Adding (Ingredient/Bottle -> Vessel)"
        elif hand_only_to_vessel and (hand_shift > 0.02 or vessel_motion > 0.06):
            scores[3] += 0.72 + 0.25 * vessel_motion
            detail = "Adding (Placing into Pan/Bowl)"

        # 3. Stirring / Active Pan Cooking: utensil or hand moving inside pan/pot/bowl
        utensil_in_vessel = any(_boxes_near(u, v, 0.06) for u in stirrers for v in vessels)
        hand_with_utensil = any(_boxes_near(h, u, 0.08) for h in hands for u in stirrers)
        hand_at_vessel = any(_boxes_near(h, v, 0.08) for h in hands for v in vessels)

        if utensil_in_vessel and (vessel_motion > 0.04 or utensil_shift > 0.012 or hand_with_utensil):
            scores[2] += 0.92 + 0.35 * max(vessel_motion, utensil_shift * 4.0)
            detail = "Stirring (Utensil in Pan/Bowl)"
        elif hand_with_utensil and (hand_motion > 0.05 or hand_shift > 0.015):
            scores[2] += 0.78 + 0.25 * hand_motion
            detail = "Stirring (Hand + Utensil)"
        elif hand_at_vessel and (vessel_motion > 0.05 or hand_shift > 0.02):
            scores[2] += 0.70 + 0.30 * vessel_motion
            detail = "Cooking / Flipping (Hand at Pan)"
        elif vessels and vessel_motion > 0.14:
            # Catches active pan sizzling/stirring even when hand is briefly occluded
            scores[2] += 0.58 + 0.30 * vessel_motion
            detail = "Cooking (Active Pan Motion)"

        # 4. Idle when workspace has negligible motion and no active interaction
        best_active = float(np.max(scores[1:]))
        if best_active < 0.20:
            scores[0] += 0.85 if global_motion < 0.20 else 0.45
            detail = "Idle (No Active Cooking Motion)"

        return scores / scores.sum(), detail

    def step(
        self,
        frame_bgr: np.ndarray,
        timestamp_sec: float,
        detections: Optional[Iterable] = None,
    ) -> ActionPrediction:
        """Process a frame; runs ResNet-18 + MS-TCN + ROI motion & interaction fusion at 5 FPS."""
        if (
            self._last_sample_sec is not None
            and (timestamp_sec - self._last_sample_sec) < self.sample_interval_sec - 1e-4
        ):
            return ActionPrediction(
                action=self._last_pred.action,
                confidence=self._last_pred.confidence,
                probs=self._last_pred.probs,
                buffer_steps=len(self._feat_buffer),
                sampled_new_step=False,
                motion_score=self._last_pred.motion_score,
                action_detail=self._last_pred.action_detail,
            )

        self._last_sample_sec = timestamp_sec
        diff_map, motion_score = self._compute_motion_map(frame_bgr)
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        inp = self._transform(rgb).unsqueeze(0).to(self.device)

        with torch.no_grad():
            feat = self._backbone(inp).squeeze(0).cpu()
            self._feat_buffer.append(feat)

            seq = torch.stack(list(self._feat_buffer), dim=1).unsqueeze(0).to(self.device)
            logits = self._mstcn(seq)[-1].squeeze(0)[:, -1]
            # Debias against 60% Idle class imbalance in TAS/labels.json
            debiased_logits = logits - 0.80 * torch.log(
                torch.tensor(TAS_TRAIN_PRIORS, device=logits.device)
            )
            mstcn_probs = F.softmax(debiased_logits, dim=0).cpu().numpy()

        cue_probs, cue_detail = self._compute_interaction_probs(detections, diff_map, motion_score)
        if cue_probs is not None:
            fused = 0.22 * mstcn_probs + 0.78 * cue_probs
        else:
            fused = mstcn_probs
        fused = fused / max(1e-6, float(fused.sum()))

        self._prob_history.append(fused)
        smoothed = np.mean(np.stack(list(self._prob_history), axis=0), axis=0)
        pred_id = int(np.argmax(smoothed))
        pred_label = ID_TO_ACTION[pred_id]
        pred_conf = float(smoothed[pred_id])

        detail = cue_detail if pred_label != "Idle" else "Idle"
        probs_dict = {ID_TO_ACTION[i]: round(float(smoothed[i]), 3) for i in range(4)}
        self.timeline.append((timestamp_sec, pred_label, pred_conf))
        if len(self.timeline) > 120:
            self.timeline = self.timeline[-120:]

        self._last_pred = ActionPrediction(
            action=pred_label,
            confidence=pred_conf,
            probs=probs_dict,
            buffer_steps=len(self._feat_buffer),
            sampled_new_step=True,
            motion_score=round(motion_score, 3),
            action_detail=detail,
        )
        return self._last_pred
