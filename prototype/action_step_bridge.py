"""Semantic bridge mapping coarse TAS actions + YOLO objects to CaptainCook4D recipe steps."""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
CC4D_SRC = WORKSPACE_ROOT / "captaincook4d_project" / "src"
if str(CC4D_SRC) not in sys.path:
    sys.path.insert(0, str(CC4D_SRC))

from dag_loader import RecipeDAGState, extract_objects  # noqa: E402


# Maps the 4 MS-TCN action classes to CaptainCook4D leading-verb families
TAS_TO_VERBS: Dict[str, Set[str]] = {
    "Idle": set(),
    "Chopping": {"chop", "slice", "dice", "mince", "cut", "peel", "grate"},
    "Stirring": {"stir", "mix", "mixing", "whisk", "saute", "sauté", "cook", "fry", "scramble", "boil", "heat"},
    "Adding": {
        "add", "pour", "transfer", "put", "place", "crack", "spread",
        "garnish", "sprinkle", "measure", "dip", "top", "scoop",
    },
}


def _extract_leading_verb(step_desc: str) -> str:
    """CaptainCook4D steps are formatted as 'Verb-Sentence...'; extract that verb."""
    prefix = step_desc.split("-", 1)[0].strip().lower()
    words = re.findall(r"[a-z]+", prefix)
    return words[0] if words else ""


def _step_matches_tas_action(step_desc: str, tas_action: str) -> bool:
    """Return True if the step's action verb belongs to `tas_action`'s verb family."""
    verbs = TAS_TO_VERBS.get(tas_action, set())
    if not verbs:
        return False
    lead = _extract_leading_verb(step_desc)
    if lead:
        if lead in verbs:
            return True
        # Check sentence-initial verb right after '-' as fallback
        body = step_desc.split("-", 1)[-1].strip().lower()
        body_words = re.findall(r"[a-z]+", body)[:3]
        return any(w in verbs for w in body_words)
    tokens = set(re.findall(r"[a-z]+", step_desc.lower())[:4])
    return bool(tokens & verbs)


class ActionToStepBridge:
    """Resolves coarse TAS actions + YOLO object context into specific CaptainCook4D steps.

    Prevents a sustained action (e.g. Chopping) from advancing multiple steps every
    0.2s frame by enforcing a per-step cooldown and action-transition gate.
    """

    def __init__(self, step_cooldown_sec: float = 5.0, min_action_conf: float = 0.45) -> None:
        self.step_cooldown_sec = step_cooldown_sec
        self.min_action_conf = min_action_conf
        self._last_emitted_action: Optional[str] = None
        self._last_emitted_step: Optional[str] = None
        self._last_emit_time: float = -999.0
        self._emit_count_for_step: int = 0

    def reset(self) -> None:
        self._last_emitted_action = None
        self._last_emitted_step = None
        self._last_emit_time = -999.0
        self._emit_count_for_step = 0

    def resolve_step(
        self,
        tas_action: str,
        tas_conf: float,
        candidate_dags: Iterable[RecipeDAGState],
        current_objects: Set[str],
        cumulative_objects: Set[str],
        timestamp_sec: float,
    ) -> Optional[str]:
        """Choose the best matching step description across `candidate_dags`.

        Emits the same matched step twice consecutively (so `CaptainCookRecipeEngine`'s
        `DEBOUNCE_MIN = 2` fires cleanly), then holds for `step_cooldown_sec` or until
        the TAS action changes.
        """
        if tas_action == "Idle" or tas_conf < self.min_action_conf:
            self._last_emitted_action = None
            return None

        if (
            self._last_emitted_step is not None
            and self._emit_count_for_step == 1
            and tas_action == self._last_emitted_action
        ):
            self._emit_count_for_step = 2
            self._last_emit_time = timestamp_sec
            return self._last_emitted_step

        if (
            tas_action == self._last_emitted_action
            and (timestamp_sec - self._last_emit_time) < self.step_cooldown_sec
        ):
            return None

        best_desc: Optional[str] = None
        best_score = -1.0

        for rank_idx, dag in enumerate(candidate_dags):
            rank_bonus = max(0.0, 0.15 - 0.03 * rank_idx)
            for sid, desc in sorted(dag.steps.items()):
                if sid in (dag.start_id, dag.end_id) or sid in dag.completed:
                    continue
                if not _step_matches_tas_action(desc, tas_action):
                    continue

                step_objs = extract_objects([desc])
                cur_overlap = len(step_objs & current_objects) / max(1, len(step_objs)) if step_objs else 0.0
                cum_overlap = len(step_objs & cumulative_objects) / max(1, len(step_objs)) if step_objs else 0.0
                ready_bonus = 0.35 if dag.is_ready(sid) else 0.0

                score = 0.40 + 0.65 * cur_overlap + 0.20 * cum_overlap + ready_bonus + rank_bonus
                if score > best_score:
                    best_score = score
                    best_desc = desc

        if best_desc is None:
            return None

        self._last_emitted_action = tas_action
        self._last_emitted_step = best_desc
        self._emit_count_for_step = 1
        self._last_emit_time = timestamp_sec
        return best_desc
