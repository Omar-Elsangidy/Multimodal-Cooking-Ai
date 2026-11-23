"""Food probe thermometer monitor acting as a cooking readiness and food-safety gate."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional


# Safe internal / target temperatures (Celsius) by CaptainCook4D / prototype recipe category
RECIPE_TARGET_TEMPS_C: Dict[str, float] = {
    "pansearedmeat": 74.0,
    "dressedupmeatballs": 74.0,
    "scrambledeggs": 71.0,
    "herbomeletwithfriedtomatoes": 71.0,
    "microwaveeggsandwich": 71.0,
    "microwavefrenchtoast": 71.0,
    "breakfastburritos": 71.0,
    "coffee": 91.0,
    "spicedhotchocolate": 75.0,
    "ramen": 95.0,
    "panfriedtofu": 68.0,
    "sautedmushrooms": 70.0,
    "broccolistirfry": 70.0,
    "zoodles": 70.0,
    "tomatochutney": 85.0,
    "blenderbananapancakes": 75.0,
    "microwavemugpizza": 74.0,
    "mugcake": 75.0,
}


@dataclass
class TempStatus:
    """Snapshot of the food probe thermometer and active safety/doneness gate."""

    temp_c: float
    target_c: Optional[float]
    mode: str
    gate_passed: bool
    alert: Optional[str] = None
    label: str = "Ambient"


class FoodProbeMonitor:
    """Tracks food probe temperature and enforces cooking safety / doneness gates.

    Default mode is 'manual' (sensor-driven): the probe holds the actual measured
    reading rather than drifting upward on its own. Pass mode='auto' for simulated
    heating during offline demos, or adjust live via +/- keys / Web UI slider.
    """

    def __init__(self, initial_temp_c: float = 22.0, mode: str = "manual") -> None:
        self.temp_c = float(initial_temp_c)
        self.mode = mode if mode in ("manual", "auto") else "manual"
        self.target_c: Optional[float] = None

    def set_temp(self, temp_c: float) -> None:
        """Set probe temperature from sensor reading or UI slider."""
        self.temp_c = max(0.0, min(260.0, float(temp_c)))

    def adjust(self, delta_c: float) -> None:
        """Nudge probe temperature (used by live HUD +/- keys)."""
        self.mode = "manual"
        self.temp_c = max(0.0, min(260.0, self.temp_c + delta_c))

    def set_mode(self, mode: str) -> None:
        """Switch between 'manual' (sensor) and 'auto' (simulated heating)."""
        if mode in ("auto", "manual"):
            self.mode = mode

    def reset(self, temp_c: float = 22.0) -> None:
        self.temp_c = float(temp_c)
        self.target_c = None

    def update(
        self,
        dt_sec: float,
        recipe_id: str,
        active_objects: set[str],
        tas_action: str,
        active_step_text: str = "",
    ) -> TempStatus:
        """Evaluate probe against recipe safety/doneness thresholds."""
        step_low = active_step_text.lower()

        # Determine target temperature for this recipe / step
        if any(k in step_low for k in ("boil", "91-96", "195-205")):
            self.target_c = 91.0
        elif recipe_id in RECIPE_TARGET_TEMPS_C:
            self.target_c = RECIPE_TARGET_TEMPS_C[recipe_id]
        elif {"pan", "pot", "stove"} & active_objects:
            self.target_c = 74.0
        else:
            self.target_c = None

        # Optional automated thermal simulation (only when mode == 'auto')
        if self.mode == "auto" and dt_sec > 0:
            heat_active = (
                tas_action in ("Stirring", "Adding")
                or bool({"pan", "pot", "stove", "kettle"} & active_objects)
            )
            goal = (self.target_c + 6.0) if (heat_active and self.target_c is not None) else 22.0
            rate = 0.08 if goal > self.temp_c else 0.03
            self.temp_c += (goal - self.temp_c) * min(1.0, rate * dt_sec)

        gate_passed = self.target_c is None or self.temp_c >= self.target_c
        alert: Optional[str] = None

        if self.temp_c > 220.0:
            alert = f"OVERHEAT ({self.temp_c:.1f}C): Pan/probe temperature dangerously high!"
        elif self.target_c is not None and self.temp_c < self.target_c:
            # Food-safety check when user attempts to plate/serve/finish undercooked food
            if any(k in step_low for k in ("plate", "serve", "garnish", "transfer cooked", "pour a small amount")):
                alert = (
                    f"FOOD SAFETY GATE ({self.temp_c:.1f}C < {self.target_c:.0f}C): "
                    f"Not safe/hot enough to serve yet!"
                )
            elif tas_action == "Adding" and any(k in step_low for k in ("heat", "saute", "place meat")) and self.temp_c < 60.0:
                alert = f"PREHEAT CHECK ({self.temp_c:.1f}C): Preheat pan before adding food"

        if self.temp_c < 40.0:
            label = "Ambient"
        elif self.temp_c < 65.0:
            label = "Warming"
        elif self.target_c is not None and self.temp_c >= self.target_c:
            label = "Safe / Ready"
        else:
            label = "Cooking"

        return TempStatus(
            temp_c=round(self.temp_c, 1),
            target_c=self.target_c,
            mode=self.mode,
            gate_passed=gate_passed,
            alert=alert,
            label=label,
        )
