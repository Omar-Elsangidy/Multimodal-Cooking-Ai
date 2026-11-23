"""Kitchen object class schemas and optional YOLO-World prompt mappings."""

from __future__ import annotations

from typing import Dict, List, Set, Tuple

LEGACY_5_CLASSES: Tuple[str, ...] = (
    "hand",
    "pan",
    "knife",
    "cutting_board",
    "ingredients",
)

KITCHEN_18_CLASSES: Tuple[str, ...] = (
    "hand",
    "pan",
    "pot",
    "stove",
    "knife",
    "spatula",
    "spoon",
    "cutting_board",
    "bowl",
    "plate",
    "cup",
    "bottle",
    "appliance",
    "egg",
    "vegetable",
    "bread",
    "meat",
    "ingredients",
)

KITCHEN_30_CLASSES: Tuple[str, ...] = (
    "hand",
    "pan",
    "pot",
    "lid",
    "hob",
    "tap",
    "sink",
    "knife",
    "spatula",
    "spoon",
    "fork",
    "cutting_board",
    "bowl",
    "plate",
    "cup",
    "bottle",
    "jar",
    "appliance",
    "onion",
    "tomato",
    "potato",
    "carrot",
    "garlic",
    "pepper",
    "egg",
    "cheese",
    "bread",
    "meat",
    "vegetable",
    "food",
)

# Optional YOLO-World prompts mapped into KITCHEN_30_CLASSES IDs
WORLD_PROMPT_TO_ID: Dict[str, int] = {
    "hand": 0,
    "frying pan": 1,
    "skillet": 1,
    "pot": 2,
    "saucepan": 2,
    "pot lid": 3,
    "stove burner": 4,
    "cooktop": 4,
    "faucet": 5,
    "kitchen sink": 6,
    "kitchen knife": 7,
    "spatula": 8,
    "wooden spoon": 9,
    "metal spoon": 9,
    "fork": 10,
    "cutting board": 11,
    "bowl": 12,
    "plate": 13,
    "cup": 14,
    "coffee mug": 14,
    "bottle": 15,
    "jar": 16,
    "microwave": 17,
    "electric kettle": 17,
    "blender": 17,
    "toaster": 17,
    "onion": 18,
    "tomato": 19,
    "potato": 20,
    "carrot": 21,
    "garlic": 22,
    "bell pepper": 23,
    "egg": 24,
    "cheese": 25,
    "bread": 26,
    "raw meat": 27,
    "chicken": 27,
    "sausage": 27,
    "bacon": 27,
    "broccoli": 28,
    "lettuce": 28,
    "cucumber": 28,
    "mushroom": 28,
    "cooked food": 29,
}

HIGH_THRESHOLD_LABELS: Dict[str, float] = {
    "electric kettle": 0.38,
    "coffee mug": 0.36,
    "toaster": 0.36,
    "microwave": 0.34,
    "blender": 0.34,
}

SMALL_OBJECT_LABELS: Set[str] = {
    "hand",
    "kitchen knife",
    "spatula",
    "wooden spoon",
    "metal spoon",
    "fork",
    "egg",
    "garlic",
    "raw meat",
    "bacon",
    "sausage",
}


def get_class_names(num_classes: int) -> Tuple[str, ...]:
    """Return class tuple matching model output dimension (5, 18, or 30 classes)."""
    if num_classes == len(LEGACY_5_CLASSES):
        return LEGACY_5_CLASSES
    if num_classes == len(KITCHEN_18_CLASSES):
        return KITCHEN_18_CLASSES
    if num_classes == len(KITCHEN_30_CLASSES):
        return KITCHEN_30_CLASSES
    return tuple(f"class_{i}" for i in range(num_classes))


def get_world_prompts() -> List[str]:
    """Return ordered list of YOLO-World prompt strings."""
    return list(WORLD_PROMPT_TO_ID.keys())


def get_label_min_conf(label: str, default_conf: float) -> float:
    """Return class-aware confidence threshold for YOLO-World prompts."""
    key = label.lower().strip()
    if key in HIGH_THRESHOLD_LABELS:
        return max(default_conf, HIGH_THRESHOLD_LABELS[key])
    if key in SMALL_OBJECT_LABELS:
        return min(default_conf, 0.20)
    return default_conf
