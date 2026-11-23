"""Kitchen object class schemas, expanded YOLO-World prompts, and CaptainCook4D vocab mapping."""

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

CLASS_COLORS: Tuple[Tuple[int, int, int], ...] = (
    (0, 180, 255),   # 0: hand
    (255, 200, 0),   # 1: pan
    (255, 130, 40),  # 2: pot
    (110, 110, 230), # 3: stove
    (60, 60, 255),   # 4: knife
    (200, 100, 255), # 5: spatula
    (240, 180, 120), # 6: spoon
    (60, 140, 200),  # 7: cutting_board
    (80, 220, 220),  # 8: bowl
    (220, 220, 220), # 9: plate
    (200, 255, 150), # 10: cup
    (180, 130, 255), # 11: bottle
    (170, 170, 170), # 12: appliance
    (120, 255, 255), # 13: egg
    (70, 220, 70),   # 14: vegetable
    (90, 170, 240),  # 15: bread
    (70, 90, 210),   # 16: meat
    (110, 230, 130), # 17: ingredients
)

# Expanded 73-object YOLO-World kitchen vocabulary -> (canonical_18_id, dag_vocab_terms).
# Intentionally omits generic "pan lid" (which false-triggers on stove burners) and
# separates beverage items ("coffee mug", "coffee grounds") from generic containers.
WORLD_PROMPT_SPECS: Dict[str, Tuple[int, Tuple[str, ...]]] = {
    # Hands
    "hand": (0, ()),
    "human hand": (0, ()),
    # Cookware & Stove
    "frying pan": (1, ("pan", "skillet")),
    "skillet": (1, ("pan", "skillet")),
    "saucepan": (2, ("pot", "pan")),
    "pot": (2, ("pot",)),
    "baking tray": (1, ("pan",)),
    "stove burner": (3, ("stove",)),
    "cooktop": (3, ("stove",)),
    "oven": (12, ("oven",)),
    # Knives & Prep Tools
    "kitchen knife": (4, ("knife",)),
    "chef knife": (4, ("knife",)),
    "peeler": (4, ("peeler",)),
    "grater": (4, ("grater",)),
    "can opener": (4, ()),
    # Stirring / Cooking Utensils
    "spatula": (5, ("spatula",)),
    "tongs": (5, ("spatula",)),
    "ladle": (6, ("spoon",)),
    "wooden spoon": (6, ("spoon",)),
    "metal spoon": (6, ("spoon",)),
    "fork": (6, ("fork",)),
    "whisk": (6, ("whisk",)),
    # Surfaces, Bowls & Containers
    "cutting board": (7, ("cutting board",)),
    "mixing bowl": (8, ("bowl",)),
    "bowl": (8, ("bowl",)),
    "colander": (8, ("bowl",)),
    "strainer": (8, ("bowl",)),
    "plate": (9, ("plate",)),
    "cup": (10, ("cup",)),
    "glass": (10, ("cup",)),
    "measuring cup": (10, ("cup",)),
    "coffee mug": (10, ("mug",)),
    "oil bottle": (11, ("oil",)),
    "bottle": (11, ("oil",)),
    "spice jar": (11, ("jar", "salt", "pepper")),
    "salt shaker": (11, ("salt",)),
    "jar": (11, ("jar", "sauce")),
    # Kitchen Appliances & Fixtures
    "electric kettle": (12, ("kettle",)),
    "microwave": (12, ("microwave",)),
    "blender": (12, ("blender",)),
    "toaster": (12, ("toaster",)),
    "sink": (12, ()),
    "faucet": (12, ()),
    "kitchen towel": (12, ()),
    "sponge": (12, ()),
    # Proteins & Meat
    "egg": (13, ("egg",)),
    "raw meat": (16, ("meat",)),
    "cooked meat": (16, ("meat",)),
    "steak": (16, ("meat",)),
    "chicken": (16, ("meat",)),
    "beef": (16, ("meat",)),
    "pork": (16, ("meat",)),
    "sausage": (16, ("sausage", "meat")),
    "bacon": (16, ("bacon", "meat")),
    "fish": (16, ("meat",)),
    "tofu": (16, ("tofu",)),
    # Produce & Vegetables
    "tomato": (14, ("tomato",)),
    "onion": (14, ("onion",)),
    "garlic": (14, ("garlic",)),
    "potato": (14, ("potato",)),
    "carrot": (14, ("carrot",)),
    "broccoli": (14, ("broccoli",)),
    "mushroom": (14, ("mushroom",)),
    "bell pepper": (14, ("pepper",)),
    "cucumber": (14, ("cucumber",)),
    "avocado": (14, ("avocado",)),
    "lemon": (14, ("lemon",)),
    "lettuce": (14, ("lettuce",)),
    "herbs": (14, ("cilantro",)),
    # Starches, Dairy & Pantry Ingredients
    "bread": (15, ("bread",)),
    "tortilla": (15, ("tortilla", "wrap", "burrito")),
    "noodles": (17, ("noodles", "pasta", "ramen")),
    "pasta": (17, ("pasta", "noodles")),
    "rice": (17, ("rice",)),
    "cheese": (17, ("cheese", "mozzarella")),
    "butter": (17, ("butter",)),
    "milk": (17, ("milk",)),
    "sauce": (17, ("sauce",)),
    "flour": (17, ("batter",)),
    "sugar": (17, ("sugar",)),
    "coffee grounds": (17, ("coffee",)),
}

# Lower threshold for small tools/ingredients; stricter threshold for false-positive-prone classes
HIGH_THRESHOLD_LABELS: Dict[str, float] = {
    "electric kettle": 0.38,
    "coffee mug": 0.36,
    "coffee grounds": 0.42,
    "toaster": 0.36,
    "microwave": 0.34,
    "blender": 0.34,
    "milk": 0.34,
}

SMALL_OBJECT_LABELS: Set[str] = {
    "hand", "human hand", "kitchen knife", "chef knife", "spatula", "tongs",
    "ladle", "wooden spoon", "metal spoon", "fork", "whisk", "peeler", "grater",
    "egg", "garlic", "lemon", "herbs", "raw meat", "cooked meat", "steak",
    "chicken", "beef", "pork", "bacon", "sausage", "spice jar", "salt shaker",
}

CANONICAL_TO_DAG_VOCAB: Dict[str, Tuple[str, ...]] = {
    "hand": (),
    "pan": ("pan", "skillet"),
    "pot": ("pot",),
    "stove": ("stove",),
    "knife": ("knife",),
    "spatula": ("spatula",),
    "spoon": ("spoon", "fork", "whisk"),
    "cutting_board": ("cutting board",),
    "bowl": ("bowl",),
    "plate": ("plate",),
    "cup": ("cup",),
    "bottle": ("oil", "jar"),
    "appliance": ("microwave", "blender", "toaster"),
    "egg": ("egg",),
    "vegetable": ("tomato", "onion", "garlic", "cucumber", "broccoli", "mushroom"),
    "bread": ("bread", "tortilla", "wrap"),
    "meat": ("meat", "bacon", "sausage", "meatball"),
    "ingredients": ("cheese", "butter", "salt", "pepper", "sauce"),
    "lid": ("lid",),
    "hob": ("stove",),
    "tap": (),
    "sink": (),
    "fork": ("fork",),
    "jar": ("jar", "salt", "pepper", "sauce"),
    "onion": ("onion",),
    "tomato": ("tomato",),
    "potato": ("potato",),
    "carrot": ("carrot",),
    "garlic": ("garlic",),
    "pepper": ("pepper",),
    "cheese": ("cheese", "mozzarella"),
    "food": ("pasta", "noodles", "rice", "butter", "sauce"),
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
    return list(WORLD_PROMPT_SPECS.keys())


def get_label_min_conf(label: str, default_conf: float) -> float:
    """Return class-aware confidence threshold to catch small tools while curbing false positives."""
    key = label.lower().strip()
    if key in HIGH_THRESHOLD_LABELS:
        return max(default_conf, HIGH_THRESHOLD_LABELS[key])
    if key in SMALL_OBJECT_LABELS:
        return min(default_conf, 0.20)
    return default_conf


def map_label_to_dag_terms(label: str) -> Set[str]:
    """Convert a detector label into CaptainCook4D KITCHEN_VOCAB terms."""
    key = label.lower().strip()
    if key in WORLD_PROMPT_SPECS:
        return set(WORLD_PROMPT_SPECS[key][1])
    if key in CANONICAL_TO_DAG_VOCAB:
        return set(CANONICAL_TO_DAG_VOCAB[key])
    return {key.replace("_", " ")}
