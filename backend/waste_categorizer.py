"""
Advanced Waste Categorization Engine (Improved Version)
-----------------------------------------------------
• Clean modular architecture
• Dataclass-based items
• Faster lookups using merged registry
• Stronger inference engine
• Structured response objects
• Extensible categories + dynamic rules
"""

import logging
from typing import Dict, List, Optional
from enum import Enum
from dataclasses import dataclass, asdict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# ENUMS
# ----------------------------------------------------------------------

class WasteCategory(str, Enum):
    RECYCLABLE = "recyclable"
    NON_RECYCLABLE = "non_recyclable"
    HEALTHY = "healthy"
    HAZARDOUS = "hazardous"


class MaterialType(str, Enum):
    PLASTIC = "plastic"
    GLASS = "glass"
    METAL = "metal"
    PAPER = "paper"
    ORGANIC = "organic"
    MIXED = "mixed"
    CHEMICAL = "chemical"
    ELECTRONIC = "electronic"
    UNKNOWN = "unknown"


class RecyclingGrade(str, Enum):
    EASY = "easy"
    MODERATE = "moderate"
    DIFFICULT = "difficult"
    NOT_RECYCLABLE = "not_recyclable"


# ----------------------------------------------------------------------
# DATACLASSES
# ----------------------------------------------------------------------

@dataclass
class BaseItem:
    name: str
    category: WasteCategory
    material: MaterialType = MaterialType.UNKNOWN
    confidence: float = 0.0

    def to_dict(self):
        d = asdict(self)
        d["category"] = self.category.value
        d["material"] = self.material.value
        return d


@dataclass
class RecyclableItem(BaseItem):
    grade: RecyclingGrade = RecyclingGrade.EASY
    instructions: str = ""
    impact: str = "medium"
    recycling_symbol: Optional[str] = None
    processing_facility: Optional[str] = None


@dataclass
class HazardousItem(BaseItem):
    hazard_type: str = ""
    severity: str = ""
    risks: Optional[List[str]] = None
    disposal: str = ""
    instructions: str = ""
    regulations: str = ""


# ----------------------------------------------------------------------
# MAIN ENGINE
# ----------------------------------------------------------------------

class WasteCategorizer:
    """
    Improved intelligent waste categorizer.
    Cleans architecture, adds extensibility, faster matching & inference.
    """

    def __init__(self):
        self._load_databases()
        self._build_registry()   # combined lookup map
        logger.info("⚡ Improved Waste Categorizer initialized")

    # ------------------------------------------------------------------
    # DATABASES
    # ------------------------------------------------------------------

    def _load_databases(self):
        """Load and store item dictionaries."""

        # --------------------------- RECYCLABLE --------------------------
        self.recyclables: Dict[str, RecyclableItem] = {
            "plastic_bottle": RecyclableItem(
                name="plastic_bottle",
                category=WasteCategory.RECYCLABLE,
                material=MaterialType.PLASTIC,
                grade=RecyclingGrade.EASY,
                instructions="Rinse, remove cap, flatten",
                impact="high",
                recycling_symbol="PET (1)",
                processing_facility="municipal"
            ),
            "paper": RecyclableItem(
                name="paper",
                category=WasteCategory.RECYCLABLE,
                material=MaterialType.PAPER,
                grade=RecyclingGrade.EASY,
                instructions="Keep dry, no stains",
                impact="low",
                recycling_symbol="PAP (20-22)",
                processing_facility="municipal"
            ),
        }

        # --------------------------- HAZARDOUS ---------------------------
        self.hazardous: Dict[str, HazardousItem] = {
            "battery": HazardousItem(
                name="battery",
                category=WasteCategory.HAZARDOUS,
                material=MaterialType.CHEMICAL,
                hazard_type="chemical",
                severity="high",
                risks=["Heavy metals", "Leakage", "Fire risk"],
                disposal="hazardous_facility",
                instructions="Never throw in regular trash",
                regulations="EPA regulated"
            ),
            "broken_glass": HazardousItem(
                name="broken_glass",
                category=WasteCategory.HAZARDOUS,
                material=MaterialType.GLASS,
                hazard_type="physical",
                severity="medium",
                risks=["Cuts, lacerations"],
                disposal="wrap_and_label",
                instructions="Label as BROKEN GLASS"
            ),
        }

        # --------------------------- HEALTHY ----------------------------
        self.healthy = {
            "banana": WasteCategory.HEALTHY,
            "apple": WasteCategory.HEALTHY,
            "food_waste": WasteCategory.HEALTHY
        }

        # ------------------------ NON-RECYCLABLE ------------------------
        self.non_recyclables = {
            "wrapper": WasteCategory.NON_RECYCLABLE,
            "styrofoam": WasteCategory.NON_RECYCLABLE,
            "general_trash": WasteCategory.NON_RECYCLABLE,
        }

    # ------------------------------------------------------------------
    # REGISTRY
    # ------------------------------------------------------------------

    def _build_registry(self):
        """Merge all supported YOLO classes into fast lookup table."""
        self.registry = {}

        # recyclable
        for k, v in self.recyclables.items():
            self.registry[k] = ("recyclable", v)

        # hazardous
        for k, v in self.hazardous.items():
            self.registry[k] = ("hazardous", v)

        # healthy
        for k in self.healthy:
            self.registry[k] = ("healthy", None)

        # non-recyclable
        for k in self.non_recyclables:
            self.registry[k] = ("non_recyclable", None)

    # ------------------------------------------------------------------
    # MAIN PUBLIC FUNCTIONS
    # ------------------------------------------------------------------

    def categorize(self, class_name: str, confidence: float = 0.0):
        """
        Main entry — categorize a single YOLO detection.
        """
        cls = class_name.lower()

        if cls in self.registry:
            ctype, obj = self.registry[cls]
            return self._handle_known(cls, confidence, ctype, obj)

        # Unknown → infer
        return self._infer_item(cls, confidence)

    def batch_categorize(self, detections: List[Dict]):
        """
        Handle many YOLO detections at once.
        """

        categorized = [self.categorize(d["class"], d["confidence"]) for d in detections]

        stats = {
            "recyclable": sum(i["category"] == "recyclable" for i in categorized),
            "non_recyclable": sum(i["category"] == "non_recyclable" for i in categorized),
            "healthy": sum(i["category"] == "healthy" for i in categorized),
            "hazardous": sum(i["category"] == "hazardous" for i in categorized),
        }

        return {
            "categorized_items": categorized,
            "summary": stats,
            "hazardous_alert": stats["hazardous"] > 0,
        }

    # ------------------------------------------------------------------
    # HANDLERS
    # ------------------------------------------------------------------

    def _handle_known(self, cls, confidence, ctype, obj):
        """Handles items that exist in registry."""
        if ctype == "recyclable":
            recyclable_item: RecyclableItem = obj
            recyclable_item.confidence = confidence
            return recyclable_item.to_dict()

        if ctype == "hazardous":
            hazardous_item: HazardousItem = obj
            d = hazardous_item.to_dict()
            d["confidence"] = confidence
            d["requires_special_handling"] = True
            return d

        return {
            "class": cls,
            "confidence": confidence,
            "category": ctype,
            "material": "mixed",
            "instructions": "Dispose accordingly",
        }

    # ------------------------------------------------------------------
    # INFERENCE ENGINE FOR UNKNOWN CLASSES
    # ------------------------------------------------------------------

    def _infer_item(self, name: str, conf: float):
        """Infer waste type based on keyword patterns."""

        low = name.lower()

        if any(k in low for k in ["broken", "chemical", "hazard", "sharp", "battery"]):
            return {
                "class": name,
                "confidence": conf,
                "category": WasteCategory.HAZARDOUS.value,
                "instructions": "Treat as hazardous. Use special disposal.",
            }

        if any(k in low for k in ["bottle", "can", "glass", "paper", "metal"]):
            return {
                "class": name,
                "confidence": conf,
                "category": WasteCategory.RECYCLABLE.value,
                "instructions": "Likely recyclable. Verify cleanup.",
            }

        if any(k in low for k in ["banana", "fruit", "vegetable", "food"]):
            return {
                "class": name,
                "confidence": conf,
                "category": WasteCategory.HEALTHY.value,
                "instructions": "Organic waste. Compost recommended.",
            }

        return {
            "class": name,
            "confidence": conf,
            "category": WasteCategory.NON_RECYCLABLE.value,
            "instructions": "Unknown item. Default to general waste.",
        }


# Singleton access
_categorizer_singleton = None

def get_waste_categorizer():
    global _categorizer_singleton
    if _categorizer_singleton is None:
        _categorizer_singleton = WasteCategorizer()
    return _categorizer_singleton