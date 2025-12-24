"""Scene-to-Asset Compatibility Matrix.

This module defines the complete mapping between:
- Scene archetypes (kitchen, warehouse, etc.)
- Placement regions (counters, shelves, drawers, etc.)
- Asset categories (dishes, utensils, groceries, etc.)
- Contextual constraints (open dishwasher gets dishes, not floor)
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple


class SceneArchetype(str, Enum):
    """Scene archetypes matching archetype_config.json."""
    KITCHEN = "kitchen"
    GROCERY = "grocery"
    WAREHOUSE = "warehouse"
    LOADING_DOCK = "loading_dock"
    LAB = "lab"
    OFFICE = "office"
    UTILITY_ROOM = "utility_room"
    HOME_LAUNDRY = "home_laundry"
    BEDROOM = "bedroom"
    LIVING_ROOM = "living_room"
    BATHROOM = "bathroom"
    GENERIC = "generic"


class AssetCategory(str, Enum):
    """Asset categories with physics defaults."""
    # Kitchen
    DISHES = "dishes"
    UTENSILS = "utensils"
    COOKWARE = "cookware"
    FOOD_ITEMS = "food_items"
    APPLIANCES_SMALL = "appliances_small"
    CONTAINERS_KITCHEN = "containers_kitchen"

    # Grocery/Retail
    GROCERIES = "groceries"
    BOTTLES = "bottles"
    CANS = "cans"
    BOXES = "boxes"
    PRODUCE = "produce"
    REFRIGERATED = "refrigerated"

    # Warehouse
    PALLETS = "pallets"
    TOTES = "totes"
    CARTONS = "cartons"
    SHIPPING_BOXES = "shipping_boxes"

    # Lab
    LAB_EQUIPMENT = "lab_equipment"
    SAMPLE_CONTAINERS = "sample_containers"
    LAB_TOOLS = "lab_tools"
    SAFETY_EQUIPMENT = "safety_equipment"

    # Office
    OFFICE_SUPPLIES = "office_supplies"
    DOCUMENTS = "documents"
    ELECTRONICS = "electronics"

    # Home/Laundry
    CLOTHING = "clothing"
    LINENS = "linens"
    HAMPERS = "hampers"
    DETERGENT = "detergent"

    # Utility
    TOOLS = "tools"
    MAINTENANCE_SUPPLIES = "maintenance_supplies"

    # Generic
    MISC_OBJECTS = "misc_objects"


class RegionType(str, Enum):
    """Types of placement regions."""
    # Horizontal surfaces
    COUNTER = "counter"
    TABLE = "table"
    SHELF = "shelf"
    FLOOR = "floor"
    PREP_SURFACE = "prep_surface"
    SERVING_SURFACE = "serving_surface"

    # Storage
    DRAWER = "drawer"
    CABINET = "cabinet"
    REFRIGERATOR = "refrigerator"
    DISHWASHER = "dishwasher"
    OVEN = "oven"
    MICROWAVE = "microwave"
    WASHER = "washer"
    DRYER = "dryer"

    # Specialized
    SINK = "sink"
    DRYING_RACK = "drying_rack"
    DISH_PIT = "dish_pit"

    # Warehouse/Industrial
    PALLET_POSITION = "pallet_position"
    RACK_LEVEL = "rack_level"
    STAGING_AREA = "staging_area"
    CONVEYOR = "conveyor"

    # Lab
    BENCH = "bench"
    GLOVEBOX = "glovebox"
    FUME_HOOD = "fume_hood"
    INCUBATOR = "incubator"

    # Office
    DESK = "desk"
    FILING_CABINET = "filing_cabinet"

    # Utility
    CONTROL_PANEL = "control_panel"
    VALVE_BANK = "valve_bank"

    # Home
    FOLDING_TABLE = "folding_table"
    HAMPER_AREA = "hamper_area"
    CLOSET = "closet"


class ArticulationState(str, Enum):
    """Articulation states that affect placement."""
    OPEN = "open"
    CLOSED = "closed"
    PARTIALLY_OPEN = "partially_open"
    EXTENDED = "extended"
    RETRACTED = "retracted"


@dataclass
class PlacementContext:
    """Context for placement decisions.

    Contains all information needed to decide what can be placed where.
    """
    scene_archetype: SceneArchetype
    region_type: RegionType
    region_id: str
    articulation_state: Optional[ArticulationState] = None
    parent_object_id: Optional[str] = None
    semantic_tags: List[str] = field(default_factory=list)
    surface_area_m2: Optional[float] = None
    height_clearance_m: Optional[float] = None
    is_wet: bool = False
    is_heated: bool = False
    is_refrigerated: bool = False
    max_weight_kg: Optional[float] = None


@dataclass
class AssetPlacementRule:
    """Rule defining how an asset category can be placed in a region."""
    asset_category: AssetCategory
    region_types: List[RegionType]
    priority: int = 0  # Higher = more preferred
    min_clearance_m: float = 0.0
    max_stack_height: int = 1
    requires_articulation_state: Optional[ArticulationState] = None
    excluded_articulation_states: List[ArticulationState] = field(default_factory=list)
    semantic_requirements: List[str] = field(default_factory=list)
    semantic_exclusions: List[str] = field(default_factory=list)
    density_per_m2: float = 4.0  # Default items per square meter
    clustering_enabled: bool = True
    notes: str = ""


@dataclass
class CompatibilityEntry:
    """Entry in the compatibility matrix."""
    scene_archetype: SceneArchetype
    rules: List[AssetPlacementRule]


# =============================================================================
# COMPATIBILITY MATRIX DEFINITIONS
# =============================================================================

# Kitchen Scene Compatibility
KITCHEN_RULES: List[AssetPlacementRule] = [
    # Dishes
    AssetPlacementRule(
        asset_category=AssetCategory.DISHES,
        region_types=[RegionType.COUNTER, RegionType.TABLE, RegionType.PREP_SURFACE,
                      RegionType.SERVING_SURFACE, RegionType.DRYING_RACK],
        priority=10,
        max_stack_height=3,
        density_per_m2=8.0,
        notes="Dishes on surfaces, stacked neatly"
    ),
    AssetPlacementRule(
        asset_category=AssetCategory.DISHES,
        region_types=[RegionType.DISHWASHER],
        priority=20,  # Higher priority - dishes belong in dishwasher
        requires_articulation_state=ArticulationState.OPEN,
        excluded_articulation_states=[ArticulationState.CLOSED],
        density_per_m2=16.0,  # Densely packed in racks
        notes="Dishes in open dishwasher racks"
    ),
    AssetPlacementRule(
        asset_category=AssetCategory.DISHES,
        region_types=[RegionType.SINK, RegionType.DISH_PIT],
        priority=15,
        semantic_tags=["dirty_dishes", "to_wash"],
        density_per_m2=12.0,
        notes="Dirty dishes in sink/dish pit"
    ),
    AssetPlacementRule(
        asset_category=AssetCategory.DISHES,
        region_types=[RegionType.CABINET],
        priority=5,
        requires_articulation_state=ArticulationState.OPEN,
        max_stack_height=4,
        notes="Clean dishes stored in cabinets"
    ),
    AssetPlacementRule(
        asset_category=AssetCategory.DISHES,
        region_types=[RegionType.DRAWER],
        priority=0,  # Low priority - only shallow bowls in drawers
        requires_articulation_state=ArticulationState.OPEN,
        semantic_requirements=["shallow"],
        notes="Only shallow dishes in drawers"
    ),

    # Utensils
    AssetPlacementRule(
        asset_category=AssetCategory.UTENSILS,
        region_types=[RegionType.DRAWER],
        priority=20,  # Utensils primarily go in drawers
        requires_articulation_state=ArticulationState.OPEN,
        density_per_m2=30.0,  # Many utensils per area
        notes="Utensils organized in drawers"
    ),
    AssetPlacementRule(
        asset_category=AssetCategory.UTENSILS,
        region_types=[RegionType.COUNTER, RegionType.PREP_SURFACE],
        priority=10,
        semantic_tags=["utensil_holder", "prep_area"],
        density_per_m2=15.0,
        notes="Utensils in holders on counter"
    ),
    AssetPlacementRule(
        asset_category=AssetCategory.UTENSILS,
        region_types=[RegionType.DISHWASHER],
        priority=15,
        requires_articulation_state=ArticulationState.OPEN,
        notes="Utensils in dishwasher basket"
    ),

    # Cookware
    AssetPlacementRule(
        asset_category=AssetCategory.COOKWARE,
        region_types=[RegionType.CABINET],
        priority=15,
        requires_articulation_state=ArticulationState.OPEN,
        max_stack_height=3,
        notes="Pots/pans in cabinets"
    ),
    AssetPlacementRule(
        asset_category=AssetCategory.COOKWARE,
        region_types=[RegionType.COUNTER, RegionType.PREP_SURFACE],
        priority=10,
        max_stack_height=2,
        notes="Cookware on counters"
    ),
    AssetPlacementRule(
        asset_category=AssetCategory.COOKWARE,
        region_types=[RegionType.DRAWER],
        priority=5,
        requires_articulation_state=ArticulationState.OPEN,
        semantic_requirements=["deep_drawer"],
        notes="Only in deep pot drawers"
    ),

    # Food items
    AssetPlacementRule(
        asset_category=AssetCategory.FOOD_ITEMS,
        region_types=[RegionType.REFRIGERATOR],
        priority=20,
        requires_articulation_state=ArticulationState.OPEN,
        semantic_requirements=["perishable"],
        notes="Perishables in fridge"
    ),
    AssetPlacementRule(
        asset_category=AssetCategory.FOOD_ITEMS,
        region_types=[RegionType.COUNTER, RegionType.PREP_SURFACE],
        priority=15,
        semantic_tags=["prep_area", "cooking"],
        notes="Food being prepared"
    ),
    AssetPlacementRule(
        asset_category=AssetCategory.FOOD_ITEMS,
        region_types=[RegionType.CABINET],
        priority=10,
        requires_articulation_state=ArticulationState.OPEN,
        semantic_requirements=["non_perishable"],
        notes="Dry goods in cabinets"
    ),

    # Small appliances
    AssetPlacementRule(
        asset_category=AssetCategory.APPLIANCES_SMALL,
        region_types=[RegionType.COUNTER],
        priority=15,
        density_per_m2=2.0,  # Sparse - appliances are larger
        clustering_enabled=False,
        notes="Appliances spaced on counter"
    ),
    AssetPlacementRule(
        asset_category=AssetCategory.APPLIANCES_SMALL,
        region_types=[RegionType.CABINET],
        priority=5,
        requires_articulation_state=ArticulationState.OPEN,
        notes="Stored appliances"
    ),

    # Kitchen containers
    AssetPlacementRule(
        asset_category=AssetCategory.CONTAINERS_KITCHEN,
        region_types=[RegionType.CABINET, RegionType.DRAWER],
        priority=15,
        requires_articulation_state=ArticulationState.OPEN,
        max_stack_height=2,
        notes="Storage containers"
    ),
    AssetPlacementRule(
        asset_category=AssetCategory.CONTAINERS_KITCHEN,
        region_types=[RegionType.REFRIGERATOR],
        priority=10,
        requires_articulation_state=ArticulationState.OPEN,
        notes="Food storage in fridge"
    ),
]

# Grocery/Retail Scene Compatibility
GROCERY_RULES: List[AssetPlacementRule] = [
    AssetPlacementRule(
        asset_category=AssetCategory.GROCERIES,
        region_types=[RegionType.SHELF],
        priority=20,
        max_stack_height=2,
        density_per_m2=20.0,
        notes="Products on shelves"
    ),
    AssetPlacementRule(
        asset_category=AssetCategory.BOTTLES,
        region_types=[RegionType.SHELF],
        priority=15,
        max_stack_height=1,
        density_per_m2=15.0,
        notes="Bottles on shelves"
    ),
    AssetPlacementRule(
        asset_category=AssetCategory.BOTTLES,
        region_types=[RegionType.REFRIGERATOR],
        priority=20,
        requires_articulation_state=ArticulationState.OPEN,
        notes="Cold beverages"
    ),
    AssetPlacementRule(
        asset_category=AssetCategory.CANS,
        region_types=[RegionType.SHELF],
        priority=15,
        max_stack_height=3,
        density_per_m2=25.0,
        notes="Canned goods on shelves"
    ),
    AssetPlacementRule(
        asset_category=AssetCategory.BOXES,
        region_types=[RegionType.SHELF],
        priority=15,
        max_stack_height=2,
        density_per_m2=12.0,
        notes="Boxed products"
    ),
    AssetPlacementRule(
        asset_category=AssetCategory.PRODUCE,
        region_types=[RegionType.SHELF],
        priority=15,
        semantic_tags=["produce_section"],
        density_per_m2=30.0,
        notes="Fresh produce display"
    ),
    AssetPlacementRule(
        asset_category=AssetCategory.REFRIGERATED,
        region_types=[RegionType.REFRIGERATOR],
        priority=20,
        requires_articulation_state=ArticulationState.OPEN,
        notes="Refrigerated items"
    ),
    AssetPlacementRule(
        asset_category=AssetCategory.CARTONS,
        region_types=[RegionType.FLOOR, RegionType.STAGING_AREA],
        priority=10,
        max_stack_height=4,
        notes="Restock cartons"
    ),
]

# Warehouse Scene Compatibility
WAREHOUSE_RULES: List[AssetPlacementRule] = [
    AssetPlacementRule(
        asset_category=AssetCategory.PALLETS,
        region_types=[RegionType.PALLET_POSITION, RegionType.FLOOR],
        priority=20,
        max_stack_height=1,
        density_per_m2=0.8,
        notes="Pallets on floor positions"
    ),
    AssetPlacementRule(
        asset_category=AssetCategory.TOTES,
        region_types=[RegionType.RACK_LEVEL, RegionType.SHELF],
        priority=20,
        max_stack_height=2,
        density_per_m2=6.0,
        notes="Totes on racks"
    ),
    AssetPlacementRule(
        asset_category=AssetCategory.TOTES,
        region_types=[RegionType.CONVEYOR],
        priority=15,
        max_stack_height=1,
        density_per_m2=4.0,
        notes="Totes on conveyor"
    ),
    AssetPlacementRule(
        asset_category=AssetCategory.CARTONS,
        region_types=[RegionType.RACK_LEVEL, RegionType.PALLET_POSITION],
        priority=15,
        max_stack_height=6,
        density_per_m2=8.0,
        notes="Cartons stacked"
    ),
    AssetPlacementRule(
        asset_category=AssetCategory.SHIPPING_BOXES,
        region_types=[RegionType.STAGING_AREA, RegionType.PALLET_POSITION],
        priority=15,
        max_stack_height=5,
        density_per_m2=10.0,
        notes="Shipping boxes staged"
    ),
]

# Lab Scene Compatibility
LAB_RULES: List[AssetPlacementRule] = [
    AssetPlacementRule(
        asset_category=AssetCategory.LAB_EQUIPMENT,
        region_types=[RegionType.BENCH, RegionType.FUME_HOOD],
        priority=20,
        density_per_m2=3.0,
        clustering_enabled=False,
        notes="Equipment on benches"
    ),
    AssetPlacementRule(
        asset_category=AssetCategory.SAMPLE_CONTAINERS,
        region_types=[RegionType.BENCH, RegionType.GLOVEBOX],
        priority=15,
        max_stack_height=1,
        density_per_m2=20.0,
        notes="Sample containers"
    ),
    AssetPlacementRule(
        asset_category=AssetCategory.SAMPLE_CONTAINERS,
        region_types=[RegionType.INCUBATOR, RegionType.REFRIGERATOR],
        priority=20,
        requires_articulation_state=ArticulationState.OPEN,
        notes="Samples in storage"
    ),
    AssetPlacementRule(
        asset_category=AssetCategory.LAB_TOOLS,
        region_types=[RegionType.BENCH, RegionType.DRAWER],
        priority=15,
        density_per_m2=10.0,
        notes="Lab tools"
    ),
    AssetPlacementRule(
        asset_category=AssetCategory.SAFETY_EQUIPMENT,
        region_types=[RegionType.BENCH, RegionType.CABINET],
        priority=10,
        notes="Safety gear"
    ),
]

# Office Scene Compatibility
OFFICE_RULES: List[AssetPlacementRule] = [
    AssetPlacementRule(
        asset_category=AssetCategory.OFFICE_SUPPLIES,
        region_types=[RegionType.DESK, RegionType.DRAWER],
        priority=15,
        density_per_m2=8.0,
        notes="Office supplies"
    ),
    AssetPlacementRule(
        asset_category=AssetCategory.DOCUMENTS,
        region_types=[RegionType.DESK, RegionType.FILING_CABINET],
        priority=15,
        max_stack_height=5,
        notes="Documents and papers"
    ),
    AssetPlacementRule(
        asset_category=AssetCategory.ELECTRONICS,
        region_types=[RegionType.DESK],
        priority=20,
        density_per_m2=2.0,
        clustering_enabled=False,
        notes="Electronics on desk"
    ),
]

# Utility Room Scene Compatibility
UTILITY_RULES: List[AssetPlacementRule] = [
    AssetPlacementRule(
        asset_category=AssetCategory.TOOLS,
        region_types=[RegionType.SHELF, RegionType.CABINET],
        priority=15,
        notes="Tools stored"
    ),
    AssetPlacementRule(
        asset_category=AssetCategory.MAINTENANCE_SUPPLIES,
        region_types=[RegionType.SHELF, RegionType.FLOOR],
        priority=10,
        notes="Maintenance supplies"
    ),
]

# Home Laundry Scene Compatibility
LAUNDRY_RULES: List[AssetPlacementRule] = [
    AssetPlacementRule(
        asset_category=AssetCategory.CLOTHING,
        region_types=[RegionType.WASHER],
        priority=20,
        requires_articulation_state=ArticulationState.OPEN,
        semantic_tags=["dirty", "to_wash"],
        notes="Clothes in open washer"
    ),
    AssetPlacementRule(
        asset_category=AssetCategory.CLOTHING,
        region_types=[RegionType.DRYER],
        priority=20,
        requires_articulation_state=ArticulationState.OPEN,
        semantic_tags=["wet", "to_dry"],
        notes="Clothes in open dryer"
    ),
    AssetPlacementRule(
        asset_category=AssetCategory.CLOTHING,
        region_types=[RegionType.FOLDING_TABLE],
        priority=15,
        semantic_tags=["clean", "to_fold"],
        notes="Clean clothes on folding table"
    ),
    AssetPlacementRule(
        asset_category=AssetCategory.CLOTHING,
        region_types=[RegionType.HAMPER_AREA],
        priority=15,
        semantic_tags=["dirty"],
        notes="Dirty clothes in hamper"
    ),
    AssetPlacementRule(
        asset_category=AssetCategory.CLOTHING,
        region_types=[RegionType.CLOSET],
        priority=10,
        requires_articulation_state=ArticulationState.OPEN,
        semantic_tags=["clean", "stored"],
        notes="Clean clothes in closet"
    ),
    AssetPlacementRule(
        asset_category=AssetCategory.LINENS,
        region_types=[RegionType.SHELF, RegionType.CLOSET],
        priority=15,
        max_stack_height=4,
        notes="Linens stored"
    ),
    AssetPlacementRule(
        asset_category=AssetCategory.DETERGENT,
        region_types=[RegionType.SHELF],
        priority=10,
        notes="Laundry supplies"
    ),
]


# =============================================================================
# COMPATIBILITY MATRIX CLASS
# =============================================================================

class CompatibilityMatrix:
    """Central compatibility matrix for scene-to-asset mapping.

    This class provides intelligent querying of what assets can be placed
    where, based on scene type, region type, articulation state, and
    semantic context.
    """

    def __init__(self):
        """Initialize with all rule sets."""
        self._rules: Dict[SceneArchetype, List[AssetPlacementRule]] = {
            SceneArchetype.KITCHEN: KITCHEN_RULES,
            SceneArchetype.GROCERY: GROCERY_RULES,
            SceneArchetype.WAREHOUSE: WAREHOUSE_RULES,
            SceneArchetype.LAB: LAB_RULES,
            SceneArchetype.OFFICE: OFFICE_RULES,
            SceneArchetype.UTILITY_ROOM: UTILITY_RULES,
            SceneArchetype.HOME_LAUNDRY: LAUNDRY_RULES,
        }

        # Build reverse lookup indices
        self._asset_to_regions: Dict[Tuple[SceneArchetype, AssetCategory], List[RegionType]] = {}
        self._region_to_assets: Dict[Tuple[SceneArchetype, RegionType], List[AssetCategory]] = {}
        self._build_indices()

    def _build_indices(self) -> None:
        """Build reverse lookup indices for fast querying."""
        for archetype, rules in self._rules.items():
            for rule in rules:
                # Asset -> Regions
                key = (archetype, rule.asset_category)
                if key not in self._asset_to_regions:
                    self._asset_to_regions[key] = []
                self._asset_to_regions[key].extend(rule.region_types)

                # Region -> Assets
                for region_type in rule.region_types:
                    key = (archetype, region_type)
                    if key not in self._region_to_assets:
                        self._region_to_assets[key] = []
                    if rule.asset_category not in self._region_to_assets[key]:
                        self._region_to_assets[key].append(rule.asset_category)

    def get_compatible_assets(
        self,
        context: PlacementContext,
        filter_by_articulation: bool = True,
    ) -> List[Tuple[AssetCategory, AssetPlacementRule]]:
        """Get all asset categories compatible with a placement context.

        Args:
            context: The placement context to query
            filter_by_articulation: Whether to filter by articulation state

        Returns:
            List of (asset_category, rule) tuples sorted by priority (descending)
        """
        archetype = context.scene_archetype
        rules = self._rules.get(archetype, [])

        compatible: List[Tuple[AssetCategory, AssetPlacementRule]] = []

        for rule in rules:
            # Check region type match
            if context.region_type not in rule.region_types:
                continue

            # Check articulation state requirements
            if filter_by_articulation:
                if rule.requires_articulation_state and context.articulation_state:
                    if rule.requires_articulation_state != context.articulation_state:
                        continue

                if context.articulation_state in rule.excluded_articulation_states:
                    continue

            # Check semantic requirements
            if rule.semantic_requirements:
                if not any(tag in context.semantic_tags for tag in rule.semantic_requirements):
                    continue

            # Check semantic exclusions
            if rule.semantic_exclusions:
                if any(tag in context.semantic_tags for tag in rule.semantic_exclusions):
                    continue

            compatible.append((rule.asset_category, rule))

        # Sort by priority (descending)
        compatible.sort(key=lambda x: x[1].priority, reverse=True)
        return compatible

    def get_suitable_regions(
        self,
        scene_archetype: SceneArchetype,
        asset_category: AssetCategory,
    ) -> List[RegionType]:
        """Get all region types suitable for an asset category.

        Args:
            scene_archetype: The scene archetype
            asset_category: The asset category to place

        Returns:
            List of compatible region types
        """
        key = (scene_archetype, asset_category)
        return list(set(self._asset_to_regions.get(key, [])))

    def get_placement_rule(
        self,
        scene_archetype: SceneArchetype,
        asset_category: AssetCategory,
        region_type: RegionType,
    ) -> Optional[AssetPlacementRule]:
        """Get the specific placement rule for an asset-region combination.

        Args:
            scene_archetype: The scene archetype
            asset_category: The asset category
            region_type: The region type

        Returns:
            The matching rule, or None if no match
        """
        rules = self._rules.get(scene_archetype, [])

        for rule in rules:
            if rule.asset_category == asset_category and region_type in rule.region_types:
                return rule

        return None

    def get_density_for_placement(
        self,
        context: PlacementContext,
        asset_category: AssetCategory,
    ) -> float:
        """Get the placement density for an asset in a context.

        Args:
            context: The placement context
            asset_category: The asset category

        Returns:
            Items per square meter
        """
        rule = self.get_placement_rule(
            context.scene_archetype,
            asset_category,
            context.region_type,
        )
        return rule.density_per_m2 if rule else 4.0  # Default

    def get_max_stack_height(
        self,
        context: PlacementContext,
        asset_category: AssetCategory,
    ) -> int:
        """Get the maximum stack height for an asset in a context.

        Args:
            context: The placement context
            asset_category: The asset category

        Returns:
            Maximum number of stacked items
        """
        rule = self.get_placement_rule(
            context.scene_archetype,
            asset_category,
            context.region_type,
        )
        return rule.max_stack_height if rule else 1

    def should_cluster(
        self,
        context: PlacementContext,
        asset_category: AssetCategory,
    ) -> bool:
        """Check if assets should be clustered together.

        Args:
            context: The placement context
            asset_category: The asset category

        Returns:
            True if clustering is enabled
        """
        rule = self.get_placement_rule(
            context.scene_archetype,
            asset_category,
            context.region_type,
        )
        return rule.clustering_enabled if rule else True

    def get_all_archetypes(self) -> List[SceneArchetype]:
        """Get all defined scene archetypes."""
        return list(self._rules.keys())

    def get_rules_for_archetype(
        self,
        archetype: SceneArchetype,
    ) -> List[AssetPlacementRule]:
        """Get all placement rules for a scene archetype."""
        return self._rules.get(archetype, [])

    def to_dict(self) -> Dict[str, Any]:
        """Export the matrix as a dictionary for serialization."""
        result = {}
        for archetype, rules in self._rules.items():
            result[archetype.value] = [
                {
                    "asset_category": rule.asset_category.value,
                    "region_types": [rt.value for rt in rule.region_types],
                    "priority": rule.priority,
                    "min_clearance_m": rule.min_clearance_m,
                    "max_stack_height": rule.max_stack_height,
                    "requires_articulation_state": (
                        rule.requires_articulation_state.value
                        if rule.requires_articulation_state else None
                    ),
                    "density_per_m2": rule.density_per_m2,
                    "clustering_enabled": rule.clustering_enabled,
                    "notes": rule.notes,
                }
                for rule in rules
            ]
        return result


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

# Global singleton instance
_matrix: Optional[CompatibilityMatrix] = None


def get_compatibility_matrix() -> CompatibilityMatrix:
    """Get the global compatibility matrix instance."""
    global _matrix
    if _matrix is None:
        _matrix = CompatibilityMatrix()
    return _matrix


def get_compatible_assets(
    context: PlacementContext,
) -> List[Tuple[AssetCategory, AssetPlacementRule]]:
    """Get compatible assets for a placement context."""
    return get_compatibility_matrix().get_compatible_assets(context)


def get_suitable_regions(
    scene_archetype: SceneArchetype,
    asset_category: AssetCategory,
) -> List[RegionType]:
    """Get suitable regions for an asset category."""
    return get_compatibility_matrix().get_suitable_regions(scene_archetype, asset_category)


if __name__ == "__main__":
    # Test the compatibility matrix
    matrix = CompatibilityMatrix()

    # Test: What can go on an open dishwasher?
    context = PlacementContext(
        scene_archetype=SceneArchetype.KITCHEN,
        region_type=RegionType.DISHWASHER,
        region_id="dishwasher_01",
        articulation_state=ArticulationState.OPEN,
    )

    compatible = matrix.get_compatible_assets(context)
    print("Assets compatible with open dishwasher:")
    for asset_cat, rule in compatible:
        print(f"  - {asset_cat.value} (priority: {rule.priority})")

    # Test: Where can dishes go?
    regions = matrix.get_suitable_regions(SceneArchetype.KITCHEN, AssetCategory.DISHES)
    print(f"\nRegions suitable for dishes in kitchen: {[r.value for r in regions]}")
