"""
USD Scene Builder - Handles USD stage creation and manipulation.

This is a lightly adapted copy from BlueprintRecipe that can operate in either
full OpenUSD mode (when pxr is available) or stub mode for environments without
USD tooling.
"""

import json
from pathlib import Path
from typing import Any, Optional


class USDSceneBuilder:
    """
    Builder for constructing USD stages and prims.

    This class abstracts USD operations and can work with:
    - Full pxr (OpenUSD) when available (Isaac Sim, standalone USD)
    - A stub mode for testing/planning without USD runtime
    """

    def __init__(self, meters_per_unit: float = 1.0, up_axis: str = "Y"):
        self.meters_per_unit = meters_per_unit
        self.up_axis = up_axis
        self._has_usd = self._check_usd_available()
        self.asset_catalog = self._load_asset_catalog()
        self._catalog_pack_name = (self.asset_catalog or {}).get("pack_info", {}).get("name")
        self._catalog_asset_paths = {
            asset.get("relative_path")
            for asset in self.asset_catalog.get("assets", [])
            if asset.get("relative_path")
        }

        if self._has_usd:
            from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics

            self.Gf = Gf
            self.Usd = Usd
            self.UsdGeom = UsdGeom
            self.Sdf = Sdf
            self.UsdPhysics = UsdPhysics

    def _check_usd_available(self) -> bool:
        """Check if OpenUSD (pxr) is available."""
        try:
            from pxr import Usd  # noqa: F401

            return True
        except ImportError:
            return False

    def create_stage(self, path: str) -> Any:
        """Create a new USD stage."""
        if self._has_usd:
            stage = self.Usd.Stage.CreateNew(path)
            self.UsdGeom.SetStageUpAxis(
                stage, self.UsdGeom.Tokens.y if self.up_axis == "Y" else self.UsdGeom.Tokens.z
            )
            self.UsdGeom.SetStageMetersPerUnit(stage, self.meters_per_unit)
            return stage
        else:
            return StubStage(path)

    def set_stage_metadata(
        self, stage: Any, up_axis: str = "Y", meters_per_unit: float = 1.0, doc: Optional[str] = None
    ) -> None:
        """Set stage-level metadata."""
        if self._has_usd:
            if doc:
                stage.SetDocumentation(doc)
        else:
            stage.metadata["up_axis"] = up_axis
            stage.metadata["meters_per_unit"] = meters_per_unit
            stage.metadata["doc"] = doc

    def add_sublayer(self, stage: Any, layer_path: str) -> None:
        """Add a sublayer reference to the stage."""
        if self._has_usd:
            root_layer = stage.GetRootLayer()
            root_layer.subLayerPaths.append(layer_path)
        else:
            stage.sublayers.append(layer_path)

    def build_room_shell(self, layer: Any, room_config: dict[str, Any]) -> None:
        """Build room shell geometry (floor, walls, ceiling)."""
        dims = room_config.get("dimensions", {})
        width = dims.get("width", 5.0)
        depth = dims.get("depth", 5.0)
        height = dims.get("height", 3.0)

        if self._has_usd:
            stage = layer.stage
            room_prim = self.UsdGeom.Xform.Define(stage, "/Room")

            floor = self.UsdGeom.Mesh.Define(stage, "/Room/Floor")
            floor.CreatePointsAttr(
                [(-width / 2, 0, -depth / 2), (width / 2, 0, -depth / 2), (width / 2, 0, depth / 2), (-width / 2, 0, depth / 2)]
            )
            floor.CreateFaceVertexCountsAttr([4])
            floor.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
            self.UsdPhysics.CollisionAPI.Apply(floor.GetPrim())
        else:
            layer.add_prim("/Room", "Xform")
            layer.add_prim(
                "/Room/Floor", "Mesh", {"width": width, "depth": depth, "is_ground": True}
            )

    def build_layout(self, layer: Any, objects: list[dict[str, Any]], asset_root: str) -> None:
        """Build object layout with references to assets."""
        if self._has_usd:
            stage = layer.stage
            self.UsdGeom.Xform.Define(stage, "/Objects")

            for obj in objects:
                obj_id = obj.get("id", "unknown")
                prim_path = f"/Objects/{obj_id}"
                obj_xform = self.UsdGeom.Xform.Define(stage, prim_path)

                transform = obj.get("transform", {})
                pos = transform.get("position", {})
                rot = transform.get("rotation", {})
                scale = transform.get("scale", {})

                obj_xform.AddTranslateOp().Set((pos.get("x", 0), pos.get("y", 0), pos.get("z", 0)))
                obj_xform.AddOrientOp().Set(
                    self.Sdf.Quatd(rot.get("w", 1), rot.get("x", 0), rot.get("y", 0), rot.get("z", 0))
                )
                obj_xform.AddScaleOp().Set((scale.get("x", 1), scale.get("y", 1), scale.get("z", 1)))

                chosen_asset = obj.get("chosen_asset", {})
                asset_path = chosen_asset.get("asset_path", "")
                if asset_path:
                    self._validate_catalog_path(asset_path)
                    full_path = self._resolve_asset_path(asset_path, asset_root)
                    obj_xform.GetPrim().GetReferences().AddReference(full_path)
        else:
            layer.add_prim("/Objects", "Xform")
            for obj in objects:
                chosen_asset = obj.get("chosen_asset", {})
                asset_path = chosen_asset.get("asset_path", "")
                if asset_path:
                    self._validate_catalog_path(asset_path)
                layer.add_prim(
                    f"/Objects/{obj.get('id')}",
                    "Xform",
                    {"transform": obj.get("transform"), "asset_ref": obj.get("chosen_asset", {}).get("asset_path")},
                )

    def build_semantics(self, layer: Any, objects: list[dict[str, Any]]) -> None:
        """Build semantic labels for objects."""
        if self._has_usd:
            stage = layer.stage

            for obj in objects:
                obj_id = obj.get("id", "unknown")
                prim_path = f"/Objects/{obj_id}"
                prim = stage.OverridePrim(prim_path)

                semantics = obj.get("semantics", {})

                if semantics.get("class"):
                    prim.CreateAttribute("semantic:class", self.Sdf.ValueTypeNames.String).Set(semantics["class"])

                if semantics.get("instance_id"):
                    prim.CreateAttribute("semantic:instance", self.Sdf.ValueTypeNames.String).Set(semantics["instance_id"])

                affordances = semantics.get("affordances", [])
                if affordances:
                    prim.CreateAttribute("semantic:affordances", self.Sdf.ValueTypeNames.StringArray).Set(affordances)
        else:
            for obj in objects:
                layer.add_override(
                    f"/Objects/{obj.get('id')}",
                    {
                        "semantic:class": obj.get("semantics", {}).get("class"),
                        "semantic:instance": obj.get("semantics", {}).get("instance_id"),
                        "semantic:affordances": obj.get("semantics", {}).get("affordances", []),
                    },
                )

    def build_physics_overrides(self, layer: Any, objects: list[dict[str, Any]]) -> None:
        """Build physics property overrides for objects."""
        if self._has_usd:
            stage = layer.stage

            for obj in objects:
                obj_id = obj.get("id", "unknown")
                prim_path = f"/Objects/{obj_id}"

                physics = obj.get("physics", {})
                if not physics.get("enabled", True):
                    continue

                prim = stage.OverridePrim(prim_path)

                if physics.get("collision_enabled", True):
                    self.UsdPhysics.CollisionAPI.Apply(prim)

                if physics.get("rigid_body", False):
                    self.UsdPhysics.RigidBodyAPI.Apply(prim)

                    # GAP-PHYSICS-007 FIX: Add kinematic enabled flag for Isaac Sim
                    if physics.get("kinematic_enabled", False):
                        prim.CreateAttribute("physics:kinematicEnabled", self.Sdf.ValueTypeNames.Bool).Set(True)

                    if "mass_override" in physics:
                        mass_api = self.UsdPhysics.MassAPI.Apply(prim)
                        mass_api.CreateMassAttr().Set(physics["mass_override"])

                        if "center_of_mass_offset" in physics:
                            offset = physics.get("center_of_mass_offset", [0, 0, 0])
                            mass_api.CreateCenterOfMassAttr().Set(self.Gf.Vec3f(*offset))

                        if "inertia_diagonal" in physics:
                            inertia = physics.get("inertia_diagonal")
                            prim.CreateAttribute("physics:diagonalInertia", self.Sdf.ValueTypeNames.Double3).Set(tuple(inertia))

                if "friction_static" in physics:
                    prim.CreateAttribute("physics:staticFriction", self.Sdf.ValueTypeNames.Float).Set(
                        float(physics["friction_static"])
                    )
                if "friction_dynamic" in physics:
                    prim.CreateAttribute("physics:dynamicFriction", self.Sdf.ValueTypeNames.Float).Set(
                        float(physics["friction_dynamic"])
                    )
                if "restitution" in physics:
                    prim.CreateAttribute("physics:restitution", self.Sdf.ValueTypeNames.Float).Set(
                        float(physics["restitution"])
                    )

                # GAP-PHYSICS-007 FIX: Add physics:approximation attribute for Isaac Sim
                if physics.get("collision_approximation"):
                    approximation = str(physics["collision_approximation"])
                    # Add both physxCollision:approximation and physics:approximation for compatibility
                    prim.CreateAttribute("physxCollision:approximation", self.Sdf.ValueTypeNames.Token).Set(approximation)
                    prim.CreateAttribute("physics:approximation", self.Sdf.ValueTypeNames.Token).Set(approximation)

                # GAP-PHYSICS-007 FIX: Add GPU collision support
                if physics.get("gpu_collision", True):
                    prim.CreateAttribute("physxCollision:gpuCollision", self.Sdf.ValueTypeNames.Bool).Set(True)

                articulation = obj.get("articulation")
                if articulation:
                    self._add_articulation(stage, prim_path, articulation)
        else:
            for obj in objects:
                layer.add_override(
                    f"/Objects/{obj.get('id')}",
                    {"physics": obj.get("physics", {}), "articulation": obj.get("articulation")},
                )

    def _add_articulation(self, stage: Any, prim_path: str, articulation: dict[str, Any]) -> None:
        """Add articulation (joint) to an object."""
        if not self._has_usd:
            return

        joint_type = articulation.get("type", "revolute")
        axis = (articulation.get("axis") or "y").upper()
        limits = articulation.get("limits", {})
        damping = articulation.get("damping")
        stiffness = articulation.get("stiffness")
        friction = articulation.get("friction")
        velocity_limit = articulation.get("velocity_limit")
        effort_limit = articulation.get("effort_limit")

        joint_path = f"{prim_path}/joint"

        drive_type = "angular"
        if joint_type == "revolute":
            joint = self.UsdPhysics.RevoluteJoint.Define(stage, joint_path)
            joint.CreateAxisAttr().Set(axis)
            if limits:
                lower = limits.get("lower")
                upper = limits.get("upper")
                if lower is not None:
                    joint.CreateLowerLimitAttr().Set(lower)
                if upper is not None:
                    joint.CreateUpperLimitAttr().Set(upper)
        elif joint_type == "prismatic":
            joint = self.UsdPhysics.PrismaticJoint.Define(stage, joint_path)
            joint.CreateAxisAttr().Set(axis)
            if limits:
                lower = limits.get("lower")
                upper = limits.get("upper")
                if lower is not None:
                    joint.CreateLowerLimitAttr().Set(lower)
                if upper is not None:
                    joint.CreateUpperLimitAttr().Set(upper)
            drive_type = "linear"
        else:
            joint = self.UsdPhysics.RevoluteJoint.Define(stage, joint_path)
            joint.CreateAxisAttr().Set(axis)

        joint_prim = joint.GetPrim()
        drive_api = self.UsdPhysics.DriveAPI.Apply(joint_prim, drive_type)
        if damping is not None:
            drive_api.CreateDampingAttr().Set(float(damping))
        if stiffness is not None:
            drive_api.CreateStiffnessAttr().Set(float(stiffness))
        if effort_limit is not None:
            drive_api.CreateMaxForceAttr().Set(float(effort_limit))
        if friction is not None:
            joint_prim.CreateAttribute(
                "physxJoint:friction", self.Sdf.ValueTypeNames.Float
            ).Set(float(friction))
        if velocity_limit is not None:
            joint_prim.CreateAttribute(
                "physxJoint:velocityLimit", self.Sdf.ValueTypeNames.Float
            ).Set(float(velocity_limit))

    def _resolve_asset_path(self, asset_path: str, asset_root: str) -> Any:
        """Resolve an asset path relative to asset root."""
        resolved_path = self._compute_asset_path(asset_path, asset_root)

        if self._has_usd:
            return self.Sdf.AssetPath(resolved_path)

        return f"@{resolved_path}@"

    def _compute_asset_path(self, asset_path: str, asset_root: str) -> str:
        """Compute the file system path for an asset."""
        if asset_path.startswith(("./", "../")) or asset_path.startswith("/"):
            return asset_path

        normalized_path = self._normalize_catalog_path(asset_path)
        if self._catalog_pack_name:
            return str(Path(asset_root) / self._catalog_pack_name / normalized_path)

        return str(Path(asset_root) / normalized_path)

    def _validate_catalog_path(self, asset_path: str) -> None:
        """Ensure the asset path exists in the loaded catalog."""
        normalized_path = self._normalize_catalog_path(asset_path)

        if not self._catalog_asset_paths:
            raise RuntimeError("Asset catalog is not loaded; cannot validate asset paths")
        if normalized_path not in self._catalog_asset_paths:
            raise ValueError(f"Asset path '{normalized_path}' not found in catalog")

    def _normalize_catalog_path(self, asset_path: str) -> str:
        """Strip pack name prefix if present."""
        if self._catalog_pack_name and asset_path.startswith(f"{self._catalog_pack_name}/"):
            return asset_path.split("/", 1)[1]
        return asset_path

    def _load_asset_catalog(self) -> dict[str, Any]:
        """Load the asset_index.json for reference validation."""
        catalog_path = Path(__file__).resolve().parents[2] / "asset_index.json"
        if not catalog_path.exists():
            return {}

        try:
            with catalog_path.open("r", encoding="utf-8") as catalog_file:
                return json.load(catalog_file)
        except (OSError, json.JSONDecodeError):
            return {}


class StubStage:
    """Stub stage for use when OpenUSD is not available."""

    def __init__(self, path: str):
        self.path = path
        self.prims = {}
        self.sublayers = []
        self.metadata = {}

    def Save(self) -> None:
        """Stub save - writes a placeholder USDA file."""
        sublayers = ", ".join(f"@{s}@" for s in self.sublayers)
        sublayer_block = ""
        if self.sublayers:
            sublayer_block = f"(\n    subLayers = [{sublayers}]\n)\n"

        content = f"""#usda 1.0
(
    doc = \"BlueprintRecipe stub stage - requires OpenUSD runtime for full functionality\"
    metersPerUnit = {self.metadata.get('meters_per_unit', 1.0)}
    upAxis = \"{self.metadata.get('up_axis', 'Y')}\"
)
{sublayer_block}"""

        with open(self.path, "w", encoding="utf-8") as f:
            f.write(content)


class StubLayer:
    """Stub layer for use when OpenUSD is not available."""

    def __init__(self, name: str, path: str):
        self.name = name
        self.path = path
        self.prims = {}
        self.overrides = {}
        self.stage = StubStage(path)

    def add_prim(self, path: str, prim_type: str, data: Optional[dict] = None) -> None:
        self.prims[path] = {"type": prim_type, "data": data or {}}

    def add_override(self, path: str, data: dict) -> None:
        self.overrides[path] = data

    def save(self) -> None:
        """Save stub layer as placeholder USDA."""
        content = f"""#usda 1.0
(
    doc = \"BlueprintRecipe stub layer: {self.name}\"
)

# Prims: {len(self.prims)}
# Overrides: {len(self.overrides)}
"""
        with open(self.path, "w", encoding="utf-8") as f:
            f.write(content)


__all__ = ["USDSceneBuilder", "StubStage", "StubLayer"]
