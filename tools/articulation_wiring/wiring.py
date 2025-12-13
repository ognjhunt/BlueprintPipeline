"""Articulation Wiring Implementation.

Wires interactive-job outputs (URDF/USD articulated assets) into the final scene.usda.
"""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import os
import sys


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class JointInfo:
    """Information about an articulated joint."""
    name: str
    joint_type: str  # "revolute", "prismatic", "fixed", "continuous"
    axis: List[float] = field(default_factory=lambda: [0, 0, 1])
    limits: Optional[Dict[str, float]] = None  # {"lower": 0, "upper": 1.57}
    parent_link: Optional[str] = None
    child_link: Optional[str] = None
    damping: float = 0.1
    friction: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "joint_type": self.joint_type,
            "axis": self.axis,
            "limits": self.limits,
            "parent_link": self.parent_link,
            "child_link": self.child_link,
            "damping": self.damping,
            "friction": self.friction,
        }


@dataclass
class ArticulatedAsset:
    """An articulated asset from interactive-job."""
    object_id: str
    urdf_path: Optional[Path] = None
    usd_path: Optional[Path] = None
    mesh_path: Optional[Path] = None
    joints: List[JointInfo] = field(default_factory=list)
    links: List[str] = field(default_factory=list)
    is_articulated: bool = False
    manifest_path: Optional[Path] = None

    @property
    def best_asset_path(self) -> Optional[Path]:
        """Return the best available asset path (prefer USD over URDF)."""
        if self.usd_path and self.usd_path.is_file():
            return self.usd_path
        if self.urdf_path and self.urdf_path.is_file():
            return self.urdf_path
        return self.mesh_path

    def to_dict(self) -> Dict[str, Any]:
        return {
            "object_id": self.object_id,
            "urdf_path": str(self.urdf_path) if self.urdf_path else None,
            "usd_path": str(self.usd_path) if self.usd_path else None,
            "mesh_path": str(self.mesh_path) if self.mesh_path else None,
            "joints": [j.to_dict() for j in self.joints],
            "links": self.links,
            "is_articulated": self.is_articulated,
        }


# =============================================================================
# URDF Parsing
# =============================================================================


def parse_urdf(urdf_path: Path) -> Tuple[List[JointInfo], List[str]]:
    """Parse URDF file to extract joint and link information.

    Args:
        urdf_path: Path to URDF file

    Returns:
        Tuple of (joints list, links list)
    """
    joints: List[JointInfo] = []
    links: List[str] = []

    try:
        tree = ET.parse(urdf_path)
        root = tree.getroot()
    except Exception as e:
        print(f"[ARTICULATION] Failed to parse URDF {urdf_path}: {e}")
        return joints, links

    # Extract links
    for link in root.findall("link"):
        name = link.attrib.get("name")
        if name:
            links.append(name)

    # Extract joints
    for joint in root.findall("joint"):
        joint_name = joint.attrib.get("name", "")
        joint_type = joint.attrib.get("type", "fixed")

        # Skip fixed joints (not articulated)
        if joint_type == "fixed":
            continue

        # Get axis
        axis = [0, 0, 1]  # default Z-axis
        axis_elem = joint.find("axis")
        if axis_elem is not None:
            xyz = axis_elem.attrib.get("xyz", "0 0 1")
            try:
                axis = [float(x) for x in xyz.split()]
            except ValueError:
                pass

        # Get limits
        limits = None
        limit_elem = joint.find("limit")
        if limit_elem is not None:
            limits = {}
            for attr in ["lower", "upper", "effort", "velocity"]:
                val = limit_elem.attrib.get(attr)
                if val:
                    try:
                        limits[attr] = float(val)
                    except ValueError:
                        pass

        # Get parent/child
        parent_link = None
        child_link = None
        parent_elem = joint.find("parent")
        child_elem = joint.find("child")
        if parent_elem is not None:
            parent_link = parent_elem.attrib.get("link")
        if child_elem is not None:
            child_link = child_elem.attrib.get("link")

        # Get dynamics
        damping = 0.1
        friction = 0.0
        dynamics_elem = joint.find("dynamics")
        if dynamics_elem is not None:
            try:
                damping = float(dynamics_elem.attrib.get("damping", "0.1"))
            except ValueError:
                pass
            try:
                friction = float(dynamics_elem.attrib.get("friction", "0"))
            except ValueError:
                pass

        joint_info = JointInfo(
            name=joint_name,
            joint_type=joint_type,
            axis=axis,
            limits=limits,
            parent_link=parent_link,
            child_link=child_link,
            damping=damping,
            friction=friction,
        )
        joints.append(joint_info)

    return joints, links


# =============================================================================
# Asset Discovery
# =============================================================================


def find_articulated_assets(
    root: Path,
    assets_prefix: str,
    scene_assets: Optional[Dict] = None,
) -> List[ArticulatedAsset]:
    """Find all articulated assets from interactive-job outputs.

    Args:
        root: GCS root path
        assets_prefix: Path prefix for assets
        scene_assets: Optional scene_assets.json data

    Returns:
        List of ArticulatedAsset objects
    """
    articulated_assets: List[ArticulatedAsset] = []

    # Primary location: assets/interactive/obj_*/
    interactive_dir = root / assets_prefix / "interactive"
    if interactive_dir.is_dir():
        for obj_dir in interactive_dir.glob("obj_*"):
            asset = _load_articulated_asset_from_dir(obj_dir)
            if asset:
                articulated_assets.append(asset)

    # Also check for interactive objects in scene_assets
    if scene_assets:
        for obj in scene_assets.get("objects", []):
            if obj.get("type") != "interactive":
                continue

            obj_id = str(obj.get("id"))
            interactive_output = obj.get("interactive_output")

            # Check if we already found this asset
            if any(a.object_id == obj_id for a in articulated_assets):
                continue

            # Try to find from interactive_output path
            if interactive_output:
                obj_dir = root / interactive_output
                if obj_dir.is_dir():
                    asset = _load_articulated_asset_from_dir(obj_dir, obj_id)
                    if asset:
                        articulated_assets.append(asset)
                        continue

            # Try default location
            obj_dir = interactive_dir / f"obj_{obj_id}"
            if obj_dir.is_dir():
                asset = _load_articulated_asset_from_dir(obj_dir, obj_id)
                if asset:
                    articulated_assets.append(asset)

    print(f"[ARTICULATION] Found {len(articulated_assets)} articulated assets")
    for asset in articulated_assets:
        print(f"[ARTICULATION]   obj_{asset.object_id}: {len(asset.joints)} joints")

    return articulated_assets


def _load_articulated_asset_from_dir(
    obj_dir: Path,
    obj_id: Optional[str] = None,
) -> Optional[ArticulatedAsset]:
    """Load articulated asset from an object directory."""
    if not obj_dir.is_dir():
        return None

    obj_id = obj_id or obj_dir.name.replace("obj_", "")

    # Find URDF
    urdf_path = None
    for name in [f"{obj_id}.urdf", f"obj_{obj_id}.urdf", "robot.urdf", "model.urdf"]:
        candidate = obj_dir / name
        if candidate.is_file():
            urdf_path = candidate
            break

    if not urdf_path:
        for candidate in obj_dir.glob("*.urdf"):
            urdf_path = candidate
            break

    # Find USD (if URDF was already converted)
    usd_path = None
    for name in ["articulated.usda", "articulated.usd", f"{obj_id}.usda", "model.usda"]:
        candidate = obj_dir / name
        if candidate.is_file():
            usd_path = candidate
            break

    # Find mesh
    mesh_path = None
    for name in ["part.glb", "mesh.glb", "model.glb"]:
        candidate = obj_dir / name
        if candidate.is_file():
            mesh_path = candidate
            break

    # Load manifest
    manifest_path = obj_dir / "interactive_manifest.json"
    manifest_data = None
    if manifest_path.is_file():
        try:
            manifest_data = json.loads(manifest_path.read_text())
        except Exception:
            pass

    # Parse URDF for joint info
    joints: List[JointInfo] = []
    links: List[str] = []
    if urdf_path:
        joints, links = parse_urdf(urdf_path)

    # Also check manifest for pre-parsed joint info
    if manifest_data and not joints:
        joint_summary = manifest_data.get("joint_summary", {})
        for j in joint_summary.get("joints", []):
            joint_info = JointInfo(
                name=j.get("name", ""),
                joint_type=j.get("type", "fixed"),
                axis=j.get("axis", [0, 0, 1]) if isinstance(j.get("axis"), list) else [0, 0, 1],
                limits={"lower": j.get("lower"), "upper": j.get("upper")} if "lower" in j else None,
                parent_link=j.get("parent"),
                child_link=j.get("child"),
            )
            if joint_info.joint_type != "fixed":
                joints.append(joint_info)

        links = [l.get("name") for l in joint_summary.get("links", [])]

    is_articulated = len(joints) > 0

    return ArticulatedAsset(
        object_id=obj_id,
        urdf_path=urdf_path,
        usd_path=usd_path,
        mesh_path=mesh_path,
        joints=joints,
        links=links,
        is_articulated=is_articulated,
        manifest_path=manifest_path if manifest_path.is_file() else None,
    )


# =============================================================================
# Scene Wiring
# =============================================================================


def wire_articulation_to_scene(
    stage,
    articulated_assets: List[ArticulatedAsset],
    root: Path,
    assets_prefix: str,
    usd_prefix: str,
) -> int:
    """Wire articulated assets into the USD scene.

    This updates object prims to reference articulated assets instead of
    static meshes, and adds articulation metadata.

    Args:
        stage: USD stage
        articulated_assets: List of ArticulatedAsset objects
        root: GCS root path
        assets_prefix: Path prefix for assets
        usd_prefix: Path prefix for USD output

    Returns:
        Number of objects wired
    """
    try:
        from pxr import Sdf, UsdGeom, UsdPhysics
    except ImportError:
        print("[ARTICULATION] WARNING: pxr not available, skipping articulation wiring")
        return 0

    stage_dir = root / usd_prefix
    wired_count = 0

    # Build lookup by object ID
    asset_lookup = {asset.object_id: asset for asset in articulated_assets}

    for prim in stage.Traverse():
        name = prim.GetName()
        if not name.startswith("obj_"):
            continue

        obj_id = name[len("obj_"):]

        # Check if this object has articulation
        asset = asset_lookup.get(obj_id)
        if not asset or not asset.is_articulated:
            continue

        print(f"[ARTICULATION] Wiring articulation for obj_{obj_id}")

        # Mark object as articulated
        prim.CreateAttribute("isArticulated", Sdf.ValueTypeNames.Bool).Set(True)
        prim.CreateAttribute("jointCount", Sdf.ValueTypeNames.Int).Set(len(asset.joints))

        # Store joint information
        joint_names = [j.name for j in asset.joints]
        joint_types = [j.joint_type for j in asset.joints]
        prim.CreateAttribute("jointNames", Sdf.ValueTypeNames.StringArray).Set(joint_names)
        prim.CreateAttribute("jointTypes", Sdf.ValueTypeNames.StringArray).Set(joint_types)

        # Store URDF/USD paths for downstream (Isaac Sim will need these)
        if asset.urdf_path:
            rel_urdf = os.path.relpath(asset.urdf_path, stage_dir)
            prim.CreateAttribute("urdfPath", Sdf.ValueTypeNames.String).Set(
                rel_urdf.replace("\\", "/")
            )

        if asset.usd_path:
            rel_usd = os.path.relpath(asset.usd_path, stage_dir)
            prim.CreateAttribute("articulatedUsdPath", Sdf.ValueTypeNames.String).Set(
                rel_usd.replace("\\", "/")
            )

        # Update asset type
        type_attr = prim.GetAttribute("assetType")
        if type_attr:
            type_attr.Set("articulated")

        # Add joint limits as custom data (useful for Isaac Lab)
        joint_limits = []
        for joint in asset.joints:
            if joint.limits:
                joint_limits.append({
                    "name": joint.name,
                    "type": joint.joint_type,
                    "lower": joint.limits.get("lower", 0),
                    "upper": joint.limits.get("upper", 0),
                })

        if joint_limits:
            prim.SetCustomDataByKey("jointLimits", joint_limits)

        wired_count += 1

    return wired_count


# =============================================================================
# Manifest Updates
# =============================================================================


def update_manifest_with_articulation(
    manifest: Dict[str, Any],
    articulated_assets: List[ArticulatedAsset],
) -> Dict[str, Any]:
    """Update scene manifest with articulation metadata.

    This enriches the manifest so downstream jobs (replicator, Isaac Lab)
    know which objects are articulated and their joint properties.

    Args:
        manifest: Scene manifest dictionary
        articulated_assets: List of ArticulatedAsset objects

    Returns:
        Updated manifest dictionary
    """
    # Build lookup
    asset_lookup = {asset.object_id: asset for asset in articulated_assets}

    for obj in manifest.get("objects", []):
        obj_id = str(obj.get("id"))
        asset = asset_lookup.get(obj_id)

        if not asset or not asset.is_articulated:
            continue

        # Update sim_role if needed
        if obj.get("sim_role") in ("static", "unknown"):
            obj["sim_role"] = "articulated_furniture"

        # Add articulation block
        obj["articulation"] = {
            "type": asset.joints[0].joint_type if asset.joints else None,
            "joint_count": len(asset.joints),
            "joints": [j.to_dict() for j in asset.joints],
            "links": asset.links,
            "urdf_path": str(asset.urdf_path) if asset.urdf_path else None,
            "usd_path": str(asset.usd_path) if asset.usd_path else None,
        }

        # Update asset path to point to articulated version
        if asset.usd_path:
            obj["asset"]["articulated_path"] = str(asset.usd_path)
            obj["asset"]["is_articulated"] = True
        elif asset.urdf_path:
            obj["asset"]["urdf_path"] = str(asset.urdf_path)
            obj["asset"]["is_articulated"] = True

    return manifest


# =============================================================================
# Main Orchestrator
# =============================================================================


class ArticulationWiring:
    """Orchestrates articulation wiring into USD scenes.

    Usage:
        wiring = ArticulationWiring(root, assets_prefix, usd_prefix)
        wiring.wire_scene(stage, scene_assets)
    """

    def __init__(
        self,
        root: Path,
        assets_prefix: str,
        usd_prefix: str,
        verbose: bool = True,
    ):
        self.root = Path(root)
        self.assets_prefix = assets_prefix
        self.usd_prefix = usd_prefix
        self.verbose = verbose

    def log(self, msg: str) -> None:
        if self.verbose:
            print(f"[ARTICULATION] {msg}")

    def find_assets(
        self,
        scene_assets: Optional[Dict] = None,
    ) -> List[ArticulatedAsset]:
        """Find all articulated assets."""
        return find_articulated_assets(
            self.root,
            self.assets_prefix,
            scene_assets,
        )

    def wire_scene(
        self,
        stage,
        scene_assets: Optional[Dict] = None,
    ) -> Tuple[int, List[ArticulatedAsset]]:
        """Wire articulation into a USD scene.

        Returns:
            Tuple of (wired_count, articulated_assets)
        """
        self.log("Finding articulated assets...")
        assets = self.find_assets(scene_assets)

        if not assets:
            self.log("No articulated assets found")
            return 0, []

        articulated = [a for a in assets if a.is_articulated]
        self.log(f"Found {len(articulated)} articulated assets")

        self.log("Wiring articulation to scene...")
        wired_count = wire_articulation_to_scene(
            stage,
            articulated,
            self.root,
            self.assets_prefix,
            self.usd_prefix,
        )

        self.log(f"Wired {wired_count} articulated objects")
        return wired_count, articulated

    def update_manifest(
        self,
        manifest: Dict[str, Any],
        assets: Optional[List[ArticulatedAsset]] = None,
    ) -> Dict[str, Any]:
        """Update manifest with articulation data.

        Args:
            manifest: Scene manifest
            assets: Optional list of articulated assets (will discover if not provided)

        Returns:
            Updated manifest
        """
        if assets is None:
            assets = self.find_assets()

        return update_manifest_with_articulation(manifest, assets)
