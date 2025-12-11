"""Typed helpers for working with the canonical scene manifest.

The dataclasses here mirror ``manifest_schema.json`` and provide a friendly API
for reading or constructing manifests in Python. Validation is delegated to the
JSON schema so that downstream jobs can still rely on the canonical contract
without pulling in heavy dependencies.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import jsonschema
except ImportError as exc:  # pragma: no cover - runtime dependency helper
    jsonschema = None

SCHEMA_PATH = Path(__file__).with_name("manifest_schema.json")


@dataclass
class Vector3:
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0


@dataclass
class Transform:
    position: Vector3 = field(default_factory=Vector3)
    scale: Vector3 = field(default_factory=lambda: Vector3(1.0, 1.0, 1.0))
    rotation_euler: Optional[Dict[str, float]] = None
    rotation_quaternion: Optional[Dict[str, float]] = None


@dataclass
class AssetRef:
    path: str
    asset_id: Optional[str] = None
    source: Optional[str] = None
    pack_name: Optional[str] = None
    relative_path: Optional[str] = None
    format: Optional[str] = None
    variants: Dict[str, str] = field(default_factory=dict)
    candidates: List[Dict[str, Any]] = field(default_factory=list)
    simready_metadata: Optional[Dict[str, Any]] = None


@dataclass
class SceneObject:
    id: str
    sim_role: str
    transform: Transform
    asset: AssetRef
    name: Optional[str] = None
    category: Optional[str] = None
    description: Optional[str] = None
    placement_region: Optional[str] = None
    must_be_separate_asset: Optional[bool] = None
    dimensions_est: Optional[Dict[str, float]] = None
    semantics: Dict[str, Any] = field(default_factory=dict)
    physics: Dict[str, Any] = field(default_factory=dict)
    physics_hints: Dict[str, Any] = field(default_factory=dict)
    articulation: Dict[str, Any] = field(default_factory=dict)
    relationships: List[Dict[str, Any]] = field(default_factory=list)
    variation_candidate: Optional[bool] = None


@dataclass
class Scene:
    coordinate_frame: str
    meters_per_unit: float
    environment_type: Optional[str] = None
    room: Dict[str, Any] = field(default_factory=dict)
    physics_defaults: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Manifest:
    version: str
    scene_id: str
    scene: Scene
    objects: List[SceneObject]
    assets: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "Manifest":
        """Validate a raw manifest dictionary and hydrate a typed instance."""
        if jsonschema is None:
            raise ImportError(
                "jsonschema is required to parse manifests. Install with `pip install jsonschema`."
            )

        with SCHEMA_PATH.open("r") as f:
            schema = json.load(f)
        jsonschema.validate(payload, schema)

        scene_data = payload["scene"]
        scene = Scene(
            coordinate_frame=scene_data["coordinate_frame"],
            meters_per_unit=scene_data["meters_per_unit"],
            environment_type=scene_data.get("environment_type"),
            room=scene_data.get("room", {}),
            physics_defaults=scene_data.get("physics_defaults", {}),
        )

        objects: List[SceneObject] = []
        for obj in payload.get("objects", []):
            transform_data = obj.get("transform", {})
            position = transform_data.get("position", {})
            scale = transform_data.get("scale", {})
            transform = Transform(
                position=Vector3(
                    float(position.get("x", 0.0)),
                    float(position.get("y", 0.0)),
                    float(position.get("z", 0.0)),
                ),
                scale=Vector3(
                    float(scale.get("x", 1.0)),
                    float(scale.get("y", 1.0)),
                    float(scale.get("z", 1.0)),
                ),
                rotation_euler=transform_data.get("rotation_euler"),
                rotation_quaternion=transform_data.get("rotation_quaternion"),
            )

            asset_data = obj.get("asset", {})
            asset = AssetRef(
                path=asset_data["path"],
                asset_id=asset_data.get("asset_id"),
                source=asset_data.get("source"),
                pack_name=asset_data.get("pack_name"),
                relative_path=asset_data.get("relative_path"),
                format=asset_data.get("format"),
                variants=asset_data.get("variants", {}),
                candidates=asset_data.get("candidates", []),
                simready_metadata=asset_data.get("simready_metadata"),
            )

            objects.append(
                SceneObject(
                    id=str(obj["id"]),
                    sim_role=obj["sim_role"],
                    transform=transform,
                    asset=asset,
                    name=obj.get("name"),
                    category=obj.get("category"),
                    description=obj.get("description"),
                    placement_region=obj.get("placement_region"),
                    must_be_separate_asset=obj.get("must_be_separate_asset"),
                    dimensions_est=obj.get("dimensions_est"),
                    semantics=obj.get("semantics", {}),
                    physics=obj.get("physics", {}),
                    physics_hints=obj.get("physics_hints", {}),
                    articulation=obj.get("articulation", {}),
                    relationships=obj.get("relationships", []),
                    variation_candidate=obj.get("variation_candidate"),
                )
            )

        return cls(
            version=payload["version"],
            scene_id=str(payload["scene_id"]),
            scene=scene,
            objects=objects,
            assets=payload.get("assets", {}),
            metadata=payload.get("metadata", {}),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the manifest to a dictionary."""
        return {
            "version": self.version,
            "scene_id": self.scene_id,
            "scene": {
                "coordinate_frame": self.scene.coordinate_frame,
                "meters_per_unit": self.scene.meters_per_unit,
                "environment_type": self.scene.environment_type,
                "room": self.scene.room,
                "physics_defaults": self.scene.physics_defaults,
            },
            "objects": [
                {
                    "id": obj.id,
                    "sim_role": obj.sim_role,
                    "name": obj.name,
                    "category": obj.category,
                    "description": obj.description,
                    "placement_region": obj.placement_region,
                    "must_be_separate_asset": obj.must_be_separate_asset,
                    "dimensions_est": obj.dimensions_est,
                    "semantics": obj.semantics,
                    "physics": obj.physics,
                    "physics_hints": obj.physics_hints,
                    "articulation": obj.articulation,
                    "relationships": obj.relationships,
                    "variation_candidate": obj.variation_candidate,
                    "transform": {
                        "position": {
                            "x": obj.transform.position.x,
                            "y": obj.transform.position.y,
                            "z": obj.transform.position.z,
                        },
                        "scale": {
                            "x": obj.transform.scale.x,
                            "y": obj.transform.scale.y,
                            "z": obj.transform.scale.z,
                        },
                        "rotation_euler": obj.transform.rotation_euler,
                        "rotation_quaternion": obj.transform.rotation_quaternion,
                    },
                    "asset": {
                        "path": obj.asset.path,
                        "asset_id": obj.asset.asset_id,
                        "source": obj.asset.source,
                        "pack_name": obj.asset.pack_name,
                        "relative_path": obj.asset.relative_path,
                        "format": obj.asset.format,
                        "variants": obj.asset.variants,
                        "candidates": obj.asset.candidates,
                        "simready_metadata": obj.asset.simready_metadata,
                    },
                }
                for obj in self.objects
            ],
            "assets": self.assets,
            "metadata": self.metadata,
        }


__all__ = ["Manifest", "Scene", "SceneObject", "Transform", "Vector3", "AssetRef"]
