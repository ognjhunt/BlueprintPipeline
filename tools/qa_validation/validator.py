"""End-to-End QA Validation Implementation.

Comprehensive validation for BlueprintPipeline scenes.
"""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


class ValidationLevel(str, Enum):
    """Validation severity levels."""
    ERROR = "error"      # Blocks deployment
    WARNING = "warning"  # Should be fixed but not blocking
    INFO = "info"        # Informational


@dataclass
class ValidationResult:
    """Result of a single validation check."""
    name: str
    passed: bool
    level: ValidationLevel = ValidationLevel.ERROR
    message: str = ""
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ValidationReport:
    """Complete validation report for a scene."""
    scene_id: str
    passed: bool = True
    timestamp: str = ""
    results: List[ValidationResult] = field(default_factory=list)
    issues: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    summary: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.utcnow().isoformat() + "Z"

    def add_result(self, result: ValidationResult) -> None:
        """Add a validation result."""
        self.results.append(result)
        if not result.passed:
            if result.level == ValidationLevel.ERROR:
                self.passed = False
                self.issues.append(f"[{result.name}] {result.message}")
            elif result.level == ValidationLevel.WARNING:
                self.warnings.append(f"[{result.name}] {result.message}")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scene_id": self.scene_id,
            "passed": self.passed,
            "timestamp": self.timestamp,
            "results": [
                {
                    "name": r.name,
                    "passed": r.passed,
                    "level": r.level.value,
                    "message": r.message,
                    "details": r.details,
                }
                for r in self.results
            ],
            "issues": self.issues,
            "warnings": self.warnings,
            "summary": self.summary,
        }


class SceneValidator:
    """Comprehensive scene validator.

    Validates:
    1. File structure and manifest
    2. USD scene integrity
    3. Physics stability
    4. Articulation controllability
    5. Replicator bundle
    6. Isaac Lab tasks
    """

    def __init__(
        self,
        scene_dir: Path,
        scene_id: Optional[str] = None,
        verbose: bool = True,
    ):
        self.scene_dir = Path(scene_dir)
        self.scene_id = scene_id or self.scene_dir.name
        self.verbose = verbose
        self.report = ValidationReport(scene_id=self.scene_id)

    def log(self, msg: str) -> None:
        if self.verbose:
            print(f"[QA-VALIDATOR] {msg}")

    def run_full_validation(self) -> ValidationReport:
        """Run all validation checks."""
        self.log(f"Starting validation for scene: {self.scene_id}")
        self.log(f"Scene directory: {self.scene_dir}")

        # 1. File structure validation
        self._validate_file_structure()

        # 2. Manifest validation
        self._validate_manifest()

        # 3. USD validation
        self._validate_usd()

        # 4. Physics validation (if USD is valid)
        if self._has_valid_usd():
            self._validate_physics()

        # 5. Articulation validation
        self._validate_articulation()

        # 6. Replicator bundle validation
        self._validate_replicator()

        # 7. Isaac Lab validation
        self._validate_isaac_lab()

        # Generate summary
        self._generate_summary()

        return self.report

    def _validate_file_structure(self) -> None:
        """Validate required files exist."""
        self.log("Checking file structure...")

        required_files = [
            ("assets/scene_manifest.json", "Canonical manifest"),
            ("layout/scene_layout_scaled.json", "Scene layout"),
            ("usd/scene.usda", "USD scene"),
        ]

        optional_files = [
            ("replicator/placement_regions.usda", "Replicator placement regions"),
            ("replicator/bundle_metadata.json", "Replicator metadata"),
            ("seg/inventory.json", "Semantic inventory"),
        ]

        for rel_path, desc in required_files:
            full_path = self.scene_dir / rel_path
            passed = full_path.is_file()
            self.report.add_result(ValidationResult(
                name=f"file_exists:{rel_path}",
                passed=passed,
                level=ValidationLevel.ERROR,
                message=f"{desc} {'found' if passed else 'MISSING'}",
                details={"path": str(full_path)},
            ))

        for rel_path, desc in optional_files:
            full_path = self.scene_dir / rel_path
            passed = full_path.is_file()
            self.report.add_result(ValidationResult(
                name=f"file_exists:{rel_path}",
                passed=passed,
                level=ValidationLevel.WARNING,
                message=f"{desc} {'found' if passed else 'missing (optional)'}",
                details={"path": str(full_path)},
            ))

    def _validate_manifest(self) -> None:
        """Validate manifest schema and content."""
        self.log("Validating manifest...")

        manifest_path = self.scene_dir / "assets" / "scene_manifest.json"
        if not manifest_path.is_file():
            self.report.add_result(ValidationResult(
                name="manifest_load",
                passed=False,
                message="Cannot load manifest - file not found",
            ))
            return

        try:
            manifest = json.loads(manifest_path.read_text())
        except json.JSONDecodeError as e:
            self.report.add_result(ValidationResult(
                name="manifest_json",
                passed=False,
                message=f"Invalid JSON: {e}",
            ))
            return

        # Check required fields
        required_fields = ["version", "scene_id", "scene", "objects"]
        for field_name in required_fields:
            passed = field_name in manifest
            self.report.add_result(ValidationResult(
                name=f"manifest_field:{field_name}",
                passed=passed,
                level=ValidationLevel.ERROR,
                message=f"Field '{field_name}' {'present' if passed else 'missing'}",
            ))

        # Check required scene fields
        scene = manifest.get("scene", {})
        for field_name in ["coordinate_frame", "meters_per_unit"]:
            passed = field_name in scene
            self.report.add_result(ValidationResult(
                name=f"manifest_scene_field:{field_name}",
                passed=passed,
                level=ValidationLevel.ERROR,
                message=f"Scene field '{field_name}' {'present' if passed else 'missing'}",
            ))

        # Check objects have required properties
        objects = manifest.get("objects", [])
        for obj in objects:
            obj_id = obj.get("id", "unknown")

            for field_name in ["id", "sim_role", "asset", "transform"]:
                passed = field_name in obj
                self.report.add_result(ValidationResult(
                    name=f"object_field:{obj_id}:{field_name}",
                    passed=passed,
                    level=ValidationLevel.ERROR,
                    message=f"Object '{obj_id}' field '{field_name}' {'present' if passed else 'missing'}",
                ))

            # Check for asset path
            asset = obj.get("asset", {})
            has_asset_path = bool(asset.get("path"))
            self.report.add_result(ValidationResult(
                name=f"object_asset:{obj_id}",
                passed=has_asset_path,
                level=ValidationLevel.ERROR,
                message=f"Object '{obj_id}' {'has' if has_asset_path else 'missing'} asset path",
            ))

            # Check for transform
            transform = obj.get("transform", {})
            has_position = "position" in transform
            self.report.add_result(ValidationResult(
                name=f"object_transform:{obj_id}",
                passed=has_position,
                level=ValidationLevel.ERROR,
                message=f"Object '{obj_id}' {'has' if has_position else 'missing'} position",
            ))
            has_scale = "scale" in transform
            self.report.add_result(ValidationResult(
                name=f"object_transform_scale:{obj_id}",
                passed=has_scale,
                level=ValidationLevel.ERROR,
                message=f"Object '{obj_id}' {'has' if has_scale else 'missing'} scale",
            ))

        # Check scale authority
        meters_per_unit = scene.get("meters_per_unit")
        self.report.add_result(ValidationResult(
            name="manifest_scale",
            passed=meters_per_unit is not None,
            level=ValidationLevel.ERROR,
            message=f"meters_per_unit: {meters_per_unit or 'not set'}",
            details={"meters_per_unit": meters_per_unit},
        ))

    def _validate_usd(self) -> None:
        """Validate USD scene loads correctly."""
        self.log("Validating USD scene...")

        usd_path = self.scene_dir / "usd" / "scene.usda"
        if not usd_path.is_file():
            self.report.add_result(ValidationResult(
                name="usd_exists",
                passed=False,
                message="USD scene file not found",
            ))
            return

        # Check USD syntax (basic text parsing)
        try:
            usd_content = usd_path.read_text()

            # Check for USD header
            has_header = usd_content.startswith("#usda 1.0")
            self.report.add_result(ValidationResult(
                name="usd_header",
                passed=has_header,
                message=f"USD header {'valid' if has_header else 'invalid or missing'}",
            ))

            # Check for missing references (basic pattern matching)
            import re
            ref_pattern = r'references\s*=\s*@([^@]+)@'
            references = re.findall(ref_pattern, usd_content)

            missing_refs = []
            for ref in references:
                if ref.startswith("./") or ref.startswith("../"):
                    ref_path = (usd_path.parent / ref).resolve()
                    if not ref_path.is_file():
                        missing_refs.append(ref)

            self.report.add_result(ValidationResult(
                name="usd_references",
                passed=len(missing_refs) == 0,
                level=ValidationLevel.WARNING,
                message=f"{len(missing_refs)} missing references" if missing_refs else "All references valid",
                details={"missing": missing_refs[:10]},  # Limit to first 10
            ))

            # Check for object prims
            obj_pattern = r'def\s+(?:Xform|Mesh)\s+"obj_[^"]+"\s*\('
            obj_prims = re.findall(obj_pattern, usd_content)
            self.report.add_result(ValidationResult(
                name="usd_objects",
                passed=len(obj_prims) > 0,
                level=ValidationLevel.WARNING,
                message=f"Found {len(obj_prims)} object prims",
                details={"count": len(obj_prims)},
            ))

        except Exception as e:
            self.report.add_result(ValidationResult(
                name="usd_parse",
                passed=False,
                message=f"Failed to parse USD: {e}",
            ))

    def _has_valid_usd(self) -> bool:
        """Check if USD validation passed."""
        for result in self.report.results:
            if result.name.startswith("usd_") and not result.passed:
                if result.level == ValidationLevel.ERROR:
                    return False
        return True

    def _validate_physics(self) -> None:
        """Validate physics configuration in USD."""
        self.log("Validating physics configuration...")

        # Check for physics APIs in USD
        usd_path = self.scene_dir / "usd" / "scene.usda"
        if not usd_path.is_file():
            return

        try:
            usd_content = usd_path.read_text()

            # Check for PhysicsRigidBodyAPI
            has_rigid_body = "PhysicsRigidBodyAPI" in usd_content
            self.report.add_result(ValidationResult(
                name="physics_rigid_body",
                passed=has_rigid_body,
                level=ValidationLevel.INFO,
                message=f"PhysicsRigidBodyAPI {'found' if has_rigid_body else 'not found'}",
            ))

            # Check for PhysicsCollisionAPI
            has_collision = "PhysicsCollisionAPI" in usd_content
            self.report.add_result(ValidationResult(
                name="physics_collision",
                passed=has_collision,
                level=ValidationLevel.INFO,
                message=f"PhysicsCollisionAPI {'found' if has_collision else 'not found'}",
            ))

            # Check for PhysicsMaterialAPI
            has_material = "PhysicsMaterialAPI" in usd_content
            self.report.add_result(ValidationResult(
                name="physics_material",
                passed=has_material,
                level=ValidationLevel.INFO,
                message=f"PhysicsMaterialAPI {'found' if has_material else 'not found'}",
            ))

            # Check for mass authoring
            has_mass = "physics:mass" in usd_content
            self.report.add_result(ValidationResult(
                name="physics_mass",
                passed=has_mass,
                level=ValidationLevel.INFO,
                message=f"Mass properties {'found' if has_mass else 'not found'}",
            ))

        except Exception as e:
            self.report.add_result(ValidationResult(
                name="physics_check",
                passed=False,
                level=ValidationLevel.WARNING,
                message=f"Failed to check physics: {e}",
            ))

    def _validate_articulation(self) -> None:
        """Validate articulated objects."""
        self.log("Validating articulation...")

        # Check for interactive assets
        interactive_dir = self.scene_dir / "assets" / "interactive"
        has_interactive = interactive_dir.is_dir() and any(interactive_dir.iterdir())

        self.report.add_result(ValidationResult(
            name="articulation_assets",
            passed=True,  # Optional
            level=ValidationLevel.INFO,
            message=f"Interactive assets {'found' if has_interactive else 'not found'}",
        ))

        if has_interactive:
            # Check for URDF/USD articulation files
            urdf_files = list(interactive_dir.glob("*/*.urdf"))
            usd_files = list(interactive_dir.glob("*/*.usda")) + list(interactive_dir.glob("*/*.usd"))

            self.report.add_result(ValidationResult(
                name="articulation_files",
                passed=len(urdf_files) + len(usd_files) > 0,
                level=ValidationLevel.INFO,
                message=f"Found {len(urdf_files)} URDF and {len(usd_files)} USD articulation files",
            ))

    def _validate_replicator(self) -> None:
        """Validate Replicator bundle."""
        self.log("Validating Replicator bundle...")

        replicator_dir = self.scene_dir / "replicator"
        if not replicator_dir.is_dir():
            self.report.add_result(ValidationResult(
                name="replicator_dir",
                passed=False,
                level=ValidationLevel.WARNING,
                message="Replicator directory not found",
            ))
            return

        # Check placement regions
        placement_regions = replicator_dir / "placement_regions.usda"
        self.report.add_result(ValidationResult(
            name="replicator_placement",
            passed=placement_regions.is_file(),
            level=ValidationLevel.WARNING,
            message=f"Placement regions {'found' if placement_regions.is_file() else 'not found'}",
        ))

        # Check policy scripts
        policies_dir = replicator_dir / "policies"
        policy_scripts = list(policies_dir.glob("*.py")) if policies_dir.is_dir() else []
        self.report.add_result(ValidationResult(
            name="replicator_policies",
            passed=len(policy_scripts) > 0,
            level=ValidationLevel.WARNING,
            message=f"Found {len(policy_scripts)} policy scripts",
            details={"scripts": [s.name for s in policy_scripts]},
        ))

        # Check variation manifest
        variation_manifest = replicator_dir / "variation_assets" / "manifest.json"
        self.report.add_result(ValidationResult(
            name="replicator_variations",
            passed=variation_manifest.is_file(),
            level=ValidationLevel.INFO,
            message=f"Variation manifest {'found' if variation_manifest.is_file() else 'not found'}",
        ))

    def _validate_isaac_lab(self) -> None:
        """Validate Isaac Lab task generation."""
        self.log("Validating Isaac Lab tasks...")

        isaac_lab_dir = self.scene_dir / "isaac_lab"
        if not isaac_lab_dir.is_dir():
            self.report.add_result(ValidationResult(
                name="isaac_lab_dir",
                passed=False,
                level=ValidationLevel.WARNING,
                message="Isaac Lab directory not found (task generation may be pending)",
            ))
            return

        # Check for env_cfg.py
        env_cfg = isaac_lab_dir / "env_cfg.py"
        self.report.add_result(ValidationResult(
            name="isaac_lab_env_cfg",
            passed=env_cfg.is_file(),
            level=ValidationLevel.WARNING,
            message=f"Environment config {'found' if env_cfg.is_file() else 'not found'}",
        ))

        # Check for task files
        task_files = list(isaac_lab_dir.glob("task_*.py"))
        self.report.add_result(ValidationResult(
            name="isaac_lab_tasks",
            passed=len(task_files) > 0,
            level=ValidationLevel.WARNING,
            message=f"Found {len(task_files)} task files",
            details={"tasks": [t.name for t in task_files]},
        ))

        # Check train config
        train_cfg = isaac_lab_dir / "train_cfg.yaml"
        self.report.add_result(ValidationResult(
            name="isaac_lab_train_cfg",
            passed=train_cfg.is_file(),
            level=ValidationLevel.WARNING,
            message=f"Training config {'found' if train_cfg.is_file() else 'not found'}",
        ))

    def _generate_summary(self) -> None:
        """Generate validation summary."""
        total_checks = len(self.report.results)
        passed_checks = sum(1 for r in self.report.results if r.passed)
        failed_checks = total_checks - passed_checks

        errors = sum(1 for r in self.report.results if not r.passed and r.level == ValidationLevel.ERROR)
        warnings = sum(1 for r in self.report.results if not r.passed and r.level == ValidationLevel.WARNING)

        self.report.summary = {
            "total_checks": total_checks,
            "passed": passed_checks,
            "failed": failed_checks,
            "errors": errors,
            "warnings": warnings,
            "pass_rate": f"{100 * passed_checks / total_checks:.1f}%" if total_checks > 0 else "N/A",
        }

        self.log(f"Validation complete: {passed_checks}/{total_checks} checks passed")
        if errors > 0:
            self.log(f"  ERRORS: {errors}")
        if warnings > 0:
            self.log(f"  WARNINGS: {warnings}")


def run_qa_validation(
    scene_dir: Path,
    scene_id: Optional[str] = None,
    output_report: Optional[Path] = None,
    verbose: bool = True,
) -> ValidationReport:
    """Run full QA validation on a scene.

    Args:
        scene_dir: Path to scene directory
        scene_id: Optional scene ID
        output_report: Optional path to save JSON report
        verbose: Print progress

    Returns:
        ValidationReport with all results
    """
    validator = SceneValidator(scene_dir, scene_id, verbose)
    report = validator.run_full_validation()

    if output_report:
        output_report = Path(output_report)
        output_report.parent.mkdir(parents=True, exist_ok=True)
        output_report.write_text(json.dumps(report.to_dict(), indent=2))
        if verbose:
            print(f"[QA-VALIDATOR] Report saved to: {output_report}")

    return report
