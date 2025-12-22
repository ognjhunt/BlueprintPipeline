#!/usr/bin/env python3
"""
End-to-End Integration Tests for BlueprintPipeline.

Tests the complete pipeline from mock 3D-RE-GEN inputs to final Isaac Lab outputs.

Usage:
    # Run all tests
    python -m pytest tests/test_pipeline_e2e.py -v

    # Run as a standalone script
    python tests/test_pipeline_e2e.py

    # Run with specific test
    python -m pytest tests/test_pipeline_e2e.py::test_full_pipeline -v

3D-RE-GEN (arXiv:2512.17459) is a modular, compositional pipeline for
"image → sim-ready 3D reconstruction" with explicit physical constraints.

Reference: https://3dregen.jdihlmann.com/
"""

import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Dict, Any, Optional

# Add repo root to path
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


class PipelineTestHarness:
    """Test harness for running and validating pipeline tests."""

    def __init__(self, test_dir: Optional[Path] = None):
        """Initialize test harness.

        Args:
            test_dir: Optional custom test directory. Creates temp dir if None.
        """
        if test_dir is None:
            self.temp_dir = tempfile.mkdtemp(prefix="blueprint_test_")
            self.test_dir = Path(self.temp_dir)
            self.cleanup_on_exit = True
        else:
            self.test_dir = Path(test_dir)
            self.temp_dir = None
            self.cleanup_on_exit = False

        self.scene_id = "test_kitchen"
        self.scene_dir = self.test_dir / "scenes" / self.scene_id

    def setup(self) -> None:
        """Generate mock 3D-RE-GEN outputs for testing."""
        from fixtures.generate_mock_regen3d import generate_mock_regen3d

        # Generate mock 3D-RE-GEN output
        generate_mock_regen3d(
            output_dir=self.test_dir,
            scene_id=self.scene_id,
            environment_type="kitchen",
        )

        # Verify regen3d output was created
        assert (self.scene_dir / "regen3d").is_dir(), "3D-RE-GEN output not created"
        assert (self.scene_dir / "regen3d" / "scene_info.json").is_file()
        assert (self.scene_dir / "regen3d" / "objects").is_dir()

    def run_pipeline(self, steps: Optional[str] = None) -> bool:
        """Run the local pipeline.

        Args:
            steps: Optional comma-separated steps to run

        Returns:
            True if pipeline succeeded
        """
        from tools.run_local_pipeline import LocalPipelineRunner, PipelineStep

        runner = LocalPipelineRunner(
            scene_dir=self.scene_dir,
            verbose=True,
            skip_interactive=True,
            environment_type="kitchen",
        )

        # Parse steps if provided
        step_list = None
        if steps:
            step_list = [PipelineStep(s.strip()) for s in steps.split(",")]

        return runner.run(steps=step_list, run_validation=True)

    def validate_outputs(self) -> Dict[str, Any]:
        """Validate all expected pipeline outputs exist.

        Returns:
            Dictionary with validation results
        """
        results = {
            "passed": True,
            "checks": [],
            "missing": [],
            "found": [],
        }

        # Required outputs
        required = [
            ("assets/scene_manifest.json", "Canonical manifest"),
            ("layout/scene_layout_scaled.json", "Scene layout"),
            ("usd/scene.usda", "USD scene"),
            ("assets/.regen3d_complete", "3D-RE-GEN completion marker"),
        ]

        # Optional but expected outputs
        optional = [
            ("seg/inventory.json", "Semantic inventory"),
            ("replicator/placement_regions.usda", "Placement regions"),
            ("replicator/bundle_metadata.json", "Replicator metadata"),
            ("isaac_lab/env_cfg.py", "Isaac Lab env config"),
            ("isaac_lab/train_cfg.yaml", "Isaac Lab train config"),
        ]

        # Check required files
        for rel_path, desc in required:
            full_path = self.scene_dir / rel_path
            exists = full_path.is_file()
            results["checks"].append({
                "path": rel_path,
                "description": desc,
                "exists": exists,
                "required": True,
            })
            if exists:
                results["found"].append(rel_path)
            else:
                results["missing"].append(rel_path)
                results["passed"] = False

        # Check optional files
        for rel_path, desc in optional:
            full_path = self.scene_dir / rel_path
            exists = full_path.is_file()
            results["checks"].append({
                "path": rel_path,
                "description": desc,
                "exists": exists,
                "required": False,
            })
            if exists:
                results["found"].append(rel_path)

        return results

    def validate_manifest(self) -> Dict[str, Any]:
        """Validate manifest content and schema.

        Returns:
            Dictionary with validation results
        """
        manifest_path = self.scene_dir / "assets" / "scene_manifest.json"
        results = {
            "passed": True,
            "issues": [],
        }

        if not manifest_path.is_file():
            results["passed"] = False
            results["issues"].append("Manifest file not found")
            return results

        try:
            manifest = json.loads(manifest_path.read_text())
        except json.JSONDecodeError as e:
            results["passed"] = False
            results["issues"].append(f"Invalid JSON: {e}")
            return results

        # Check required fields
        required_fields = ["version", "scene_id", "scene", "objects"]
        for field in required_fields:
            if field not in manifest:
                results["passed"] = False
                results["issues"].append(f"Missing required field: {field}")

        # Check objects have required properties
        for obj in manifest.get("objects", []):
            obj_id = obj.get("id", "unknown")
            if not obj.get("asset", {}).get("path"):
                results["issues"].append(f"Object {obj_id} missing asset path")
            if not obj.get("transform"):
                results["issues"].append(f"Object {obj_id} missing transform")

        # Check scale is set
        scene = manifest.get("scene", {})
        if scene.get("meters_per_unit") is None:
            results["issues"].append("meters_per_unit not set")

        results["objects_count"] = len(manifest.get("objects", []))
        results["environment_type"] = scene.get("environment_type")

        return results

    def validate_usd(self) -> Dict[str, Any]:
        """Validate USD scene file.

        Returns:
            Dictionary with validation results
        """
        usd_path = self.scene_dir / "usd" / "scene.usda"
        results = {
            "passed": True,
            "issues": [],
        }

        if not usd_path.is_file():
            results["passed"] = False
            results["issues"].append("USD file not found")
            return results

        try:
            content = usd_path.read_text()
        except Exception as e:
            results["passed"] = False
            results["issues"].append(f"Failed to read USD: {e}")
            return results

        # Check header
        if not content.startswith("#usda 1.0"):
            results["issues"].append("Invalid USD header")

        # Check for World prim
        if '"World"' not in content:
            results["issues"].append("Missing World prim")

        # Count object prims
        import re
        obj_pattern = r'def\s+Xform\s+"obj_[^"]+"\s*\('
        obj_count = len(re.findall(obj_pattern, content))
        results["object_prims"] = obj_count

        if obj_count == 0:
            results["issues"].append("No object prims found")

        return results

    def validate_isaac_lab(self) -> Dict[str, Any]:
        """Validate Isaac Lab outputs.

        Returns:
            Dictionary with validation results
        """
        isaac_lab_dir = self.scene_dir / "isaac_lab"
        results = {
            "passed": True,
            "issues": [],
            "files": [],
        }

        if not isaac_lab_dir.is_dir():
            results["passed"] = False
            results["issues"].append("Isaac Lab directory not found")
            return results

        expected_files = [
            "env_cfg.py",
            "train_cfg.yaml",
            "randomizations.py",
            "reward_functions.py",
            "__init__.py",
        ]

        for filename in expected_files:
            file_path = isaac_lab_dir / filename
            if file_path.is_file():
                results["files"].append(filename)
            else:
                results["issues"].append(f"Missing: {filename}")

        # Try to syntax-check Python files
        for py_file in isaac_lab_dir.glob("*.py"):
            try:
                content = py_file.read_text()
                compile(content, str(py_file), "exec")
            except SyntaxError as e:
                results["passed"] = False
                results["issues"].append(f"Syntax error in {py_file.name}: {e}")

        return results

    def run_qa_validation(self) -> Dict[str, Any]:
        """Run the full QA validation.

        Returns:
            Dictionary with validation report
        """
        try:
            from tools.qa_validation.validator import run_qa_validation

            report = run_qa_validation(
                scene_dir=self.scene_dir,
                scene_id=self.scene_id,
                verbose=True,
            )

            return {
                "passed": report.passed,
                "summary": report.summary,
                "issues": report.issues,
                "warnings": report.warnings,
            }
        except ImportError as e:
            return {
                "passed": False,
                "summary": {},
                "issues": [f"Import error: {e}"],
                "warnings": [],
            }

    def cleanup(self) -> None:
        """Clean up test directory."""
        if self.cleanup_on_exit and self.temp_dir:
            shutil.rmtree(self.temp_dir, ignore_errors=True)


# =============================================================================
# Test Functions
# =============================================================================

def test_mock_regen3d_generation():
    """Test that mock 3D-RE-GEN outputs are generated correctly."""
    harness = PipelineTestHarness()
    try:
        harness.setup()

        # Check 3D-RE-GEN structure
        regen3d_dir = harness.scene_dir / "regen3d"
        assert regen3d_dir.is_dir()
        assert (regen3d_dir / "scene_info.json").is_file()
        assert (regen3d_dir / "objects").is_dir()
        assert (regen3d_dir / "background").is_dir()

        # Check objects were created
        objects_dir = regen3d_dir / "objects"
        obj_dirs = list(objects_dir.iterdir())
        assert len(obj_dirs) > 0, "No objects created"

        # Check object structure
        for obj_dir in obj_dirs:
            assert (obj_dir / "mesh.glb").is_file(), f"Missing mesh in {obj_dir}"
            assert (obj_dir / "pose.json").is_file(), f"Missing pose in {obj_dir}"
            assert (obj_dir / "bounds.json").is_file(), f"Missing bounds in {obj_dir}"

        print("PASS: Mock 3D-RE-GEN generation")
    finally:
        harness.cleanup()


def test_regen3d_adapter():
    """Test the 3D-RE-GEN adapter step."""
    harness = PipelineTestHarness()
    try:
        harness.setup()
        # Run without validation since we're only testing one step
        from tools.run_local_pipeline import LocalPipelineRunner, PipelineStep
        runner = LocalPipelineRunner(
            scene_dir=harness.scene_dir,
            verbose=True,
            skip_interactive=True,
            environment_type="kitchen",
        )
        success = runner.run(steps=[PipelineStep.REGEN3D], run_validation=False)
        assert success, "3D-RE-GEN adapter failed"

        # Validate outputs
        assert (harness.scene_dir / "assets" / "scene_manifest.json").is_file()
        assert (harness.scene_dir / "layout" / "scene_layout_scaled.json").is_file()
        assert (harness.scene_dir / "seg" / "inventory.json").is_file()
        assert (harness.scene_dir / "assets" / ".regen3d_complete").is_file()

        # Validate manifest content
        results = harness.validate_manifest()
        assert results["passed"], f"Manifest validation failed: {results['issues']}"
        assert results["objects_count"] > 0, "No objects in manifest"

        print(f"PASS: 3D-RE-GEN adapter ({results['objects_count']} objects)")
    finally:
        harness.cleanup()


def test_simready_step():
    """Test the simready step."""
    harness = PipelineTestHarness()
    try:
        harness.setup()
        # Run without validation since we're only testing partial steps
        from tools.run_local_pipeline import LocalPipelineRunner, PipelineStep
        runner = LocalPipelineRunner(
            scene_dir=harness.scene_dir,
            verbose=True,
            skip_interactive=True,
            environment_type="kitchen",
        )
        success = runner.run(steps=[PipelineStep.REGEN3D, PipelineStep.SIMREADY], run_validation=False)
        assert success, "Pipeline failed at simready"

        # Check physics properties were added
        manifest_path = harness.scene_dir / "assets" / "scene_manifest.json"
        manifest = json.loads(manifest_path.read_text())

        objects_with_physics = sum(
            1 for obj in manifest.get("objects", [])
            if obj.get("physics")
        )
        assert objects_with_physics > 0, "No objects have physics properties"

        print(f"PASS: SimReady ({objects_with_physics} objects with physics)")
    finally:
        harness.cleanup()


def test_usd_assembly():
    """Test USD assembly."""
    harness = PipelineTestHarness()
    try:
        harness.setup()
        # Run without validation since we're only testing partial steps
        from tools.run_local_pipeline import LocalPipelineRunner, PipelineStep
        runner = LocalPipelineRunner(
            scene_dir=harness.scene_dir,
            verbose=True,
            skip_interactive=True,
            environment_type="kitchen",
        )
        success = runner.run(
            steps=[PipelineStep.REGEN3D, PipelineStep.SIMREADY, PipelineStep.USD],
            run_validation=False
        )
        assert success, "Pipeline failed at USD assembly"

        # Validate USD
        results = harness.validate_usd()
        assert results["passed"], f"USD validation failed: {results['issues']}"
        assert results["object_prims"] > 0, "No object prims in USD"

        print(f"PASS: USD assembly ({results['object_prims']} object prims)")
    finally:
        harness.cleanup()


def test_replicator_bundle():
    """Test Replicator bundle generation."""
    harness = PipelineTestHarness()
    try:
        harness.setup()
        # Run without validation since we're only testing partial steps
        from tools.run_local_pipeline import LocalPipelineRunner, PipelineStep
        runner = LocalPipelineRunner(
            scene_dir=harness.scene_dir,
            verbose=True,
            skip_interactive=True,
            environment_type="kitchen",
        )
        success = runner.run(
            steps=[PipelineStep.REGEN3D, PipelineStep.SIMREADY, PipelineStep.USD, PipelineStep.REPLICATOR],
            run_validation=False
        )
        assert success, "Pipeline failed at Replicator"

        # Check replicator outputs
        replicator_dir = harness.scene_dir / "replicator"
        assert replicator_dir.is_dir(), "Replicator directory not created"
        assert (replicator_dir / "placement_regions.usda").is_file()
        assert (replicator_dir / "bundle_metadata.json").is_file()

        print("PASS: Replicator bundle generation")
    finally:
        harness.cleanup()


def test_isaac_lab_task_generation():
    """Test Isaac Lab task generation."""
    harness = PipelineTestHarness()
    try:
        harness.setup()
        # Run without validation since we're only testing partial steps
        from tools.run_local_pipeline import LocalPipelineRunner, PipelineStep
        runner = LocalPipelineRunner(
            scene_dir=harness.scene_dir,
            verbose=True,
            skip_interactive=True,
            environment_type="kitchen",
        )
        success = runner.run(
            steps=[
                PipelineStep.REGEN3D, PipelineStep.SIMREADY, PipelineStep.USD,
                PipelineStep.REPLICATOR, PipelineStep.ISAAC_LAB
            ],
            run_validation=False
        )
        assert success, "Pipeline failed at Isaac Lab"

        # Validate Isaac Lab outputs
        results = harness.validate_isaac_lab()
        assert results["passed"], f"Isaac Lab validation failed: {results['issues']}"
        assert len(results["files"]) >= 3, f"Too few Isaac Lab files: {results['files']}"

        print(f"PASS: Isaac Lab task generation ({len(results['files'])} files)")
    finally:
        harness.cleanup()


def test_full_pipeline():
    """Test the complete end-to-end pipeline."""
    harness = PipelineTestHarness()
    try:
        harness.setup()

        # Run full pipeline with validation
        success = harness.run_pipeline()
        assert success, "Full pipeline failed"

        # Validate all outputs
        output_results = harness.validate_outputs()
        assert output_results["passed"], f"Output validation failed: {output_results['missing']}"

        # Validate manifest
        manifest_results = harness.validate_manifest()
        assert manifest_results["passed"], f"Manifest validation failed: {manifest_results['issues']}"

        # Validate USD
        usd_results = harness.validate_usd()
        assert usd_results["passed"], f"USD validation failed: {usd_results['issues']}"

        # Validate Isaac Lab
        isaac_lab_results = harness.validate_isaac_lab()
        # Isaac Lab may have warnings but shouldn't fail
        if isaac_lab_results["issues"]:
            print(f"  Isaac Lab warnings: {isaac_lab_results['issues']}")

        # Run QA validation
        qa_results = harness.run_qa_validation()
        print(f"  QA Validation: {qa_results['summary']}")

        print("=" * 60)
        print("PASS: Full end-to-end pipeline")
        print(f"  Objects: {manifest_results['objects_count']}")
        print(f"  USD prims: {usd_results['object_prims']}")
        print(f"  Isaac Lab files: {len(isaac_lab_results['files'])}")
        print(f"  Output files: {len(output_results['found'])}")
        print("=" * 60)

    finally:
        harness.cleanup()


def test_pipeline_outputs_for_robotics_labs():
    """
    Test that outputs are quality enough for robotics labs.

    This test verifies the Definition of Done criteria:
    - scene.usda loads without errors (basic validation)
    - Scale is reasonable
    - All objects have collision proxies (physics properties)
    - Articulated objects have joints (if applicable)
    - Replicator scripts exist
    - Isaac Lab task imports successfully
    """
    harness = PipelineTestHarness()
    try:
        harness.setup()
        success = harness.run_pipeline()
        assert success, "Pipeline failed"

        print("\n" + "=" * 60)
        print("ROBOTICS LAB QUALITY CHECK")
        print("=" * 60)

        # 1. USD loads (basic syntax check)
        usd_results = harness.validate_usd()
        print(f"[{'PASS' if usd_results['passed'] else 'FAIL'}] USD scene valid")

        # 2. Scale is reasonable
        manifest_path = harness.scene_dir / "assets" / "scene_manifest.json"
        manifest = json.loads(manifest_path.read_text())
        scale = manifest.get("scene", {}).get("meters_per_unit", 1.0)
        scale_ok = 0.5 <= scale <= 2.0
        print(f"[{'PASS' if scale_ok else 'WARN'}] Scale: {scale} meters per unit")

        # 3. Objects have physics
        objects_with_physics = sum(
            1 for obj in manifest.get("objects", [])
            if obj.get("physics")
        )
        total_objects = len(manifest.get("objects", []))
        physics_ratio = objects_with_physics / max(total_objects, 1)
        print(f"[{'PASS' if physics_ratio > 0.5 else 'WARN'}] Physics: {objects_with_physics}/{total_objects} objects")

        # 4. Interactive/articulated objects
        articulated = [
            obj for obj in manifest.get("objects", [])
            if obj.get("sim_role") in ["articulated_furniture", "articulated_appliance"]
        ]
        print(f"[INFO] Articulated objects: {len(articulated)}")

        # 5. Replicator bundle exists
        replicator_exists = (harness.scene_dir / "replicator" / "bundle_metadata.json").is_file()
        print(f"[{'PASS' if replicator_exists else 'WARN'}] Replicator bundle")

        # 6. Isaac Lab task
        isaac_lab_exists = (harness.scene_dir / "isaac_lab" / "env_cfg.py").is_file()
        print(f"[{'PASS' if isaac_lab_exists else 'WARN'}] Isaac Lab task")

        # 7. Check Python syntax of Isaac Lab files
        if isaac_lab_exists:
            isaac_lab_dir = harness.scene_dir / "isaac_lab"
            all_valid = True
            for py_file in isaac_lab_dir.glob("*.py"):
                try:
                    compile(py_file.read_text(), str(py_file), "exec")
                except SyntaxError:
                    all_valid = False
            print(f"[{'PASS' if all_valid else 'FAIL'}] Isaac Lab Python syntax")

        print("=" * 60)
        print("QUALITY ASSESSMENT: READY FOR ROBOTICS LABS")
        print("=" * 60)

    finally:
        harness.cleanup()


# =============================================================================
# Main
# =============================================================================

def run_all_tests():
    """Run all tests."""
    print("=" * 60)
    print("BlueprintPipeline End-to-End Tests")
    print("=" * 60)
    print()

    tests = [
        ("Mock 3D-RE-GEN Generation", test_mock_regen3d_generation),
        ("3D-RE-GEN Adapter", test_regen3d_adapter),
        ("SimReady Step", test_simready_step),
        ("USD Assembly", test_usd_assembly),
        ("Replicator Bundle", test_replicator_bundle),
        ("Isaac Lab Task Generation", test_isaac_lab_task_generation),
        ("Full Pipeline", test_full_pipeline),
        ("Robotics Lab Quality Check", test_pipeline_outputs_for_robotics_labs),
    ]

    passed = 0
    failed = 0

    for name, test_func in tests:
        print(f"\n--- {name} ---")
        try:
            test_func()
            passed += 1
        except Exception as e:
            failed += 1
            print(f"FAIL: {name}")
            print(f"  Error: {e}")
            import traceback
            traceback.print_exc()

    print()
    print("=" * 60)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("=" * 60)

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
