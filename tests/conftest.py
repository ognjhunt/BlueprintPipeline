"""
Shared pytest fixtures for BlueprintPipeline tests.

This module provides common test utilities and fixtures to reduce duplication
across test files and improve test maintainability.
"""

from __future__ import annotations

import importlib.util
import base64
import hashlib
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Generator, Optional

import pytest


# ============================================================================
# PATH AND MODULE FIXTURES
# ============================================================================

@pytest.fixture(scope="session")
def repo_root() -> Path:
    """Return the repository root directory."""
    return Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def add_repo_to_path(repo_root: Path) -> None:
    """Add repository root to sys.path for imports."""
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))


@pytest.fixture
def job_dir_paths(repo_root: Path) -> Dict[str, Path]:
    """Return paths to all job directories."""
    return {
        "arena_export": repo_root / "arena-export-job",
        "dataset_delivery": repo_root / "dataset-delivery-job",
        "dream2flow_preparation": repo_root / "dream2flow-preparation-job",
        "dwm_preparation": repo_root / "dwm-preparation-job",
        "episode_generation": repo_root / "episode-generation-job",
        "firebase_cleanup": repo_root / "firebase-cleanup-job",
        "geniesim_export": repo_root / "genie-sim-export-job",
        "geniesim_gpu": repo_root / "genie-sim-gpu-job",
        "geniesim_import": repo_root / "genie-sim-import-job",
        "geniesim_local": repo_root / "genie-sim-local-job",
        "geniesim_submit": repo_root / "genie-sim-submit-job",
        "interactive": repo_root / "interactive-job",
        "isaac_lab": repo_root / "isaac-lab-job",
        "meshy": repo_root / "meshy-job",
        "objects": repo_root / "objects-job",
        "regen3d": repo_root / "regen3d-job",
        "replicator": repo_root / "replicator-job",
        "scale": repo_root / "scale-job",
        "scene_batch": repo_root / "scene-batch-job",
        "scene_generation": repo_root / "scene-generation-job",
        "simready": repo_root / "simready-job",
        "smart_placement_engine": repo_root / "smart-placement-engine-job",
        "upsell_features": repo_root / "upsell-features-job",
        "usd": repo_root / "usd-assembly-job",
        "variation_asset_pipeline": repo_root / "variation-asset-pipeline-job",
        "variation_gen": repo_root / "variation-gen-job",
    }


def _load_module_from_path(module_name: str, file_path: Path):
    """Load a Python module from a file path."""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load module {module_name} from {file_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def load_job_module(job_dir_paths: Dict[str, Path]):
    """Factory fixture for loading job modules dynamically."""
    def _loader(job_name: str, module_file: str):
        job_dir = job_dir_paths.get(job_name)
        if not job_dir:
            raise ValueError(f"Unknown job: {job_name}")

        if str(job_dir) not in sys.path:
            sys.path.insert(0, str(job_dir))

        module_path = job_dir / module_file
        return _load_module_from_path(f"{job_name}.{module_file.replace('.py', '')}", module_path)

    return _loader


# ============================================================================
# TEMPORARY DIRECTORY FIXTURES
# ============================================================================

@pytest.fixture
def temp_test_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for test outputs."""
    temp_dir = tempfile.mkdtemp(prefix="blueprint_test_")
    try:
        yield Path(temp_dir)
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def mock_scene_dir(temp_test_dir: Path) -> Path:
    """Create a mock scene directory structure."""
    scene_id = "test_scene"
    scene_dir = temp_test_dir / "scenes" / scene_id
    scene_dir.mkdir(parents=True, exist_ok=True)

    # Create standard subdirectories
    (scene_dir / "regen3d").mkdir(exist_ok=True)
    (scene_dir / "assets").mkdir(exist_ok=True)
    (scene_dir / "layout").mkdir(exist_ok=True)
    (scene_dir / "usd").mkdir(exist_ok=True)
    (scene_dir / "replicator").mkdir(exist_ok=True)

    return scene_dir


# ============================================================================
# MOCK DATA FIXTURES
# ============================================================================

@pytest.fixture
def mock_scene_manifest() -> Dict[str, Any]:
    """Return a minimal valid scene manifest."""
    return {
        "version": "1.0.0",
        "scene_id": "test_scene",
        "scene": {
            "environment_type": "kitchen",
            "coordinate_frame": "y_up",
            "meters_per_unit": 1.0,
            "room": {
                "bounds": {"width": 4.0, "depth": 4.0, "height": 2.5}
            },
        },
        "objects": [
            {
                "id": "table_0",
                "name": "table_0",
                "category": "table",
                "description": "wooden table",
                "sim_role": "static_object",
                "dimensions_est": {"width": 1.2, "depth": 0.8, "height": 0.75},
                "transform": {
                    "position": {"x": 0.0, "y": 0.375, "z": 0.0},
                    "rotation_quaternion": {"w": 1.0, "x": 0.0, "y": 0.0, "z": 0.0},
                    "scale": {"x": 1.0, "y": 1.0, "z": 1.0},
                },
                "asset": {"path": "table_0.usd"},
                "physics": {"mass": 10.0},
                "physics_hints": {"material_type": "wood"},
                "semantics": {"affordances": ["Support"]},
                "relationships": [],
            },
            {
                "id": "mug_0",
                "name": "mug_0",
                "category": "mug",
                "description": "coffee mug",
                "sim_role": "manipulable_object",
                "dimensions_est": {"width": 0.08, "depth": 0.08, "height": 0.1},
                "transform": {
                    "position": {"x": 0.0, "y": 0.8, "z": 0.0},
                    "rotation_quaternion": {"w": 1.0, "x": 0.0, "y": 0.0, "z": 0.0},
                    "scale": {"x": 1.0, "y": 1.0, "z": 1.0},
                },
                "asset": {"path": "mug_0.usd"},
                "physics": {"mass": 0.2},
                "physics_hints": {"material_type": "ceramic"},
                "semantics": {"affordances": ["Graspable", "Containable"]},
                "relationships": [],
            },
        ],
    }


@pytest.fixture
def write_mock_scene_manifest(mock_scene_dir: Path, mock_scene_manifest: Dict[str, Any]):
    """Write a mock scene manifest to the test scene directory."""
    manifest_path = mock_scene_dir / "assets" / "scene_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(mock_scene_manifest, indent=2))
    return manifest_path


@pytest.fixture
def mock_regen3d_output(mock_scene_dir: Path) -> Path:
    """Create minimal mock 3D-RE-GEN output structure."""
    regen3d_dir = mock_scene_dir / "regen3d"
    regen3d_dir.mkdir(parents=True, exist_ok=True)

    # Create mock scene_info.json
    scene_info = {
        "scene_id": "test_scene",
        "environment_type": "kitchen",
        "reconstruction_method": "3D-RE-GEN",
        "timestamp": "2026-01-15T00:00:00Z",
        "objects": [
            {"id": "table_0", "category": "table", "mesh": "table_0.glb"},
            {"id": "mug_0", "category": "mug", "mesh": "mug_0.glb"},
        ],
    }
    (regen3d_dir / "scene_info.json").write_text(json.dumps(scene_info, indent=2))

    # Create mock objects directory
    objects_dir = regen3d_dir / "objects"
    objects_dir.mkdir(exist_ok=True)

    # Create mock GLB files (empty files for testing)
    for obj_id in ["table_0", "mug_0"]:
        (objects_dir / f"{obj_id}.glb").write_bytes(b"mock_glb_data")

    return regen3d_dir


# ============================================================================
# ENVIRONMENT VARIABLE FIXTURES
# ============================================================================

@pytest.fixture
def clean_env(monkeypatch):
    """Clear common environment variables for clean test state."""
    env_vars_to_clear = [
        "PIPELINE_ENV",
        "PRODUCTION",
        "K_SERVICE",
        "KUBERNETES_SERVICE_HOST",
        "GEMINI_API_KEY",
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "SIMREADY_PHYSICS_MODE",
        "USE_GENIESIM",
        "BYPASS_QUALITY_GATES",
        "GOOGLE_APPLICATION_CREDENTIALS",
        "GCP_PROJECT_ID",
    ]
    for var in env_vars_to_clear:
        monkeypatch.delenv(var, raising=False)


@pytest.fixture
def mock_production_env(monkeypatch, clean_env):
    """Set up a mock production environment."""
    monkeypatch.setenv("PIPELINE_ENV", "production")
    monkeypatch.setenv("K_SERVICE", "test-service")
    monkeypatch.setenv("BP_QUALITY_HUMAN_APPROVAL_NOTIFICATION_CHANNELS", "#test-approvals")


@pytest.fixture
def mock_local_env(monkeypatch, clean_env):
    """Set up a mock local development environment."""
    monkeypatch.setenv("PIPELINE_ENV", "local")
    monkeypatch.setenv("SIMREADY_PHYSICS_MODE", "deterministic")
    monkeypatch.setenv("USE_GENIESIM", "false")


# ============================================================================
# GCS EMULATION FIXTURES
# ============================================================================

@pytest.fixture
def mock_gcs_bucket(temp_test_dir: Path, monkeypatch):
    """Create a mock GCS bucket using local filesystem."""
    bucket_dir = temp_test_dir / "mock_gcs_bucket"
    bucket_dir.mkdir(parents=True, exist_ok=True)

    # Mock GCS client
    class MockBlob:
        def __init__(self, bucket, name):
            self.bucket = bucket
            self.name = name
            self._path = bucket._bucket_dir / name
            self.size = None
            self.md5_hash = None

        def upload_from_filename(self, filename):
            self._path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(filename, self._path)
            payload = self._path.read_bytes()
            self.size = len(payload)
            self.md5_hash = base64.b64encode(hashlib.md5(payload).digest()).decode("utf-8")

        def download_to_filename(self, filename):
            Path(filename).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(self._path, filename)

        def exists(self):
            return self._path.exists()

        def reload(self):
            if self._path.exists():
                payload = self._path.read_bytes()
                self.size = len(payload)
                self.md5_hash = base64.b64encode(hashlib.md5(payload).digest()).decode("utf-8")

    class MockBucket:
        def __init__(self, bucket_dir):
            self._bucket_dir = bucket_dir

        def blob(self, name):
            return MockBlob(self, name)

        def list_blobs(self, prefix=None):
            if prefix:
                search_dir = self._bucket_dir / prefix
            else:
                search_dir = self._bucket_dir

            if not search_dir.exists():
                return []

            blobs = []
            for path in search_dir.rglob("*"):
                if path.is_file():
                    rel_path = path.relative_to(self._bucket_dir)
                    blobs.append(MockBlob(self, str(rel_path)))
            return blobs

    return MockBucket(bucket_dir)


# ============================================================================
# PIPELINE RUNNER FIXTURES
# ============================================================================

@pytest.fixture
def pipeline_test_harness(mock_scene_dir: Path, add_repo_to_path):
    """Create a pipeline test harness for E2E tests."""
    from tests.test_pipeline_e2e import PipelineTestHarness

    harness = PipelineTestHarness(test_dir=mock_scene_dir.parent.parent)
    return harness


# ============================================================================
# PYTEST CONFIGURATION
# ============================================================================

def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "gpu: marks tests as requiring GPU (deselect with '-m \"not gpu\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "e2e: marks tests as end-to-end tests"
    )
    config.addinivalue_line(
        "markers", "unit: marks tests as unit tests"
    )
    config.addinivalue_line(
        "markers", "requires_secrets: marks tests that need real API keys/secrets"
    )
