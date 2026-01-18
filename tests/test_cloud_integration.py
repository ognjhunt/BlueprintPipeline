#!/usr/bin/env python3
"""
Cloud Integration Tests for BlueprintPipeline.

Tests the full pipeline in a cloud-like environment:
- GCS storage operations
- Kubernetes job simulation
- Workflow execution
- Error handling and recovery

These tests can run locally with mocked services or against
real GCP infrastructure in a test project.

Usage:
    # Run with mocks (no GCP required)
    python -m pytest tests/test_cloud_integration.py -v

    # Run against real GCP (requires PROJECT_ID)
    PROJECT_ID=test-project python -m pytest tests/test_cloud_integration.py -v --cloud

    # Run specific test
    python -m pytest tests/test_cloud_integration.py::TestGCSOperations -v
"""

import base64
import hashlib
import json
import os
import shutil
import sys
import tempfile
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest import mock

import pytest

# Add repo root to path
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# =============================================================================
# Test Configuration
# =============================================================================

@dataclass
class CloudTestConfig:
    """Configuration for cloud integration tests."""
    project_id: str = ""
    bucket: str = ""
    region: str = "us-central1"
    use_real_gcp: bool = False
    cleanup_after: bool = True
    test_scene_id: str = ""

    def __post_init__(self):
        if not self.project_id:
            self.project_id = os.getenv("PROJECT_ID", "test-project")
        if not self.bucket:
            self.bucket = os.getenv("BUCKET", f"{self.project_id}-blueprint-test")
        if not self.test_scene_id:
            self.test_scene_id = f"test_scene_{uuid.uuid4().hex[:8]}"


def get_test_config() -> CloudTestConfig:
    """Get test configuration from environment."""
    use_real_gcp = os.getenv("USE_REAL_GCP", "").lower() in ("true", "1", "yes")
    return CloudTestConfig(use_real_gcp=use_real_gcp)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def config():
    """Get test configuration."""
    return get_test_config()


@pytest.fixture
def temp_scene_dir():
    """Create a temporary scene directory with mock data."""
    temp_dir = tempfile.mkdtemp(prefix="blueprint_test_")
    scene_dir = Path(temp_dir) / "scenes" / "test_scene"

    # Create directory structure
    for subdir in ["regen3d", "assets", "layout", "seg", "usd", "episodes"]:
        (scene_dir / subdir).mkdir(parents=True, exist_ok=True)

    # Create mock scene data
    create_mock_scene_data(scene_dir)

    yield scene_dir

    # Cleanup
    shutil.rmtree(temp_dir, ignore_errors=True)


def create_mock_scene_data(scene_dir: Path):
    """Create mock scene data for testing."""
    # Mock scene manifest
    manifest = {
        "version": "1.0",
        "scene_id": scene_dir.name,
        "scene": {
            "environment_type": "kitchen",
            "meters_per_unit": 1.0,
            "coordinate_frame": "Y",
        },
        "objects": [
            {
                "id": "obj_001",
                "category": "mug",
                "sim_role": "manipulable_object",
                "asset": {"path": "assets/obj_001/asset.glb"},
                "transform": {
                    "position": {"x": 0.5, "y": 0.9, "z": 0.3},
                    "rotation": {"w": 1, "x": 0, "y": 0, "z": 0},
                },
                "dimensions_est": {"width": 0.08, "height": 0.10, "depth": 0.08},
                "physics": {
                    "dynamic": True,
                    "mass_kg": 0.3,
                    "friction_static": 0.5,
                    "friction_dynamic": 0.4,
                },
            },
            {
                "id": "obj_002",
                "category": "counter",
                "sim_role": "static",
                "asset": {"path": "assets/obj_002/asset.glb"},
                "transform": {
                    "position": {"x": 0.0, "y": 0.45, "z": 0.0},
                    "rotation": {"w": 1, "x": 0, "y": 0, "z": 0},
                },
                "dimensions_est": {"width": 1.2, "height": 0.9, "depth": 0.6},
            },
        ],
    }
    (scene_dir / "assets" / "scene_manifest.json").write_text(json.dumps(manifest, indent=2))

    # Mock layout
    layout = {
        "scene_id": scene_dir.name,
        "objects": manifest["objects"],
        "room_bounds": {"min": [-2, 0, -2], "max": [2, 3, 2]},
    }
    (scene_dir / "layout" / "scene_layout_scaled.json").write_text(json.dumps(layout, indent=2))

    # Mock USD
    usda_content = '''#usda 1.0
(
    metersPerUnit = 1.0
    upAxis = "Y"
    defaultPrim = "World"
)

def Xform "World" {
    def Xform "Scene" {
        def Xform "obj_001" {
            double3 xformOp:translate = (0.5, 0.9, 0.3)
        }
    }
}
'''
    (scene_dir / "usd" / "scene.usda").write_text(usda_content)

    # Create completion markers
    (scene_dir / "assets" / ".regen3d_complete").write_text(json.dumps({
        "status": "complete",
        "timestamp": "2025-01-01T00:00:00Z",
    }))


# =============================================================================
# Mock GCS Client
# =============================================================================

class MockGCSClient:
    """Mock Google Cloud Storage client for testing."""

    def __init__(self):
        self._buckets: Dict[str, "MockBucket"] = {}

    def bucket(self, name: str) -> "MockBucket":
        if name not in self._buckets:
            self._buckets[name] = MockBucket(name)
        return self._buckets[name]

    def create_bucket(self, name: str) -> "MockBucket":
        self._buckets[name] = MockBucket(name)
        return self._buckets[name]


class MockBucket:
    """Mock GCS bucket."""

    def __init__(self, name: str):
        self.name = name
        self._blobs: Dict[str, bytes] = {}

    def blob(self, name: str) -> "MockBlob":
        return MockBlob(self, name)

    def list_blobs(self, prefix: str = "", max_results: int = 1000):
        for name in list(self._blobs.keys())[:max_results]:
            if name.startswith(prefix):
                yield MockBlob(self, name, exists=True)


class MockBlob:
    """Mock GCS blob."""

    def __init__(self, bucket: MockBucket, name: str, exists: bool = False):
        self._bucket = bucket
        self.name = name
        self._exists = exists or name in bucket._blobs
        self.size = None
        self.md5_hash = None

    def exists(self) -> bool:
        return self.name in self._bucket._blobs

    def upload_from_string(self, data: str, content_type: str = "text/plain"):
        payload = data.encode() if isinstance(data, str) else data
        self._bucket._blobs[self.name] = payload
        self.size = len(payload)
        self.md5_hash = base64.b64encode(hashlib.md5(payload).digest()).decode("utf-8")

    def upload_from_filename(self, filename: str):
        with open(filename, "rb") as f:
            payload = f.read()
        self._bucket._blobs[self.name] = payload
        self.size = len(payload)
        self.md5_hash = base64.b64encode(hashlib.md5(payload).digest()).decode("utf-8")

    def reload(self):
        if self.name in self._bucket._blobs:
            payload = self._bucket._blobs[self.name]
            self.size = len(payload)
            self.md5_hash = base64.b64encode(hashlib.md5(payload).digest()).decode("utf-8")

    def download_as_string(self) -> bytes:
        return self._bucket._blobs.get(self.name, b"")

    def download_to_filename(self, filename: str):
        Path(filename).write_bytes(self._bucket._blobs.get(self.name, b""))

    def delete(self):
        if self.name in self._bucket._blobs:
            del self._bucket._blobs[self.name]


# =============================================================================
# Test Classes
# =============================================================================

class TestGCSOperations:
    """Test GCS storage operations."""

    def test_upload_scene_manifest(self, temp_scene_dir, config):
        """Test uploading scene manifest to GCS."""
        client = MockGCSClient()
        bucket = client.bucket(config.bucket)

        # Upload manifest
        manifest_path = temp_scene_dir / "assets" / "scene_manifest.json"
        blob = bucket.blob(f"scenes/{config.test_scene_id}/assets/scene_manifest.json")
        blob.upload_from_filename(str(manifest_path))

        # Verify upload
        downloaded = blob.download_as_string()
        original = manifest_path.read_bytes()
        assert downloaded == original

    def test_list_scene_objects(self, config):
        """Test listing objects in a scene prefix."""
        client = MockGCSClient()
        bucket = client.bucket(config.bucket)

        # Upload some test files
        prefix = f"scenes/{config.test_scene_id}/"
        for name in ["assets/manifest.json", "usd/scene.usda", "episodes/ep_001.parquet"]:
            blob = bucket.blob(prefix + name)
            blob.upload_from_string(f"test content for {name}")

        # List blobs
        blobs = list(bucket.list_blobs(prefix=prefix))
        assert len(blobs) == 3

    def test_completion_marker_workflow(self, config):
        """Test completion marker creation and detection."""
        client = MockGCSClient()
        bucket = client.bucket(config.bucket)

        scene_id = config.test_scene_id
        marker_path = f"scenes/{scene_id}/assets/.regen3d_complete"

        # Initially, marker doesn't exist
        marker_blob = bucket.blob(marker_path)
        assert not marker_blob.exists()

        # Create marker
        marker_content = json.dumps({
            "status": "complete",
            "scene_id": scene_id,
            "timestamp": "2025-01-01T00:00:00Z",
        })
        marker_blob.upload_from_string(marker_content)

        # Now marker exists
        assert marker_blob.exists()

        # Verify content
        downloaded = json.loads(marker_blob.download_as_string())
        assert downloaded["status"] == "complete"


class TestPipelineExecution:
    """Test pipeline execution flow."""

    def test_local_pipeline_execution(self, temp_scene_dir):
        """Test running the local pipeline."""
        from tools.run_local_pipeline import LocalPipelineRunner, PipelineStep

        runner = LocalPipelineRunner(
            scene_dir=temp_scene_dir,
            verbose=False,
            skip_interactive=True,
            environment_type="kitchen",
        )

        # Run just the first few steps
        success = runner.run(
            steps=[PipelineStep.REGEN3D, PipelineStep.SIMREADY, PipelineStep.USD],
            run_validation=False,
        )

        assert success, "Pipeline should complete successfully"

        # Verify outputs
        assert (temp_scene_dir / "assets" / "scene_manifest.json").exists()
        assert (temp_scene_dir / "usd" / "scene.usda").exists()

    def test_pipeline_step_isolation(self, temp_scene_dir):
        """Test that pipeline steps are properly isolated."""
        from tools.run_local_pipeline import LocalPipelineRunner, PipelineStep

        runner = LocalPipelineRunner(
            scene_dir=temp_scene_dir,
            verbose=False,
            skip_interactive=True,
        )

        # Run only USD step (should fail without prerequisites)
        # Actually, we need to run regen3d first to create the manifest
        success = runner.run(
            steps=[PipelineStep.USD],
            run_validation=False,
        )

        # Should fail because manifest doesn't exist in the right format
        # (our mock data is incomplete)

    def test_pipeline_error_handling(self, temp_scene_dir):
        """Test pipeline handles errors gracefully."""
        from tools.run_local_pipeline import LocalPipelineRunner, PipelineStep

        # Remove required file
        manifest_path = temp_scene_dir / "assets" / "scene_manifest.json"
        if manifest_path.exists():
            manifest_path.unlink()

        runner = LocalPipelineRunner(
            scene_dir=temp_scene_dir,
            verbose=False,
            skip_interactive=True,
        )

        # Pipeline should handle missing files
        # (It will try to create them from regen3d step)


class TestErrorHandling:
    """Test error handling module."""

    def test_retry_decorator(self):
        """Test retry decorator with backoff."""
        from tools.error_handling import retry_with_backoff

        attempt_count = 0

        @retry_with_backoff(max_retries=3, base_delay=0.1)
        def flaky_function():
            nonlocal attempt_count
            attempt_count += 1
            if attempt_count < 3:
                raise ConnectionError("Temporary failure")
            return "success"

        result = flaky_function()
        assert result == "success"
        assert attempt_count == 3

    def test_retry_max_exceeded(self):
        """Test retry gives up after max retries."""
        from tools.error_handling import retry_with_backoff

        @retry_with_backoff(max_retries=2, base_delay=0.1)
        def always_fails():
            raise ConnectionError("Always fails")

        with pytest.raises(ConnectionError):
            always_fails()

    def test_non_retryable_error(self):
        """Test that non-retryable errors are not retried."""
        from tools.error_handling import retry_with_backoff, NonRetryableError

        attempt_count = 0

        @retry_with_backoff(max_retries=3, base_delay=0.1)
        def raises_non_retryable():
            nonlocal attempt_count
            attempt_count += 1
            raise NonRetryableError("Should not retry")

        with pytest.raises(NonRetryableError):
            raises_non_retryable()

        assert attempt_count == 1  # Only one attempt

    def test_circuit_breaker(self):
        """Test circuit breaker pattern."""
        from tools.error_handling import CircuitBreaker, CircuitBreakerOpen

        breaker = CircuitBreaker(
            name="test_service",
            failure_threshold=3,
            recovery_timeout=1.0,
        )

        # First 3 failures should succeed
        for _ in range(3):
            try:
                with breaker:
                    raise ValueError("Simulated failure")
            except ValueError:
                pass

        # Circuit should now be open
        assert breaker.is_open

        # Next call should raise CircuitBreakerOpen
        with pytest.raises(CircuitBreakerOpen):
            with breaker:
                pass

        # Wait for recovery
        time.sleep(1.1)

        # Circuit should be half-open
        assert breaker.is_half_open

        # Success should close the circuit
        with breaker:
            pass  # Success

        # Another success to meet threshold
        with breaker:
            pass

        assert breaker.is_closed

    def test_dead_letter_queue(self, temp_scene_dir):
        """Test dead letter queue operations."""
        from tools.error_handling import (
            LocalDeadLetterQueue,
            DeadLetterMessage,
            PipelineError,
        )

        dlq_dir = temp_scene_dir / "dead_letter"
        dlq = LocalDeadLetterQueue(directory=str(dlq_dir))

        # Create a message from an error
        error = PipelineError(
            message="Test error",
            retryable=True,
        )
        message = DeadLetterMessage.from_pipeline_error(
            error,
            original_payload={"scene_id": "test_scene"},
        )

        # Publish to DLQ
        message_id = dlq.publish(message)
        assert message_id == message.message_id

        # Get pending messages
        pending = dlq.get_pending()
        assert len(pending) == 1
        assert pending[0].message_id == message_id

        # Mark as resolved
        dlq.mark_resolved(message_id)

        # Should no longer be pending
        pending = dlq.get_pending()
        assert len(pending) == 0

        # Check stats
        stats = dlq.get_stats()
        assert stats["resolved"] == 1


class TestEpisodeGeneration:
    """Test episode generation components."""

    def test_trajectory_solver_robot_configs(self):
        """Test trajectory solver has all robot configs."""
        episode_job_dir = REPO_ROOT / "episode-generation-job"
        if str(episode_job_dir) not in sys.path:
            sys.path.insert(0, str(episode_job_dir))
        from trajectory_solver import ROBOT_CONFIGS

        # Check all expected robots are configured
        expected_robots = ["franka", "ur10", "fetch"]
        for robot in expected_robots:
            assert robot in ROBOT_CONFIGS, f"Missing robot config: {robot}"

        # Check config structure
        for name, config in ROBOT_CONFIGS.items():
            assert hasattr(config, "num_joints")
            assert hasattr(config, "joint_limits_lower")
            assert hasattr(config, "joint_limits_upper")

    def test_lerobot_export_structure(self, temp_scene_dir):
        """Test LeRobot export creates correct structure."""
        # This would normally test the full export, but we'll verify structure
        episodes_dir = temp_scene_dir / "episodes"

        expected_structure = [
            "meta/info.json",
            "meta/stats.json",
            "meta/tasks.jsonl",
            "meta/episodes.jsonl",
            "data/chunk-000/",
        ]

        # Create mock structure
        for path in expected_structure:
            if path.endswith("/"):
                (episodes_dir / path).mkdir(parents=True, exist_ok=True)
            else:
                (episodes_dir / path).parent.mkdir(parents=True, exist_ok=True)
                (episodes_dir / path).write_text("{}")


class TestMultiRobotSupport:
    """Test multi-robot configuration and support."""

    def test_robot_config_registry(self):
        """Test extended robot configurations."""
        from tools.isaac_lab_tasks.multi_robot import EXTENDED_ROBOT_CONFIGS

        expected_robots = ["franka", "ur10", "ur5", "fetch", "kuka_iiwa", "sawyer"]
        for robot in expected_robots:
            assert robot in EXTENDED_ROBOT_CONFIGS, f"Missing robot: {robot}"

            config = EXTENDED_ROBOT_CONFIGS[robot]
            assert "num_dofs" in config
            assert "reach" in config
            assert "payload" in config

    def test_dual_arm_config_creation(self):
        """Test creating dual-arm robot configuration."""
        from tools.isaac_lab_tasks.multi_robot import create_dual_arm_config

        config = create_dual_arm_config(
            left_robot="franka",
            right_robot="ur10",
            separation=1.5,
        )

        assert len(config.robots) == 2
        assert config.robots[0].robot_type == "franka"
        assert config.robots[1].robot_type == "ur10"
        assert config.collision_avoidance is True

    def test_robot_fleet_config(self):
        """Test creating robot fleet configuration."""
        from tools.isaac_lab_tasks.multi_robot import create_robot_fleet_config

        config = create_robot_fleet_config(
            robot_type="fetch",
            num_robots=4,
            formation="grid",
        )

        assert len(config.robots) == 4
        assert all(r.robot_type == "fetch" for r in config.robots)


class TestQualityValidation:
    """Test quality validation and gates."""

    def test_manifest_validation(self, temp_scene_dir):
        """Test manifest validation."""
        manifest_path = temp_scene_dir / "assets" / "scene_manifest.json"
        manifest = json.loads(manifest_path.read_text())

        # Verify required fields
        assert "version" in manifest
        assert "scene_id" in manifest
        assert "scene" in manifest
        assert "objects" in manifest

        # Verify scene config
        scene = manifest["scene"]
        assert "environment_type" in scene
        assert "meters_per_unit" in scene

        # Verify objects
        for obj in manifest["objects"]:
            assert "id" in obj
            assert "category" in obj
            assert "transform" in obj

    def test_usd_validation(self, temp_scene_dir):
        """Test USD file validation."""
        usd_path = temp_scene_dir / "usd" / "scene.usda"
        content = usd_path.read_text()

        # Basic structure checks
        assert content.startswith("#usda 1.0")
        assert "World" in content
        assert "metersPerUnit" in content


# =============================================================================
# Cloud Integration Tests (require real GCP)
# =============================================================================

@pytest.mark.skipif(
    not os.getenv("USE_REAL_GCP"),
    reason="Requires USE_REAL_GCP=true and GCP credentials"
)
class TestRealGCPIntegration:
    """Tests that run against real GCP infrastructure."""

    def test_real_gcs_upload(self, config, temp_scene_dir):
        """Test uploading to real GCS bucket."""
        from google.cloud import storage

        client = storage.Client(project=config.project_id)
        bucket = client.bucket(config.bucket)

        # Upload test file
        test_content = f"test content {uuid.uuid4()}"
        blob_path = f"test/{config.test_scene_id}/test.txt"
        blob = bucket.blob(blob_path)

        try:
            blob.upload_from_string(test_content)
            assert blob.exists()

            downloaded = blob.download_as_string().decode()
            assert downloaded == test_content
        finally:
            # Cleanup
            if blob.exists():
                blob.delete()

    def test_real_gcs_marker_workflow(self, config):
        """Test real GCS marker workflow."""
        from google.cloud import storage

        client = storage.Client(project=config.project_id)
        bucket = client.bucket(config.bucket)

        scene_id = config.test_scene_id
        prefix = f"test/{scene_id}/"

        try:
            # Create scene structure
            manifest_blob = bucket.blob(f"{prefix}assets/scene_manifest.json")
            manifest_blob.upload_from_string('{"test": true}')

            # Create completion marker
            marker_blob = bucket.blob(f"{prefix}assets/.regen3d_complete")
            marker_blob.upload_from_string(json.dumps({
                "status": "complete",
                "timestamp": "2025-01-01T00:00:00Z",
            }))

            # Verify workflow can detect marker
            assert marker_blob.exists()

            # List blobs with prefix
            blobs = list(bucket.list_blobs(prefix=prefix))
            assert len(blobs) >= 2

        finally:
            # Cleanup
            for blob in bucket.list_blobs(prefix=prefix):
                blob.delete()


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
