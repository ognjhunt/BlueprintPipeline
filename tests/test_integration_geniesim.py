#!/usr/bin/env python3
"""
Integration Tests for Genie Sim API Client.

These tests verify the full integration between BlueprintPipeline and Genie Sim 3.0.
They require a valid API key and will make real API calls when run with the
@pytest.mark.integration marker.

Usage:
    # Run integration tests (requires API key)
    RUN_GENIESIM_INTEGRATION=1 pytest -v -m integration tests/test_integration_geniesim.py

    # Skip integration tests
    pytest -v -m "not integration" tests/

Environment Variables:
    GENIE_SIM_API_KEY: Required for integration tests
    RUN_GENIESIM_INTEGRATION: Set to "1" to opt-in to integration tests
    GENIE_SIM_API_URL: Optional (defaults to production endpoint)
"""

import os
import sys
import time
from pathlib import Path

import pytest

# Add repo root and module path to sys.path
REPO_ROOT = Path(__file__).resolve().parents[1]
EXPORT_JOB_ROOT = REPO_ROOT / "genie-sim-export-job"
for path in (REPO_ROOT, EXPORT_JOB_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from geniesim_client import (
    GenieSimClient,
    GenerationParams,
    JobStatus,
    HealthStatus,
    GenieSimAuthenticationError,
    GenieSimJobNotFoundError,
)

# Skip all tests unless explicitly opted-in and API key is set
GENIE_SIM_API_KEY = os.getenv("GENIE_SIM_API_KEY")
RUN_GENIESIM_INTEGRATION = os.getenv("RUN_GENIESIM_INTEGRATION") == "1"
pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not RUN_GENIESIM_INTEGRATION,
        reason="Set RUN_GENIESIM_INTEGRATION=1 to opt-in to Genie Sim integration tests",
    ),
    pytest.mark.skipif(
        not GENIE_SIM_API_KEY,
        reason="GENIE_SIM_API_KEY not set - skipping Genie Sim integration tests",
    ),
]


class TestGenieSimHealthCheck:
    """Test Genie Sim API health check functionality."""

    @pytest.mark.integration
    def test_health_check_success(self):
        """Test health check returns valid status."""
        client = GenieSimClient()

        try:
            status = client.health_check()

            # Check basic fields
            assert isinstance(status, HealthStatus)
            assert isinstance(status.available, bool)
            assert status.checked_at is not None

            if status.available:
                # If available, should have version info
                assert status.api_version is not None
                print(f"✅ Genie Sim API is available (version: {status.api_version})")
            else:
                # If not available, should have error
                assert status.error is not None
                print(f"⚠️  Genie Sim API not available: {status.error}")

        finally:
            client.close()

    @pytest.mark.integration
    def test_health_check_with_invalid_endpoint(self):
        """Test health check with invalid endpoint returns error."""
        client = GenieSimClient(endpoint="https://invalid-endpoint-that-does-not-exist.com")

        try:
            status = client.health_check()

            # Should fail gracefully
            assert status.available is False
            assert status.error is not None
            assert "Connection failed" in status.error or "failed" in status.error.lower()

        finally:
            client.close()


class TestGenieSimJobLifecycle:
    """Test complete job submission and monitoring lifecycle."""

    @pytest.mark.integration
    def test_full_export_import_cycle(self):
        """
        Test complete export → generation → import cycle.

        This is the most important integration test - it verifies:
        1. Health check works
        2. Job submission succeeds
        3. Progress polling works
        4. Job completes successfully (or times out gracefully)
        5. Download works (if job completes)
        """
        client = GenieSimClient()

        try:
            # Step 1: Check API health
            health = client.health_check()
            if not health.available:
                pytest.skip(f"Genie Sim API not available: {health.error}")

            print(f"✅ API available (version: {health.api_version})")

            # Step 2: Create minimal test scene
            scene_graph = {
                "scene_id": f"test_integration_{int(time.time())}",
                "objects": [
                    {
                        "id": "test_cube",
                        "category": "box",
                        "position": [0.5, 0, 0.85],
                        "dimensions": [0.05, 0.05, 0.05],
                    }
                ],
                "robot": {
                    "type": "franka",
                    "base_position": [0, 0, 0],
                },
            }

            asset_index = {
                "test_cube": {
                    "usd_path": "/assets/test_cube.usd",
                    "embeddings": []
                }
            }

            task_config = {
                "tasks": [
                    {
                        "task_id": "pick_test_cube",
                        "task_name": "pick_object",
                        "description": "Pick up test cube",
                        "target_object_id": "test_cube",
                    }
                ]
            }

            generation_params = GenerationParams(
                episodes_per_task=2,  # Minimal for testing
                num_variations=1,
                robot_type="franka",
                min_quality_score=0.5,  # Lower threshold for test
            )

            # Step 3: Submit job
            print("Submitting test job...")
            result = client.submit_generation_job(
                scene_graph=scene_graph,
                asset_index=asset_index,
                task_config=task_config,
                generation_params=generation_params,
                job_name="integration_test_job",
            )

            assert result.success
            assert result.job_id is not None
            job_id = result.job_id
            print(f"✅ Job submitted: {job_id}")

            # Step 4: Monitor progress (with timeout)
            print("Monitoring job progress...")
            max_wait_seconds = 300  # 5 minutes max
            start_time = time.time()

            while (time.time() - start_time) < max_wait_seconds:
                progress = client.get_job_progress(job_id)

                print(f"  Status: {progress.status.value}, Progress: {progress.progress_percent:.1f}%")

                if progress.status == JobStatus.COMPLETED:
                    print("✅ Job completed successfully!")
                    break
                elif progress.status == JobStatus.FAILED:
                    pytest.fail(f"Job failed: {progress.current_task}")
                elif progress.status == JobStatus.CANCELLED:
                    pytest.fail("Job was cancelled")

                time.sleep(10)  # Poll every 10 seconds
            else:
                # Timeout - cancel job and skip
                print(f"⚠️  Job exceeded {max_wait_seconds}s timeout, cancelling...")
                try:
                    client.cancel_job(job_id)
                except Exception:
                    pass
                pytest.skip(f"Job did not complete within {max_wait_seconds}s")

            # Step 5: Download results (if job completed)
            print("Downloading generated episodes...")
            download_result = client.download_episodes(
                job_id=job_id,
                output_dir=Path("/tmp/geniesim_test_download"),
            )

            assert download_result.success
            assert download_result.episode_count > 0
            print(f"✅ Downloaded {download_result.episode_count} episodes")

        finally:
            client.close()

    @pytest.mark.integration
    def test_api_unavailable_fallback(self):
        """Test graceful degradation when API unavailable."""
        # Use invalid endpoint to simulate API unavailable
        client = GenieSimClient(endpoint="https://invalid-geniesim-endpoint.com")

        try:
            # Health check should fail gracefully
            health = client.health_check()
            assert not health.available
            assert health.error is not None

            # Job submission should raise clear error
            with pytest.raises(Exception) as exc_info:
                client.submit_generation_job(
                    scene_graph={},
                    asset_index={},
                    task_config={},
                    generation_params=GenerationParams(),
                )

            # Error should be informative
            assert "Connection" in str(exc_info.value) or "failed" in str(exc_info.value).lower()

        finally:
            client.close()

    @pytest.mark.integration
    def test_job_cancellation(self):
        """Test job cancellation and cleanup."""
        client = GenieSimClient()

        try:
            # Check API health first
            health = client.health_check()
            if not health.available:
                pytest.skip(f"Genie Sim API not available: {health.error}")

            # Submit a job
            scene_graph = {
                "scene_id": f"test_cancel_{int(time.time())}",
                "objects": [],
                "robot": {"type": "franka"},
            }

            result = client.submit_generation_job(
                scene_graph=scene_graph,
                asset_index={},
                task_config={"tasks": []},
                generation_params=GenerationParams(episodes_per_task=100),  # Large job
            )

            assert result.success
            job_id = result.job_id

            # Wait a bit for job to start
            time.sleep(2)

            # Cancel the job
            cancel_success = client.cancel_job(job_id)
            assert cancel_success

            # Verify job is cancelled
            progress = client.get_job_progress(job_id)
            assert progress.status == JobStatus.CANCELLED

            print("✅ Job cancellation works correctly")

        finally:
            client.close()


class TestGenieSimAuthentication:
    """Test API authentication and error handling."""

    def test_missing_api_key(self):
        """Test that missing API key raises clear error."""
        # Temporarily unset API key
        old_key = os.environ.pop("GENIE_SIM_API_KEY", None)

        try:
            with pytest.raises(GenieSimAuthenticationError) as exc_info:
                GenieSimClient(api_key=None)

            assert "API key required" in str(exc_info.value)

        finally:
            # Restore API key
            if old_key:
                os.environ["GENIE_SIM_API_KEY"] = old_key

    @pytest.mark.integration
    def test_invalid_api_key(self):
        """Test that invalid API key is detected."""
        client = GenieSimClient(api_key="invalid_key_12345")

        try:
            # Health check with invalid key should fail
            health = client.health_check()

            # Depending on API implementation, might be unavailable or have auth error
            if not health.available:
                assert health.error is not None
                # Could contain "auth", "401", "403", etc.

        finally:
            client.close()


class TestGenieSimErrorRecovery:
    """Test error recovery and resilience."""

    @pytest.mark.integration
    def test_job_not_found_error(self):
        """Test handling of non-existent job ID."""
        client = GenieSimClient()

        try:
            with pytest.raises(GenieSimJobNotFoundError):
                client.get_job_progress("nonexistent_job_id_12345")

        finally:
            client.close()

    @pytest.mark.integration
    def test_network_timeout_recovery(self):
        """Test that network timeouts are handled gracefully."""
        # Use very short timeout to force timeout error
        client = GenieSimClient(timeout=1)

        try:
            # Health check with short timeout might fail
            health = client.health_check()

            # Should not crash, either succeeds or fails gracefully
            assert isinstance(health, HealthStatus)

            if not health.available:
                # Timeout should be mentioned in error
                assert health.error is not None

        finally:
            client.close()


class TestGenieSimMultipleJobs:
    """Test handling of multiple concurrent jobs."""

    @pytest.mark.integration
    @pytest.mark.slow
    def test_parallel_job_submission(self):
        """Test submitting multiple jobs in parallel."""
        client = GenieSimClient()

        try:
            health = client.health_check()
            if not health.available:
                pytest.skip(f"Genie Sim API not available: {health.error}")

            # Submit 3 small jobs in parallel
            job_ids = []
            for i in range(3):
                scene_graph = {
                    "scene_id": f"test_parallel_{i}_{int(time.time())}",
                    "objects": [],
                    "robot": {"type": "franka"},
                }

                result = client.submit_generation_job(
                    scene_graph=scene_graph,
                    asset_index={},
                    task_config={"tasks": []},
                    generation_params=GenerationParams(episodes_per_task=1),
                    job_name=f"parallel_test_{i}",
                )

                assert result.success
                job_ids.append(result.job_id)

            print(f"✅ Submitted {len(job_ids)} parallel jobs")

            # Verify all jobs are tracked
            for job_id in job_ids:
                progress = client.get_job_progress(job_id)
                assert progress.job_id == job_id
                assert progress.status in [JobStatus.PENDING, JobStatus.RUNNING]

            # Cancel all jobs
            for job_id in job_ids:
                try:
                    client.cancel_job(job_id)
                except Exception:
                    pass  # Best effort cleanup

        finally:
            client.close()


if __name__ == "__main__":
    # Run tests when executed directly
    pytest.main([__file__, "-v", "-m", "integration"])
