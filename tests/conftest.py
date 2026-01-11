"""Pytest configuration and shared fixtures."""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import MagicMock


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def temp_file(temp_dir):
    """Create a temporary file for tests."""
    file_path = temp_dir / "test_file.txt"
    file_path.write_text("test content")
    return file_path


@pytest.fixture
def mock_logger():
    """Create a mock logger for tests."""
    logger = MagicMock()
    logger.info = MagicMock()
    logger.error = MagicMock()
    logger.warning = MagicMock()
    logger.debug = MagicMock()
    return logger


@pytest.fixture
def mock_gcs_client():
    """Create a mock GCS client for tests."""
    mock_client = MagicMock()
    mock_bucket = MagicMock()
    mock_client.bucket.return_value = mock_bucket
    return mock_client


@pytest.fixture
def mock_pubsub_client():
    """Create a mock Pub/Sub client for tests."""
    mock_client = MagicMock()
    mock_publisher = MagicMock()
    mock_subscriber = MagicMock()
    mock_client.publisher = mock_publisher
    mock_client.subscriber = mock_subscriber
    return mock_client


@pytest.fixture
def sample_scene_manifest():
    """Create a sample scene manifest for testing."""
    return {
        "scene_id": "test_scene_001",
        "version": "1.0",
        "objects": [
            {
                "id": "obj_001",
                "name": "Cube",
                "type": "primitive",
                "position": [0.0, 0.0, 0.5],
                "dimensions": [0.1, 0.1, 0.1],
                "mass": 1.0,
                "physics": {
                    "static": False,
                    "friction": 0.5,
                    "restitution": 0.1,
                },
            },
        ],
        "robot": {
            "type": "franka",
            "position": [0.0, 0.0, 0.0],
        },
    }


@pytest.fixture
def sample_task_config():
    """Create a sample task configuration for testing."""
    return {
        "task_id": "reach_001",
        "task_type": "reaching",
        "robot": "franka",
        "observation_space": {
            "joint_positions": 7,
            "joint_velocities": 7,
            "ee_position": 3,
            "target_position": 3,
        },
        "action_space": {
            "type": "joint_position",
            "dimension": 7,
        },
        "reward": {
            "reaching": 1.0,
            "motion_smoothness": 0.1,
            "action_penalty": 0.01,
        },
    }


@pytest.fixture
def sample_episode_data():
    """Create sample episode data for testing."""
    return {
        "episode_id": "ep_001",
        "task_id": "reach_001",
        "success": True,
        "frames": [
            {
                "timestamp": 0.0,
                "observation": {
                    "joint_pos": [0.0] * 7,
                    "joint_vel": [0.0] * 7,
                    "ee_pos": [0.0, 0.0, 0.5],
                },
                "action": [0.0] * 7,
                "reward": 0.0,
            },
        ],
        "duration": 5.0,
    }


@pytest.fixture
def sample_asset_metadata():
    """Create sample asset metadata for testing."""
    return {
        "asset_id": "chair_001",
        "name": "Office Chair",
        "category": "furniture",
        "description": "A modern office chair",
        "tags": ["modern", "office", "seating"],
        "properties": {
            "material": "fabric",
            "color": "black",
            "dimensions": {"height": 1.0, "width": 0.6, "depth": 0.6},
        },
    }


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "unit: Unit tests that don't require external resources"
    )
    config.addinivalue_line(
        "markers", "integration: Integration tests that may require external resources"
    )
    config.addinivalue_line(
        "markers", "slow: Slow running tests"
    )
    config.addinivalue_line(
        "markers", "production: Tests related to production readiness"
    )


@pytest.fixture(scope="session")
def test_data_dir():
    """Get the test data directory."""
    return Path(__file__).parent / "data"


@pytest.fixture
def mock_isaac_sim_env():
    """Create a mock Isaac Sim environment for testing."""
    mock_env = MagicMock()
    mock_env.reset.return_value = {"obs": [0.0] * 17}  # 7 joint pos + 7 joint vel + 3 ee pos
    mock_env.step.return_value = (
        {"obs": [0.1] * 17},
        1.0,  # reward
        False,  # done
        {"success": False},  # info
    )
    mock_env.render.return_value = None
    mock_env.close.return_value = None
    return mock_env


@pytest.fixture
def mock_genie_sim_framework():
    """Create a mock Genie Sim framework for testing."""
    mock_framework = MagicMock()
    mock_framework.is_available.return_value = True
    mock_framework.run_data_collection.return_value = {
        "success": True,
        "episodes": 10,
        "frames": 1000,
    }
    return mock_framework


class TestConfig:
    """Test configuration constants."""

    # Timeouts
    UNIT_TEST_TIMEOUT = 5
    INTEGRATION_TEST_TIMEOUT = 30

    # Paths
    FIXTURES_DIR = Path(__file__).parent / "fixtures"
    DATA_DIR = Path(__file__).parent / "data"

    # Test data counts
    SMALL_BATCH = 10
    MEDIUM_BATCH = 100
    LARGE_BATCH = 1000


@pytest.fixture
def test_config():
    """Provide test configuration."""
    return TestConfig()
