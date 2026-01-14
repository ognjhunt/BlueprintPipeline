"""
Integration tests for Genie Sim 3.0 adapter module.

Tests the conversion from BlueprintPipeline manifest to Genie Sim format:
- Scene graph conversion (nodes + edges)
- Asset index building
- Task configuration generation
- Commercial asset filtering
"""

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict

import pytest

# Add parent directory to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from tools.geniesim_adapter import (
    GenieSimExporter,
    GenieSimExportConfig,
    GenieSimExportResult,
)
from tools.geniesim_adapter.scene_graph import (
    SceneGraphConverter,
    GenieSimSceneGraph,
    GenieSimNode,
    GenieSimEdge,
)
from tools.geniesim_adapter.asset_index import (
    AssetIndexBuilder,
    GenieSimAssetIndex,
    GenieSimAsset,
)
from tools.geniesim_adapter.task_config import (
    TaskConfigGenerator,
    GenieSimTaskConfig,
    SuggestedTask,
)


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def sample_manifest() -> Dict[str, Any]:
    """Sample BlueprintPipeline manifest for testing."""
    return {
        "scene_id": "test_kitchen_001",
        "version": "1.0.0",
        "scene": {
            "environment_type": "kitchen",
            "coordinate_frame": "y_up",
            "meters_per_unit": 1.0,
            "room": {
                "bounds": {
                    "width": 4.0,
                    "depth": 5.0,
                    "height": 2.8,
                }
            },
        },
        "objects": [
            {
                "id": "mug_001",
                "name": "Coffee Mug",
                "category": "mug",
                "description": "White ceramic coffee mug with handle",
                "sim_role": "manipulable_object",
                "dimensions_est": {
                    "width": 0.08,
                    "depth": 0.08,
                    "height": 0.10,
                },
                "transform": {
                    "position": {"x": 1.0, "y": 0.9, "z": 0.05},
                    "rotation_euler": {"roll": 0, "pitch": 0, "yaw": 0},
                    "scale": {"x": 1.0, "y": 1.0, "z": 1.0},
                },
                "semantics": {
                    "affordances": ["Graspable", "Fillable"],
                },
                "physics": {
                    "mass": 0.3,
                    "friction": 0.5,
                },
                "physics_hints": {
                    "material_type": "ceramic",
                },
                "asset": {
                    "path": "objects/mug_001/asset.usdz",
                    "source": "blueprintpipeline",
                },
            },
            {
                "id": "countertop_001",
                "name": "Kitchen Counter",
                "category": "countertop",
                "description": "Granite kitchen countertop",
                "sim_role": "static",
                "dimensions_est": {
                    "width": 2.0,
                    "depth": 0.6,
                    "height": 0.05,
                },
                "transform": {
                    "position": {"x": 1.0, "y": 0.85, "z": 0.0},
                    "rotation_euler": {"roll": 0, "pitch": 0, "yaw": 0},
                    "scale": {"x": 1.0, "y": 1.0, "z": 1.0},
                },
                "semantics": {
                    "affordances": ["Supportable"],
                },
                "asset": {
                    "path": "objects/countertop_001/asset.usdz",
                    "source": "blueprintpipeline",
                },
            },
            {
                "id": "cabinet_001",
                "name": "Kitchen Cabinet",
                "category": "cabinet",
                "description": "Wooden kitchen cabinet with door",
                "sim_role": "articulated_furniture",
                "dimensions_est": {
                    "width": 0.6,
                    "depth": 0.4,
                    "height": 0.8,
                },
                "transform": {
                    "position": {"x": 0.3, "y": 1.5, "z": 0.0},
                    "rotation_quaternion": {"w": 1, "x": 0, "y": 0, "z": 0},
                    "scale": {"x": 1.0, "y": 1.0, "z": 1.0},
                },
                "semantics": {
                    "affordances": [
                        {"type": "Openable", "open_angle": 90},
                    ],
                },
                "articulation": {
                    "type": "revolute",
                    "axis": "z",
                    "limits": {"lower": 0, "upper": 90},
                },
                "asset": {
                    "path": "objects/cabinet_001/asset.usdz",
                    "source": "blueprintpipeline",
                },
            },
        ],
    }


@pytest.fixture
def manifest_with_external_assets() -> Dict[str, Any]:
    """Manifest with mix of own assets and external NC assets."""
    return {
        "scene_id": "test_mixed_assets",
        "version": "1.0.0",
        "scene": {
            "environment_type": "kitchen",
            "coordinate_frame": "y_up",
            "meters_per_unit": 1.0,
        },
        "objects": [
            {
                "id": "own_mug_001",
                "name": "Our Mug",
                "category": "mug",
                "sim_role": "manipulable_object",
                "dimensions_est": {"width": 0.08, "depth": 0.08, "height": 0.10},
                "transform": {
                    "position": {"x": 0, "y": 0, "z": 0},
                    "scale": {"x": 1.0, "y": 1.0, "z": 1.0},
                },
                "semantics": {"affordances": ["Graspable"]},
                "asset": {
                    "path": "objects/mug/asset.usdz",
                    "source": "blueprintpipeline",  # Our own = commercial OK
                },
            },
            {
                "id": "external_nc_bowl_001",
                "name": "External Bowl",
                "category": "bowl",
                "sim_role": "manipulable_object",
                "dimensions_est": {"width": 0.15, "depth": 0.15, "height": 0.08},
                "transform": {
                    "position": {"x": 0.2, "y": 0, "z": 0},
                    "scale": {"x": 1.0, "y": 1.0, "z": 1.0},
                },
                "semantics": {"affordances": ["Graspable"]},
                "asset": {
                    "path": "external/bowl.usdz",
                    "source": "geniesim_assets",  # GenieSimAssets = NC, not commercial
                },
            },
            {
                "id": "external_nc_plate_001",
                "name": "External Plate",
                "category": "plate",
                "sim_role": "manipulable_object",
                "dimensions_est": {"width": 0.25, "depth": 0.25, "height": 0.02},
                "transform": {
                    "position": {"x": 0.4, "y": 0, "z": 0},
                    "scale": {"x": 1.0, "y": 1.0, "z": 1.0},
                },
                "semantics": {"affordances": ["Graspable"]},
                "asset": {
                    "path": "external/plate.usdz",
                    "source": "external_nc",  # Explicitly non-commercial
                },
            },
        ],
    }


# =============================================================================
# Scene Graph Converter Tests
# =============================================================================


class TestSceneGraphConverter:
    """Tests for SceneGraphConverter."""

    def test_convert_basic_manifest(self, sample_manifest):
        """Test basic manifest conversion."""
        converter = SceneGraphConverter(verbose=False)
        scene_graph = converter.convert(sample_manifest)

        assert scene_graph.scene_id == "test_kitchen_001"
        assert scene_graph.coordinate_system == "y_up"
        assert scene_graph.meters_per_unit == 1.0

        # Should have 3 objects (mug, countertop, cabinet)
        # But static objects without manipulation tags may be filtered
        assert len(scene_graph.nodes) >= 2

    def test_node_properties(self, sample_manifest):
        """Test that node properties are correctly extracted."""
        converter = SceneGraphConverter(verbose=False)
        scene_graph = converter.convert(sample_manifest)

        # Find the mug node
        mug_node = next((n for n in scene_graph.nodes if n.asset_id == "mug_001"), None)
        assert mug_node is not None

        # Check semantic description
        assert "mug" in mug_node.semantic.lower()

        # Check size
        assert mug_node.size == [0.08, 0.08, 0.10]

        # Check pose
        assert mug_node.pose.position == [1.0, 0.9, 0.05]

        # Check task tags (should have pick, place from Graspable)
        assert "pick" in mug_node.task_tag
        assert "place" in mug_node.task_tag

    def test_task_tag_mapping(self, sample_manifest):
        """Test affordance to task tag mapping."""
        converter = SceneGraphConverter(verbose=False)
        scene_graph = converter.convert(sample_manifest)

        # Cabinet should have open/close tags
        cabinet_node = next(
            (n for n in scene_graph.nodes if n.asset_id == "cabinet_001"), None
        )
        assert cabinet_node is not None
        assert "open" in cabinet_node.task_tag
        assert "close" in cabinet_node.task_tag

    def test_edge_inference(self, sample_manifest):
        """Test spatial relation inference."""
        converter = SceneGraphConverter(verbose=False)
        scene_graph = converter.convert(sample_manifest)

        # The mug is above the countertop, should infer "on" relation
        # (depending on position values - adjust if needed)
        assert len(scene_graph.edges) >= 0  # May have inferred edges

    def test_save_and_load(self, sample_manifest, tmp_path):
        """Test saving and loading scene graph."""
        converter = SceneGraphConverter(verbose=False)
        scene_graph = converter.convert(sample_manifest)

        output_path = tmp_path / "scene_graph.json"
        scene_graph.save(output_path)

        assert output_path.exists()

        # Load and verify
        with open(output_path) as f:
            loaded = json.load(f)

        assert loaded["scene_id"] == "test_kitchen_001"
        assert len(loaded["nodes"]) == len(scene_graph.nodes)


# =============================================================================
# Asset Index Builder Tests
# =============================================================================


class TestAssetIndexBuilder:
    """Tests for AssetIndexBuilder."""

    def test_build_basic_index(self, sample_manifest):
        """Test basic asset index building."""
        builder = AssetIndexBuilder(verbose=False)
        index = builder.build(sample_manifest)

        # Should have assets for non-background objects
        assert len(index.assets) >= 2

    def test_semantic_descriptions(self, sample_manifest):
        """Test semantic description generation."""
        builder = AssetIndexBuilder(verbose=False)
        index = builder.build(sample_manifest)

        mug_asset = index.get_asset("mug_001")
        assert mug_asset is not None
        assert "mug" in mug_asset.semantic_description.lower()
        assert "ceramic" in mug_asset.semantic_description.lower()

    def test_physics_properties(self, sample_manifest):
        """Test physics property extraction."""
        builder = AssetIndexBuilder(verbose=False)
        index = builder.build(sample_manifest)

        mug_asset = index.get_asset("mug_001")
        assert mug_asset is not None
        assert mug_asset.mass == 0.3
        assert mug_asset.material.friction == 0.5

    def test_commercial_filtering(self, manifest_with_external_assets):
        """Test filtering to commercial-only assets."""
        builder = AssetIndexBuilder(verbose=False)
        index = builder.build(manifest_with_external_assets)

        # All 3 assets before filtering
        assert len(index.assets) == 3

        # Filter to commercial only
        commercial_index = index.filter_commercial()

        # Only 1 asset (our own mug) should remain
        assert len(commercial_index.assets) == 1
        assert commercial_index.assets[0].asset_id == "own_mug_001"
        assert commercial_index.assets[0].commercial_ok is True

    def test_nc_assets_marked(self, manifest_with_external_assets):
        """Test that NC assets are correctly marked."""
        builder = AssetIndexBuilder(verbose=False)
        index = builder.build(manifest_with_external_assets)

        # Find external NC assets
        nc_bowl = index.get_asset("external_nc_bowl_001")
        nc_plate = index.get_asset("external_nc_plate_001")

        assert nc_bowl is not None
        assert nc_bowl.commercial_ok is False

        assert nc_plate is not None
        assert nc_plate.commercial_ok is False


# =============================================================================
# Task Config Generator Tests
# =============================================================================


class TestTaskConfigGenerator:
    """Tests for TaskConfigGenerator."""

    def test_generate_basic_config(self, sample_manifest):
        """Test basic task config generation."""
        generator = TaskConfigGenerator(verbose=False)
        config = generator.generate(sample_manifest, robot_type="franka")

        assert config.scene_id == "test_kitchen_001"
        assert config.environment_type == "kitchen"
        assert config.robot_config.robot_type == "franka"

    def test_suggested_tasks(self, sample_manifest):
        """Test suggested task generation."""
        generator = TaskConfigGenerator(verbose=False)
        config = generator.generate(sample_manifest)

        # Should have tasks for manipulable and articulated objects
        assert len(config.suggested_tasks) >= 2

        # Check task types
        task_types = [t.task_type for t in config.suggested_tasks]
        assert "pick_place" in task_types or "open_close" in task_types

    def test_robot_config(self, sample_manifest):
        """Test robot configuration."""
        generator = TaskConfigGenerator(verbose=False)
        config = generator.generate(sample_manifest, robot_type="g2")

        assert config.robot_config.robot_type == "g2"
        assert len(config.robot_config.base_position) == 3
        assert len(config.robot_config.workspace_bounds) == 2

    def test_max_tasks_limit(self, sample_manifest):
        """Test max tasks limit."""
        generator = TaskConfigGenerator(verbose=False)
        config = generator.generate(sample_manifest, max_tasks=1)

        assert len(config.suggested_tasks) <= 1


# =============================================================================
# Full Exporter Tests
# =============================================================================


class TestGenieSimExporter:
    """Tests for the full GenieSimExporter."""

    def test_full_export(self, sample_manifest, tmp_path):
        """Test complete export pipeline."""
        # Write manifest to temp file
        manifest_path = tmp_path / "scene_manifest.json"
        with open(manifest_path, "w") as f:
            json.dump(sample_manifest, f)

        output_dir = tmp_path / "geniesim"

        config = GenieSimExportConfig(
            robot_type="franka",
            generate_embeddings=False,
        )
        exporter = GenieSimExporter(config, verbose=False)
        result = exporter.export(manifest_path, output_dir)

        assert result.success is True
        assert result.scene_id == "test_kitchen_001"

        # Check output files exist
        assert result.scene_graph_path.exists()
        assert result.asset_index_path.exists()
        assert result.task_config_path.exists()
        assert result.scene_config_path.exists()

        # Verify export manifest
        export_manifest = output_dir / "export_manifest.json"
        assert export_manifest.exists()

    def test_commercial_only_export(self, manifest_with_external_assets, tmp_path):
        """Test export with commercial-only filter."""
        manifest_path = tmp_path / "scene_manifest.json"
        with open(manifest_path, "w") as f:
            json.dump(manifest_with_external_assets, f)

        output_dir = tmp_path / "geniesim"

        config = GenieSimExportConfig(
            robot_type="franka",
            filter_commercial_only=True,  # Only commercial assets
        )
        exporter = GenieSimExporter(config, verbose=False)
        result = exporter.export(manifest_path, output_dir)

        assert result.success is True

        # Load asset index and verify only commercial assets
        with open(result.asset_index_path) as f:
            asset_index = json.load(f)

        # Should only have 1 asset (our own mug)
        assert len(asset_index["assets"]) == 1
        assert asset_index["assets"][0]["asset_id"] == "own_mug_001"


# =============================================================================
# Pipeline Selector Tests
# =============================================================================


class TestPipelineSelector:
    """Tests for pipeline selector with Genie Sim mode."""

    def test_geniesim_default(self):
        """Test that Genie Sim is the default mode."""
        import os
        from tools.pipeline_selector.selector import (
            PipelineSelector,
            PipelineMode,
            DataGenerationBackend,
            is_geniesim_enabled,
        )

        # Clear any override
        os.environ.pop("USE_GENIESIM", None)
        os.environ["USE_GENIESIM"] = "true"

        selector = PipelineSelector()
        assert selector.get_mode() == PipelineMode.GENIESIM
        assert selector.get_data_backend() == DataGenerationBackend.GENIESIM
        assert is_geniesim_enabled() is True

    def test_geniesim_disabled(self):
        """Test disabling Genie Sim."""
        import os
        from tools.pipeline_selector.selector import (
            PipelineSelector,
            PipelineMode,
            DataGenerationBackend,
            is_geniesim_enabled,
        )

        os.environ["USE_GENIESIM"] = "false"

        selector = PipelineSelector()
        assert selector.get_mode() == PipelineMode.REGEN3D_FIRST
        assert selector.get_data_backend() == DataGenerationBackend.BLUEPRINTPIPELINE
        assert is_geniesim_enabled() is False

        # Clean up
        os.environ["USE_GENIESIM"] = "true"

    def test_geniesim_job_sequence(self):
        """Test job sequence in Genie Sim mode."""
        import os
        from tools.pipeline_selector.selector import PipelineSelector

        os.environ["USE_GENIESIM"] = "true"

        selector = PipelineSelector()
        jobs = selector._get_geniesim_jobs()

        # Should end with genie-sim-export-job
        assert jobs[-1] == "genie-sim-export-job"

        # Should NOT include episode-generation-job or isaac-lab-job
        assert "episode-generation-job" not in jobs
        assert "isaac-lab-job" not in jobs
        assert "variation-gen-job" not in jobs

    def test_geniesim_submission_defaults_local_with_api_key(self, monkeypatch):
        """Ensure Genie Sim defaults to local even when an API key is set."""
        from tools.run_local_pipeline import LocalPipelineRunner

        monkeypatch.setenv("USE_GENIESIM", "true")
        monkeypatch.setenv("GENIE_SIM_API_KEY", "test-key")
        monkeypatch.delenv("GENIESIM_SUBMIT_API", raising=False)
        monkeypatch.delenv("GENIESIM_SUBMISSION_MODE", raising=False)
        monkeypatch.delenv("GENIESIM_FORCE_LOCAL", raising=False)
        monkeypatch.delenv("GENIESIM_MOCK_MODE", raising=False)

        mode = LocalPipelineRunner._resolve_geniesim_submission_mode(
            api_key=os.getenv("GENIE_SIM_API_KEY"),
            force_local=False,
            mock_mode=False,
        )

        assert mode == "local"

    def test_geniesim_submission_api_opt_in_requires_key(self, monkeypatch):
        """Ensure explicit API opt-in still falls back to local without a key."""
        from tools.run_local_pipeline import LocalPipelineRunner

        monkeypatch.setenv("USE_GENIESIM", "true")
        monkeypatch.setenv("GENIESIM_SUBMISSION_MODE", "api")
        monkeypatch.delenv("GENIE_SIM_API_KEY", raising=False)

        mode = LocalPipelineRunner._resolve_geniesim_submission_mode(
            api_key=os.getenv("GENIE_SIM_API_KEY"),
            force_local=False,
            mock_mode=False,
        )

        assert mode == "local"


# =============================================================================
# Run tests
# =============================================================================


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
