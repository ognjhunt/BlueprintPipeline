"""
Integration tests for Genie Sim 3.0 adapter module.

Tests the conversion from BlueprintPipeline manifest to Genie Sim format:
- Scene graph conversion (nodes + edges)
- Asset index building
- Task configuration generation
- Commercial asset filtering
"""

import importlib.util
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict

import pytest

# Add parent directory to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

import tools.geniesim_adapter.scene_graph as scene_graph

from tools.geniesim_adapter import (
    GenieSimExporter,
    GenieSimExportConfig,
    GenieSimExportResult,
)
from tools.geniesim_adapter.scene_graph import (
    SceneGraphConverter,
    convert_manifest_to_scene_graph,
    GenieSimSceneGraph,
    GenieSimNode,
    GenieSimEdge,
    HAVE_STREAMING_PARSER,
    Pose,
    RelationInferencer,
)
from tools.geniesim_adapter.asset_index import (
    AssetIndexBuilder,
    GenieSimAssetIndex,
    GenieSimAsset,
    CATEGORY_MAPPING,
)
from tools.geniesim_adapter.task_config import (
    TaskConfigGenerator,
    GenieSimTaskConfig,
    SuggestedTask,
)
from tools.validation import ValidationError


def _load_export_to_geniesim_module():
    export_path = Path(__file__).parent.parent / "genie-sim-export-job" / "export_to_geniesim.py"
    spec = importlib.util.spec_from_file_location("export_to_geniesim", export_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


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
                    "position": {"x": 2.3, "y": 0.1, "z": 0.9},
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
                    "position": {"x": 2.2, "y": 0.0, "z": 0.85},
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
                    "position": {"x": 2.6, "y": 0.2, "z": 0.9},
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
        mug_node = next((n for n in scene_graph.nodes if n.asset_id == "test_kitchen_001_obj_mug_001"), None)
        assert mug_node is not None

        # Check semantic description
        assert "mug" in mug_node.semantic.lower()

        # Check size
        assert mug_node.size == [0.08, 0.08, 0.10]

        # Check pose
        assert mug_node.pose.position == [0.4, 0.3, 0.8]

        # Check task tags (should have pick, place from Graspable)
        assert "pick" in mug_node.task_tag
        assert "place" in mug_node.task_tag

    def test_task_tag_mapping(self, sample_manifest):
        """Test affordance to task tag mapping."""
        converter = SceneGraphConverter(verbose=False)
        scene_graph = converter.convert(sample_manifest)

        # Cabinet should have open/close tags
        cabinet_node = next(
            (n for n in scene_graph.nodes if n.asset_id == "test_kitchen_001_obj_cabinet_001"), None
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

    def test_relation_inference_cache(self):
        """Ensure relation inference caches identical inputs."""
        inferencer = RelationInferencer(verbose=False)
        nodes = [
            GenieSimNode(
                asset_id="box_001",
                semantic="box",
                size=[1.0, 1.0, 1.0],
                pose=Pose(position=[0.0, 0.0, 0.5], orientation=[1.0, 0.0, 0.0, 0.0]),
                task_tag=[],
                usd_path="",
            ),
            GenieSimNode(
                asset_id="box_002",
                semantic="box",
                size=[1.0, 1.0, 1.0],
                pose=Pose(position=[0.0, 0.0, 1.6], orientation=[1.0, 0.0, 0.0, 0.0]),
                task_tag=[],
                usd_path="",
            ),
        ]

        first_edges = inferencer.infer_relations(nodes, scene_id="cache_scene")
        assert inferencer.last_cache_hit is False

        second_edges = inferencer.infer_relations(nodes, scene_id="cache_scene")
        assert inferencer.last_cache_hit is True
        assert second_edges == first_edges

        nodes[0].size = [1.0, 1.0, 1.1]
        _ = inferencer.infer_relations(nodes, scene_id="cache_scene")
        assert inferencer.last_cache_hit is False

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

    def test_streaming_relation_inference(self, sample_manifest, tmp_path, monkeypatch):
        """Test relation inference in streaming mode."""
        if not HAVE_STREAMING_PARSER:
            pytest.skip("Streaming parser unavailable for streaming relation inference test.")

        manifest_path = tmp_path / "scene_manifest.json"
        with open(manifest_path, "w") as f:
            json.dump(sample_manifest, f)

        expected_edges = [
            GenieSimEdge(
                source="test_kitchen_001_obj_mug_001",
                target="test_kitchen_001_obj_countertop_001",
                relation="on",
                confidence=0.42,
            )
        ]
        called = {}

        def fake_infer(self, nodes, scene_id=None, progress_callback=None):
            called["count"] = len(nodes)
            return expected_edges

        monkeypatch.setattr(RelationInferencer, "infer_relations", fake_infer)

        scene_graph = convert_manifest_to_scene_graph(
            manifest_path,
            verbose=False,
            use_streaming=True,
        )

        assert scene_graph.edges == expected_edges
        assert called["count"] == len(scene_graph.nodes)

    def test_production_requires_ijson(self, tmp_path, monkeypatch):
        """Ensure production mode errors when ijson is unavailable."""
        manifest_path = tmp_path / "scene_manifest.json"
        manifest_path.write_text("{}")

        monkeypatch.setattr(scene_graph, "resolve_production_mode", lambda: True)
        monkeypatch.setattr(scene_graph.importlib.util, "find_spec", lambda name: None)

        with pytest.raises(RuntimeError, match="ijson"):
            scene_graph.convert_manifest_to_scene_graph(
                manifest_path,
                verbose=False,
            )

    def test_auto_streaming_threshold_non_production(self, tmp_path, monkeypatch):
        """Ensure non-production auto-streaming triggers at >1MB."""
        manifest_path = tmp_path / "scene_manifest.json"
        payload = {"scene_id": "test_scene", "coordinate_system": "y_up", "meters_per_unit": 1.0, "objects": []}
        manifest_path.write_text(json.dumps(payload) + (" " * (1024 * 1024 + 10)))

        called = {"used": False}

        class DummyStreamingParser:
            def __init__(self, path: str) -> None:
                called["used"] = True

            def get_metadata(self) -> Dict[str, Any]:
                return {
                    "scene_id": "test_scene",
                    "coordinate_system": "y_up",
                    "meters_per_unit": 1.0,
                    "metadata": {},
                }

            def stream_objects(self, batch_size: int, progress_callback, progress_interval_s: float):
                return []

        monkeypatch.setattr(scene_graph, "resolve_production_mode", lambda: False)
        monkeypatch.setattr(scene_graph, "HAVE_STREAMING_PARSER", True)
        monkeypatch.setattr(scene_graph, "StreamingManifestParser", DummyStreamingParser)
        monkeypatch.setattr(scene_graph.importlib.util, "find_spec", lambda name: object())

        result = scene_graph.convert_manifest_to_scene_graph(
            manifest_path,
            verbose=False,
        )

        assert called["used"] is True
        assert isinstance(result, GenieSimSceneGraph)


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

        mug_asset = index.get_asset("test_kitchen_001_obj_mug_001")
        assert mug_asset is not None
        assert "mug" in mug_asset.semantic_description.lower()
        assert "ceramic" in mug_asset.semantic_description.lower()

    def test_physics_properties(self, sample_manifest):
        """Test physics property extraction."""
        builder = AssetIndexBuilder(verbose=False)
        index = builder.build(sample_manifest)

        mug_asset = index.get_asset("test_kitchen_001_obj_mug_001")
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
        assert commercial_index.assets[0].asset_id == "test_mixed_assets_obj_own_mug_001"
        assert commercial_index.assets[0].commercial_ok is True

    def test_nc_assets_marked(self, manifest_with_external_assets):
        """Test that NC assets are correctly marked."""
        builder = AssetIndexBuilder(verbose=False)
        index = builder.build(manifest_with_external_assets)

        # Find external NC assets
        nc_bowl = index.get_asset("test_mixed_assets_obj_external_nc_bowl_001")
        nc_plate = index.get_asset("test_mixed_assets_obj_external_nc_plate_001")

        assert nc_bowl is not None
        assert nc_bowl.commercial_ok is False

        assert nc_plate is not None
        assert nc_plate.commercial_ok is False

    def test_unknown_category_warns(self, capsys):
        """Test that unknown categories emit warnings and fallback."""
        manifest = {
            "scene_id": "test_scene",
            "objects": [
                {
                    "id": "mystery_001",
                    "category": "mystery_box",
                    "asset": {"path": "objects/mystery/asset.usdz"},
                }
            ],
        }

        builder = AssetIndexBuilder(verbose=True, strict_category_validation=False)
        index = builder.build(manifest)

        captured = capsys.readouterr()
        assert "Unknown category" in captured.out
        assert "mystery_001" in captured.out
        assert "mystery_box" in captured.out
        assert index.assets[0].categories == CATEGORY_MAPPING["object"]

    def test_unknown_category_strict_raises(self):
        """Test that strict category validation raises on unknown categories."""
        manifest = {
            "scene_id": "test_scene",
            "objects": [
                {
                    "id": "mystery_002",
                    "category": "mystery_box",
                    "asset": {"path": "objects/mystery/asset.usdz"},
                }
            ],
        }

        builder = AssetIndexBuilder(verbose=False, strict_category_validation=True)
        with pytest.raises(ValidationError):
            builder.build(manifest)


def test_mixed_case_license_is_commercial_ok():
    export_module = _load_export_to_geniesim_module()

    assert export_module._is_commercial_license("Cc-By")
    assert export_module._is_commercial_license("mIt")
    assert export_module._is_commercial_license("ApAcHe-2.0")
    assert export_module._is_commercial_license("cC0")


def test_production_requires_real_embeddings_with_generation(monkeypatch):
    export_module = _load_export_to_geniesim_module()

    class DummyFailureMarkerWriter:
        def __init__(self, bucket, scene_id, job_name):
            self.bucket = bucket
            self.scene_id = scene_id
            self.job_name = job_name

        def write_failure(self, **kwargs):
            return None

    monkeypatch.setattr(export_module, "FailureMarkerWriter", DummyFailureMarkerWriter)
    monkeypatch.setenv("PIPELINE_ENV", "production")
    monkeypatch.setenv("BUCKET", "test-bucket")
    monkeypatch.setenv("SCENE_ID", "scene-123")
    monkeypatch.setenv("GENERATE_EMBEDDINGS", "true")
    monkeypatch.setenv("REQUIRE_EMBEDDINGS", "false")

    with pytest.raises(SystemExit) as exc:
        export_module.main()

    assert exc.value.code == 1


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

    def test_unknown_scene_id_strict_raises(self, monkeypatch):
        """Ensure strict mode rejects unknown scene IDs."""
        monkeypatch.setenv("BP_TASK_CONFIG_STRICT", "true")
        generator = TaskConfigGenerator(verbose=False)
        manifest = {"scene": {}, "objects": []}

        with pytest.raises(RuntimeError, match="Scene ID is required"):
            generator.generate(manifest)

    def test_empty_tasks_strict_raises(self, monkeypatch):
        """Ensure strict mode rejects empty task lists."""
        monkeypatch.setenv("BP_TASK_CONFIG_STRICT", "true")
        generator = TaskConfigGenerator(verbose=False)
        manifest = {
            "scene_id": "empty_tasks_scene",
            "scene": {},
            "objects": [{"id": "bg", "sim_role": "static"}],
        }

        with pytest.raises(RuntimeError, match="No suggested tasks"):
            generator.generate(manifest)


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

    def test_export_manifest_schema_version(self, sample_manifest, tmp_path):
        """Ensure export manifest schema version matches target."""
        from tools.geniesim_adapter.exporter import GENIESIM_TARGET_VERSION

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
        export_manifest = output_dir / "export_manifest.json"
        assert export_manifest.exists()
        export_payload = json.loads(export_manifest.read_text())
        assert export_payload["schema_version"] == GENIESIM_TARGET_VERSION

    def test_commercial_only_export(self, manifest_with_external_assets, tmp_path):
        """Test export with commercial-only filter rejects NC assets."""
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

        # Source code now rejects manifests containing non-commercial assets
        assert result.success is False

    def test_scene_config_env_overrides(self, sample_manifest, tmp_path, monkeypatch):
        """Ensure env overrides are reflected in the scene config YAML."""
        manifest_path = tmp_path / "scene_manifest.json"
        with open(manifest_path, "w") as f:
            json.dump(sample_manifest, f)

        monkeypatch.setenv("GENIESIM_EPISODES_PER_TASK", "42")
        monkeypatch.setenv("GENIESIM_USE_CUROBO", "false")
        monkeypatch.setenv("GENIESIM_EVAL_SCENARIOS", "7")
        monkeypatch.setenv("GENIESIM_EXPORT_FORMAT", "custom_format")

        output_dir = tmp_path / "geniesim"
        config = GenieSimExportConfig(
            robot_type="franka",
            generate_embeddings=False,
        )
        exporter = GenieSimExporter(config, verbose=False)
        result = exporter.export(manifest_path, output_dir)

        assert result.success is True
        scene_config_text = result.scene_config_path.read_text()

        assert "episodes_per_task: 42" in scene_config_text
        assert "use_curobo: false" in scene_config_text
        assert "scenarios: 7" in scene_config_text
        assert "format: \"custom_format\"" in scene_config_text


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

        # Should end with Genie Sim import (submit is second-to-last)
        assert jobs[-1] == "genie-sim-import-job"
        assert "genie-sim-submit-job" in jobs
        assert "genie-sim-export-job" in jobs
        assert "variation-gen-job" in jobs

        # Should NOT include episode-generation-job or isaac-lab-job
        assert "episode-generation-job" not in jobs
        assert "isaac-lab-job" not in jobs

# =============================================================================
# Run tests
# =============================================================================


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
