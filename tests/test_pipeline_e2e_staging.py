#!/usr/bin/env python3
"""
Staging End-to-End Integration Tests for BlueprintPipeline.

These tests are meant for labs validating real reconstruction outputs and
running Isaac Sim before production rollouts. They are intentionally gated
behind an explicit environment flag so CI stays lightweight.

Usage (inside Isaac Sim):
    RUN_STAGING_E2E=1 \
    STAGING_SCENE_DIR=/mnt/gcs/scenes/<scene_id> \
    /isaac-sim/python.sh -m pytest tests/test_pipeline_e2e_staging.py -v

Or using data root + scene id:
    RUN_STAGING_E2E=1 \
    STAGING_DATA_ROOT=/mnt/gcs \
    STAGING_SCENE_ID=<scene_id> \
    /isaac-sim/python.sh -m pytest tests/test_pipeline_e2e_staging.py -v
"""

import os
import sys
from pathlib import Path

import pytest

# Add repo root to path
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _staging_enabled() -> bool:
    return os.environ.get("RUN_STAGING_E2E") == "1"


def _resolve_scene_dir() -> Path:
    scene_dir_env = os.environ.get("STAGING_SCENE_DIR")
    if scene_dir_env:
        return Path(scene_dir_env).expanduser().resolve()

    data_root = os.environ.get("STAGING_DATA_ROOT")
    scene_id = os.environ.get("STAGING_SCENE_ID")
    if data_root and scene_id:
        return Path(data_root).expanduser().resolve() / "scenes" / scene_id

    raise RuntimeError(
        "Set STAGING_SCENE_DIR or STAGING_DATA_ROOT + STAGING_SCENE_ID to "
        "point at real reconstruction outputs."
    )


def _require_isaac_sim() -> None:
    try:
        import omni  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "Isaac Sim modules not available. Run with /isaac-sim/python.sh."
        ) from exc

    from tools.run_full_isaacsim_pipeline import initialize_isaac_sim

    if not initialize_isaac_sim(headless=True):
        raise RuntimeError("Failed to initialize Isaac Sim in staging test.")


def _validate_regen3d_inputs(scene_dir: Path) -> None:
    regen3d_dir = scene_dir / "regen3d"
    if not regen3d_dir.is_dir():
        raise AssertionError(f"Missing regen3d output at {regen3d_dir}")

    required_files = [
        regen3d_dir / "scene_info.json",
        regen3d_dir / "objects",
    ]
    for path in required_files:
        if not path.exists():
            raise AssertionError(f"Missing real reconstruction artifact: {path}")


@pytest.mark.staging
def test_staging_pipeline_with_real_reconstruction():
    if not _staging_enabled():
        pytest.skip("Set RUN_STAGING_E2E=1 to run staging Isaac Sim tests.")

    scene_dir = _resolve_scene_dir()
    _validate_regen3d_inputs(scene_dir)

    from tools.run_local_pipeline import LocalPipelineRunner, PipelineStep

    runner = LocalPipelineRunner(
        scene_dir=scene_dir,
        verbose=True,
        skip_interactive=True,
        environment_type=os.environ.get("STAGING_ENVIRONMENT_TYPE", "kitchen"),
    )

    success = runner.run(
        steps=[
            PipelineStep.REGEN3D,
            PipelineStep.SIMREADY,
            PipelineStep.USD,
            PipelineStep.REPLICATOR,
            PipelineStep.ISAAC_LAB,
        ],
        run_validation=True,
    )
    assert success, "Staging pipeline failed"

    usd_path = scene_dir / "usd" / "scene.usda"
    assert usd_path.is_file(), "USD scene missing after pipeline run"

    _require_isaac_sim()

    import omni
    from pxr import UsdGeom

    omni.usd.get_context().open_stage(str(usd_path))
    stage = omni.usd.get_context().get_stage()

    prim_count = 0
    mesh_count = 0
    for prim in stage.Traverse():
        prim_count += 1
        if prim.IsA(UsdGeom.Mesh):
            mesh_count += 1

    assert prim_count > 0, "Isaac Sim stage has no prims"
    assert mesh_count > 0, "Isaac Sim stage has no mesh prims"
