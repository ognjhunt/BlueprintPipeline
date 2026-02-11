from __future__ import annotations

import json
from pathlib import Path

from tools.regen3d_adapter.adapter import Regen3DObject, Regen3DOutput, Regen3DPose
from tools.run_local_pipeline import LocalPipelineRunner


def _make_object(obj_id: str, mesh_path: str, sim_role: str = "unknown") -> Regen3DObject:
    return Regen3DObject(
        id=obj_id,
        mesh_path=mesh_path,
        pose=Regen3DPose(transform_matrix=[[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]),
        bounds={"min": [0, 0, 0], "max": [1, 1, 1], "center": [0.5, 0.5, 0.5], "size": [1, 1, 1]},
        sim_role=sim_role,
    )


def test_stage1_quality_gate_fails_without_background(tmp_path: Path) -> None:
    runner = LocalPipelineRunner(scene_dir=tmp_path / "scene", verbose=False, skip_interactive=True)
    regen3d_output = Regen3DOutput(
        scene_id="s1",
        objects=[_make_object("obj_0", str(tmp_path / "obj.glb"))],
        background=None,
    )
    manifest = {"objects": [{"id": "obj_0"}]}
    layout = {"objects": [{"id": "obj_0"}]}

    runner._inspect_glb_metadata = lambda _path: {  # type: ignore[method-assign]
        "exists": True,
        "meshes": 1,
        "materials": 1,
        "textures": 1,
        "parse_error": None,
    }
    ok, message, details = runner._validate_stage1_quality(
        regen3d_output=regen3d_output,
        manifest=manifest,
        layout=layout,
    )
    assert not ok
    assert "Background mesh is missing" in message
    assert details["has_background"] is False


def test_stage1_quality_gate_passes_in_compat_mode_without_textures(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("REGEN3D_QUALITY_MODE", "compat")
    monkeypatch.delenv("REGEN3D_ALLOW_TEXTURELESS", raising=False)
    runner = LocalPipelineRunner(scene_dir=tmp_path / "scene", verbose=False, skip_interactive=True)
    regen3d_output = Regen3DOutput(
        scene_id="s1",
        objects=[_make_object("obj_0", str(tmp_path / "obj.glb"))],
        background=_make_object("scene_background", str(tmp_path / "bg.glb"), sim_role="background"),
    )
    manifest = {"objects": [{"id": "obj_0"}, {"id": "scene_background"}]}
    layout = {"objects": [{"id": "obj_0"}, {"id": "scene_background"}]}

    runner._inspect_glb_metadata = lambda _path: {  # type: ignore[method-assign]
        "exists": True,
        "meshes": 1,
        "materials": 1,
        "textures": 0,
        "parse_error": None,
    }
    ok, message, details = runner._validate_stage1_quality(
        regen3d_output=regen3d_output,
        manifest=manifest,
        layout=layout,
    )
    assert ok
    assert message == "Stage 1 quality gate passed"
    assert details["mesh_stats"]["with_materials"] == 1


def test_stage1_quality_gate_fails_textureless_in_quality_mode(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("REGEN3D_QUALITY_MODE", "quality")
    monkeypatch.delenv("REGEN3D_ALLOW_TEXTURELESS", raising=False)
    runner = LocalPipelineRunner(scene_dir=tmp_path / "scene", verbose=False, skip_interactive=True)
    regen3d_output = Regen3DOutput(
        scene_id="s1",
        objects=[_make_object("obj_0", str(tmp_path / "obj.glb"))],
        background=_make_object("scene_background", str(tmp_path / "bg.glb"), sim_role="background"),
    )
    manifest = {"objects": [{"id": "obj_0"}, {"id": "scene_background"}]}
    layout = {"objects": [{"id": "obj_0"}, {"id": "scene_background"}]}

    runner._inspect_glb_metadata = lambda _path: {  # type: ignore[method-assign]
        "exists": True,
        "meshes": 1,
        "materials": 1,
        "textures": 0,
        "parse_error": None,
    }
    ok, message, details = runner._validate_stage1_quality(
        regen3d_output=regen3d_output,
        manifest=manifest,
        layout=layout,
    )
    assert not ok
    assert "no texture definitions" in message
    assert details["quality_mode"] == "quality"
    assert details["allow_textureless_override"] is False


def test_stage1_quality_gate_allows_textureless_with_override(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("REGEN3D_QUALITY_MODE", "quality")
    monkeypatch.setenv("REGEN3D_ALLOW_TEXTURELESS", "true")
    runner = LocalPipelineRunner(scene_dir=tmp_path / "scene", verbose=False, skip_interactive=True)
    regen3d_output = Regen3DOutput(
        scene_id="s1",
        objects=[_make_object("obj_0", str(tmp_path / "obj.glb"))],
        background=_make_object("scene_background", str(tmp_path / "bg.glb"), sim_role="background"),
    )
    manifest = {"objects": [{"id": "obj_0"}, {"id": "scene_background"}]}
    layout = {"objects": [{"id": "obj_0"}, {"id": "scene_background"}]}

    runner._inspect_glb_metadata = lambda _path: {  # type: ignore[method-assign]
        "exists": True,
        "meshes": 1,
        "materials": 1,
        "textures": 0,
        "parse_error": None,
    }
    ok, message, details = runner._validate_stage1_quality(
        regen3d_output=regen3d_output,
        manifest=manifest,
        layout=layout,
    )
    assert ok
    assert message == "Stage 1 quality gate passed"
    assert details["allow_textureless_override"] is True


def test_stage1_quality_gate_allows_materialless_with_override(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("REGEN3D_ALLOW_MATERIALLESS", "true")
    runner = LocalPipelineRunner(scene_dir=tmp_path / "scene", verbose=False, skip_interactive=True)
    regen3d_output = Regen3DOutput(
        scene_id="s1",
        objects=[_make_object("obj_0", str(tmp_path / "obj.glb"))],
        background=_make_object("scene_background", str(tmp_path / "bg.glb"), sim_role="background"),
    )
    manifest = {"objects": [{"id": "obj_0"}, {"id": "scene_background"}]}
    layout = {"objects": [{"id": "obj_0"}, {"id": "scene_background"}]}

    runner._inspect_glb_metadata = lambda _path: {  # type: ignore[method-assign]
        "exists": True,
        "meshes": 1,
        "materials": 0,
        "textures": 1,
        "parse_error": None,
    }
    ok, message, details = runner._validate_stage1_quality(
        regen3d_output=regen3d_output,
        manifest=manifest,
        layout=layout,
    )
    assert ok
    assert message == "Stage 1 quality gate passed"
    assert details["allow_materialless_override"] is True


def test_stage1_quality_gate_rejects_materialless_override_in_production(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("REGEN3D_ALLOW_MATERIALLESS", "true")
    runner = LocalPipelineRunner(scene_dir=tmp_path / "scene", verbose=False, skip_interactive=True)
    runner._is_production_mode = lambda: True  # type: ignore[method-assign]
    regen3d_output = Regen3DOutput(
        scene_id="s1",
        objects=[_make_object("obj_0", str(tmp_path / "obj.glb"))],
        background=_make_object("scene_background", str(tmp_path / "bg.glb"), sim_role="background"),
    )
    manifest = {"objects": [{"id": "obj_0"}, {"id": "scene_background"}]}
    layout = {"objects": [{"id": "obj_0"}, {"id": "scene_background"}]}

    runner._inspect_glb_metadata = lambda _path: {  # type: ignore[method-assign]
        "exists": True,
        "meshes": 1,
        "materials": 0,
        "textures": 1,
        "parse_error": None,
    }
    ok, message, _ = runner._validate_stage1_quality(
        regen3d_output=regen3d_output,
        manifest=manifest,
        layout=layout,
    )
    assert not ok
    assert "REGEN3D_ALLOW_MATERIALLESS is not allowed in production mode." in message


def test_stage1_quality_gate_fails_when_combined_scene_lacks_materials(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("REGEN3D_QUALITY_MODE", "quality")
    monkeypatch.delenv("REGEN3D_ALLOW_TEXTURELESS", raising=False)
    monkeypatch.delenv("REGEN3D_ALLOW_MATERIALLESS", raising=False)
    runner = LocalPipelineRunner(scene_dir=tmp_path / "scene", verbose=False, skip_interactive=True)
    regen3d_output = Regen3DOutput(
        scene_id="s1",
        objects=[_make_object("obj_0", str(tmp_path / "obj.glb"))],
        background=_make_object("scene_background", str(tmp_path / "bg.glb"), sim_role="background"),
    )
    manifest = {"objects": [{"id": "obj_0"}, {"id": "scene_background"}]}
    layout = {"objects": [{"id": "obj_0"}, {"id": "scene_background"}]}

    def _inspect(path: Path) -> dict:
        if path.name == "combined_scene.glb":
            return {
                "exists": True,
                "meshes": 1,
                "materials": 0,
                "textures": 0,
                "parse_error": None,
            }
        return {
            "exists": True,
            "meshes": 1,
            "materials": 1,
            "textures": 1,
            "parse_error": None,
        }

    runner._inspect_glb_metadata = _inspect  # type: ignore[method-assign]
    ok, message, details = runner._validate_stage1_quality(
        regen3d_output=regen3d_output,
        manifest=manifest,
        layout=layout,
    )
    assert not ok
    assert "Combined scene GLB contains no material definitions." in message
    assert details["combined_scene"]["materials"] == 0


def test_stage1_quality_gate_fails_when_foreground_count_exceeds_cap(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("REGEN3D_QUALITY_MODE", "quality")
    runner = LocalPipelineRunner(scene_dir=tmp_path / "scene", verbose=False, skip_interactive=True)
    objects = [
        _make_object(f"obj_{idx}", str(tmp_path / f"obj_{idx}.glb"))
        for idx in range(19)
    ]
    regen3d_output = Regen3DOutput(
        scene_id="s1",
        objects=objects,
        background=_make_object("scene_background", str(tmp_path / "bg.glb"), sim_role="background"),
    )
    manifest = {"objects": [{"id": obj.id} for obj in objects] + [{"id": "scene_background"}]}
    layout = {"objects": [{"id": obj.id} for obj in objects] + [{"id": "scene_background"}]}

    runner._inspect_glb_metadata = lambda _path: {  # type: ignore[method-assign]
        "exists": True,
        "meshes": 1,
        "materials": 1,
        "textures": 1,
        "parse_error": None,
    }
    ok, message, details = runner._validate_stage1_quality(
        regen3d_output=regen3d_output,
        manifest=manifest,
        layout=layout,
    )
    assert not ok
    assert "Foreground object count exceeds Stage 1 cap" in message
    assert details["foreground_object_count"] == 19


def test_stage1_quality_gate_fails_when_class_repetition_exceeds_cap(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("REGEN3D_QUALITY_MODE", "quality")
    runner = LocalPipelineRunner(scene_dir=tmp_path / "scene", verbose=False, skip_interactive=True)
    repeated = [
        _make_object(
            f"detergent_bottle__({200 + idx}, {170 + idx})",
            str(tmp_path / f"obj_{idx}.glb"),
        )
        for idx in range(5)
    ]
    regen3d_output = Regen3DOutput(
        scene_id="s1",
        objects=repeated,
        background=_make_object("scene_background", str(tmp_path / "bg.glb"), sim_role="background"),
    )
    manifest = {"objects": [{"id": obj.id} for obj in repeated] + [{"id": "scene_background"}]}
    layout = {"objects": [{"id": obj.id} for obj in repeated] + [{"id": "scene_background"}]}

    runner._inspect_glb_metadata = lambda _path: {  # type: ignore[method-assign]
        "exists": True,
        "meshes": 1,
        "materials": 1,
        "textures": 1,
        "parse_error": None,
    }
    ok, message, details = runner._validate_stage1_quality(
        regen3d_output=regen3d_output,
        manifest=manifest,
        layout=layout,
    )
    assert not ok
    assert "Foreground class repetition exceeds Stage 1 cap" in message
    assert details["foreground_class_repeat_offenders"]["detergent_bottle"] == 5


def test_stage1_quality_gate_fails_when_mask_stats_detect_zero_area(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("REGEN3D_QUALITY_MODE", "quality")
    runner = LocalPipelineRunner(scene_dir=tmp_path / "scene", verbose=False, skip_interactive=True)
    regen3d_output = Regen3DOutput(
        scene_id="s1",
        objects=[_make_object("obj_0", str(tmp_path / "obj.glb"))],
        background=_make_object("scene_background", str(tmp_path / "bg.glb"), sim_role="background"),
    )
    manifest = {"objects": [{"id": "obj_0"}, {"id": "scene_background"}]}
    layout = {"objects": [{"id": "obj_0"}, {"id": "scene_background"}]}

    runner._inspect_glb_metadata = lambda _path: {  # type: ignore[method-assign]
        "exists": True,
        "meshes": 1,
        "materials": 1,
        "textures": 1,
        "parse_error": None,
    }
    runner._inspect_mask_quality = lambda *_args, **_kwargs: {  # type: ignore[method-assign]
        "file_count": 12,
        "checked": 12,
        "zero_area": 1,
        "tiny_area": 5,
    }

    ok, message, _ = runner._validate_stage1_quality(
        regen3d_output=regen3d_output,
        manifest=manifest,
        layout=layout,
    )
    assert not ok
    assert "Detected 1 zero-area mask(s)." in message


def test_stage1_quality_report_written(tmp_path: Path) -> None:
    runner = LocalPipelineRunner(scene_dir=tmp_path / "scene", verbose=False, skip_interactive=True)
    report_path = runner._write_stage1_quality_report(
        quality_ok=False,
        quality_message="Stage 1 quality gate failed: missing background",
        quality_details={"issues": ["Background mesh is missing."]},
    )
    assert report_path.is_file()
    payload = json.loads(report_path.read_text())
    assert payload["scene_id"] == runner.scene_id
    assert payload["status"] == "fail"
    assert payload["details"]["issues"] == ["Background mesh is missing."]
