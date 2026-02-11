import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from fixtures.generate_mock_regen3d import create_minimal_glb
from tools.regen3d_runner.output_harvester import (
    HarvestResult,
    harvest_regen3d_native_output,
)
from tools.regen3d_runner.runner import Regen3DConfig, Regen3DRunner
from tools.regen3d_runner.vm_executor import SSHConnectionError, VMConfig, VMExecutor


def _write_native_output(
    native_dir: Path,
    object_names: list[str],
    *,
    include_transforms: bool = True,
) -> None:
    glb_dir = native_dir / "glb"
    scene_dir = glb_dir / "scene"
    scene_dir.mkdir(parents=True, exist_ok=True)

    for name in object_names:
        (glb_dir / f"{name}.glb").write_bytes(create_minimal_glb())

    (scene_dir / "combined_scene.glb").write_bytes(create_minimal_glb())
    if include_transforms:
        transforms = {name: [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]] for name in object_names}
        (scene_dir / "transforms.json").write_text(json.dumps(transforms, indent=2))


def test_generate_and_upload_config_renders_valid_yaml(monkeypatch, tmp_path: Path) -> None:
    config = Regen3DConfig(
        repo_path="/tmp/3D-RE-GEN",
        labels=["chair", "plant in pot"],
        use_vggt=True,
        use_hunyuan21=False,
        gemini_api_key="",
    )
    runner = Regen3DRunner(config=config, verbose=False)

    captured: dict[str, str] = {}

    def _fake_scp_upload(local_path: Path, remote_path: str) -> None:
        if remote_path.endswith("/src/config.yaml"):
            captured["config_text"] = Path(local_path).read_text()

    monkeypatch.setattr(runner._vm, "scp_upload", _fake_scp_upload)
    monkeypatch.setattr(runner._vm, "ssh_exec", lambda *args, **kwargs: (0, "", ""))

    runner._generate_and_upload_config(
        remote_image_path="/tmp/3D-RE-GEN/input_images/scene.jpg",
        remote_output_dir="/tmp/3D-RE-GEN/output_scene",
        scene_id="scene",
    )

    rendered = captured["config_text"]
    parsed = yaml.safe_load(rendered)
    assert parsed["labels"] == ["chair", "plant in pot"]
    assert isinstance(parsed["Use_VGGT"], bool)
    assert isinstance(parsed["use_hunyuan21"], bool)


def test_run_reconstruction_clears_native_output_before_download(
    monkeypatch,
    tmp_path: Path,
) -> None:
    from tools.regen3d_runner import runner as runner_module

    input_image = tmp_path / "input.jpg"
    input_image.write_bytes(b"mock")
    output_dir = tmp_path / "regen3d"
    native_dir = output_dir / "_native_output"
    native_dir.mkdir(parents=True, exist_ok=True)
    (native_dir / "stale.txt").write_text("stale")

    runner = Regen3DRunner(
        config=Regen3DConfig(gemini_api_key="", labels=["chair"]),
        verbose=False,
    )

    monkeypatch.setattr(runner, "_ensure_vm_ready", lambda: None)
    monkeypatch.setattr(runner, "_setup_remote", lambda: None)
    monkeypatch.setattr(runner, "_upload_input_image", lambda *_args, **_kwargs: "/remote/input.jpg")
    monkeypatch.setattr(runner, "_generate_and_upload_config", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(runner, "_clear_remote_segmentation_outputs", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(runner, "_clear_remote_segmentation_outputs", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(runner, "_execute_pipeline", lambda *_args, **_kwargs: (0, "", ""))
    monkeypatch.setattr(runner, "_count_remote_masks", lambda *_args, **_kwargs: 0)
    monkeypatch.setattr(runner, "_create_glb_symlinks", lambda *_args, **_kwargs: None)

    def _fake_download_outputs(_remote_output_dir: str, local_dir: Path) -> None:
        assert not (local_dir / "stale.txt").exists()
        (local_dir / "glb" / "scene").mkdir(parents=True, exist_ok=True)
        (local_dir / "glb" / "scene" / "combined_scene.glb").write_bytes(create_minimal_glb())
        (local_dir / "glb" / "chair.glb").write_bytes(create_minimal_glb())
        transforms = {"chair": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]}
        (local_dir / "glb" / "scene" / "transforms.json").write_text(json.dumps(transforms))

    monkeypatch.setattr(runner, "_download_outputs", _fake_download_outputs)

    def _fake_harvest(**kwargs):
        return HarvestResult(
            target_dir=kwargs["target_dir"],
            objects_count=1,
            has_background=True,
            has_camera=True,
            has_depth=False,
            warnings=[],
        )

    monkeypatch.setattr(runner_module, "harvest_regen3d_native_output", _fake_harvest)

    result = runner.run_reconstruction(
        input_image=input_image,
        scene_id="scene",
        output_dir=output_dir,
    )

    assert result.success
    assert result.objects_count == 1


def test_from_env_defaults_match_reconstruct_config(monkeypatch) -> None:
    monkeypatch.delenv("REGEN3D_SEG_BACKEND", raising=False)
    monkeypatch.delenv("REGEN3D_VM_ZONE", raising=False)
    monkeypatch.delenv("REGEN3D_REPO_PATH", raising=False)
    monkeypatch.delenv("REGEN3D_SETUP_TIMEOUT_S", raising=False)
    monkeypatch.delenv("REGEN3D_QUALITY_MODE", raising=False)
    monkeypatch.delenv("REGEN3D_MAX_LABELS", raising=False)
    monkeypatch.delenv("REGEN3D_MAX_MASKS", raising=False)

    cfg = Regen3DConfig.from_env()
    assert cfg.seg_backend == "grounded_sam"
    assert cfg.vm_zone == "us-east1-c"
    assert cfg.repo_path == "/home/nijelhunt1/3D-RE-GEN"
    assert cfg.setup_timeout_s == 3600
    assert cfg.quality_mode == "quality"
    assert cfg.max_labels == 12
    assert cfg.max_masks == 40


def test_compare_backends_calls_setup_remote(monkeypatch, tmp_path: Path) -> None:
    image_path = tmp_path / "room.png"
    image_path.write_bytes(b"image")
    out_dir = tmp_path / "compare"
    runner = Regen3DRunner(config=Regen3DConfig(labels=["chair"]), verbose=False)

    calls = {"setup_remote": 0}

    monkeypatch.setattr(runner, "_ensure_vm_ready", lambda: None)

    def _fake_setup_remote():
        calls["setup_remote"] += 1

    monkeypatch.setattr(runner, "_setup_remote", _fake_setup_remote)
    monkeypatch.setattr(runner, "_resolve_auto_labels", lambda _img: None)
    monkeypatch.setattr(runner._vm, "scp_upload", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(runner, "_generate_and_upload_config", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(runner, "_execute_pipeline", lambda *_args, **_kwargs: (0, "", ""))
    monkeypatch.setattr(runner, "_count_remote_masks", lambda *_args, **_kwargs: 2)
    monkeypatch.setattr(runner, "_download_findings_if_present", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(
        runner,
        "_run_sam3_segmentation",
        lambda *_args, **_kwargs: {"mask_count": 2, "fallback_used": False, "error": None},
    )
    monkeypatch.setattr(runner._vm, "ssh_exec", lambda *_args, **_kwargs: (0, "", ""))

    results = runner.compare_backends(image_path, out_dir)
    assert calls["setup_remote"] == 1
    assert results["grounded_sam"]["status"] == "success"
    assert results["sam3"]["status"] == "success"
    assert results["success"] is True


def test_compare_backends_reports_sam3_failure(monkeypatch, tmp_path: Path) -> None:
    image_path = tmp_path / "room.png"
    image_path.write_bytes(b"image")
    out_dir = tmp_path / "compare"
    runner = Regen3DRunner(config=Regen3DConfig(labels=["chair"]), verbose=False)

    monkeypatch.setattr(runner, "_ensure_vm_ready", lambda: None)
    monkeypatch.setattr(runner, "_setup_remote", lambda: None)
    monkeypatch.setattr(runner, "_resolve_auto_labels", lambda _img: None)
    monkeypatch.setattr(runner._vm, "scp_upload", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(runner, "_generate_and_upload_config", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(runner, "_execute_pipeline", lambda *_args, **_kwargs: (0, "", ""))
    monkeypatch.setattr(
        runner,
        "_count_remote_masks",
        lambda remote_dir: 2 if "grounded_sam" in remote_dir else 0,
    )
    monkeypatch.setattr(runner, "_download_findings_if_present", lambda *_args, **_kwargs: True)

    def _fail_sam3(*_args, **_kwargs):
        raise RuntimeError("sam3 failed hard")

    monkeypatch.setattr(runner, "_run_sam3_segmentation", _fail_sam3)
    monkeypatch.setattr(runner._vm, "ssh_exec", lambda *_args, **_kwargs: (0, "", ""))

    results = runner.compare_backends(image_path, out_dir)
    assert results["grounded_sam"]["status"] == "success"
    assert results["sam3"]["status"] == "failed"
    assert results["sam3"]["fallback_used"] is False
    assert "sam3 failed hard" in (results["sam3"]["error"] or "")
    assert results["success"] is False


def test_generate_and_upload_config_does_not_leak_gemini_key_in_ssh(
    monkeypatch,
    tmp_path: Path,
) -> None:
    config = Regen3DConfig(
        repo_path="/tmp/3D-RE-GEN",
        labels=["chair"],
        gemini_api_key="AIza_test_secret_1234567890",
        huggingface_token="hf_test_secret_abcdefghijklmnopqrstuvwxyz",
    )
    runner = Regen3DRunner(config=config, verbose=False)

    recorded_cmds: list[str] = []
    uploaded_paths: list[str] = []
    uploaded_env_content = {"text": ""}

    def _fake_scp_upload(local_path: Path, remote_path: str) -> None:
        uploaded_paths.append(remote_path)
        if remote_path.endswith("/src/config.yaml"):
            # Ensure config upload path still works.
            assert Path(local_path).is_file()
        if remote_path.endswith("/.env_keys"):
            uploaded_env_content["text"] = Path(local_path).read_text()

    def _fake_ssh_exec(command: str, *args, **kwargs):
        recorded_cmds.append(command)
        return (0, "", "")

    monkeypatch.setattr(runner._vm, "scp_upload", _fake_scp_upload)
    monkeypatch.setattr(runner._vm, "ssh_exec", _fake_ssh_exec)

    runner._generate_and_upload_config(
        remote_image_path="/tmp/3D-RE-GEN/input_images/scene.jpg",
        remote_output_dir="/tmp/3D-RE-GEN/output_scene",
        scene_id="scene",
    )

    assert any(path.endswith("/.env_keys") for path in uploaded_paths)
    assert all("GEMINI_API_KEY=" not in cmd for cmd in recorded_cmds)
    assert all("HF_TOKEN=" not in cmd for cmd in recorded_cmds)
    assert all("HF_HUB_TOKEN=" not in cmd for cmd in recorded_cmds)
    assert all("HUGGINGFACE_HUB_TOKEN=" not in cmd for cmd in recorded_cmds)
    assert "GEMINI_API_KEY=" in uploaded_env_content["text"]
    assert "HF_TOKEN=" in uploaded_env_content["text"]
    assert "HF_HUB_TOKEN=" in uploaded_env_content["text"]
    assert "HUGGINGFACE_HUB_TOKEN=" in uploaded_env_content["text"]


def test_step7_background_pointcloud_fallback_creates_empty_room_file(
    monkeypatch,
) -> None:
    runner = Regen3DRunner(config=Regen3DConfig(), verbose=False)
    commands: list[str] = []

    def _fake_ssh_exec(command: str, *args, **kwargs):
        commands.append(command)
        if "test -f" in command and "points_emptyRoom.ply" in command:
            return (1, "", "")
        if "test -f" in command and "points_merged.ply" in command:
            return (0, "EXISTS\n", "")
        if command.startswith("cp "):
            return (0, "", "")
        return (1, "", "")

    monkeypatch.setattr(runner._vm, "ssh_exec", _fake_ssh_exec)
    runner._ensure_step7_background_pointcloud("/remote/out")
    assert any(cmd.startswith("cp ") and "points_emptyRoom.ply" in cmd for cmd in commands)


def test_harvest_is_idempotent_and_removes_stale_objects(tmp_path: Path) -> None:
    native_one = tmp_path / "native_one"
    native_two = tmp_path / "native_two"
    target = tmp_path / "regen3d"

    _write_native_output(native_one, ["chair", "table"], include_transforms=True)
    _write_native_output(native_two, ["lamp"], include_transforms=True)

    harvest_regen3d_native_output(native_one, target, scene_id="scene")
    harvest_regen3d_native_output(native_two, target, scene_id="scene")

    object_dirs = sorted(p.name for p in (target / "objects").iterdir() if p.is_dir())
    assert object_dirs == ["obj_0"]


def test_harvest_uses_identity_when_object_transforms_are_missing(tmp_path: Path) -> None:
    native_dir = tmp_path / "native"
    target_dir = tmp_path / "regen3d"
    _write_native_output(native_dir, ["chair"], include_transforms=False)

    result = harvest_regen3d_native_output(native_dir, target_dir, scene_id="scene")
    assert result.objects_count == 1

    pose_path = target_dir / "objects" / "obj_0" / "pose.json"
    pose = json.loads(pose_path.read_text())
    assert pose["transform_matrix"] == [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]


def test_harvest_handles_missing_object_glbs(tmp_path: Path) -> None:
    native_dir = tmp_path / "native"
    target_dir = tmp_path / "regen3d"
    native_dir.mkdir(parents=True, exist_ok=True)

    result = harvest_regen3d_native_output(native_dir, target_dir, scene_id="scene")

    assert result.objects_count == 0
    assert "No object GLBs found in glb/ or 3D/ directories" in result.warnings


def test_harvest_finds_hunyuan_shape_glb_fallback(tmp_path: Path) -> None:
    native_dir = tmp_path / "native"
    target_dir = tmp_path / "regen3d"
    obj_name = "chair__(100, 200)"
    obj_dir = native_dir / "3D" / obj_name
    obj_dir.mkdir(parents=True, exist_ok=True)
    (obj_dir / f"{obj_name}_shape.glb").write_bytes(create_minimal_glb())

    scene_dir = native_dir / "glb" / "scene"
    scene_dir.mkdir(parents=True, exist_ok=True)
    (scene_dir / "combined_scene.glb").write_bytes(create_minimal_glb())
    transforms = {
        obj_name: [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
    }
    (scene_dir / "transforms.json").write_text(json.dumps(transforms, indent=2))

    result = harvest_regen3d_native_output(native_dir, target_dir, scene_id="scene")
    assert result.objects_count == 1
    assert (target_dir / "objects" / "obj_0" / "mesh.glb").is_file()


def test_vm_executor_download_dir_uses_find_with_symlink_support(monkeypatch, tmp_path: Path) -> None:
    vm = VMExecutor(
        VMConfig(host="example-host", zone="example-zone"),
        verbose=False,
    )
    captured_cmds: list[str] = []
    downloaded_paths: list[tuple[str, Path]] = []

    def _fake_ssh_exec(command: str, *args, **kwargs):
        captured_cmds.append(command)
        return (0, "/remote/out/3D/chair/chair.glb\n", "")

    def _fake_scp_download(remote_path: str, local_path: Path):
        local_path.parent.mkdir(parents=True, exist_ok=True)
        local_path.write_bytes(b"glb")
        downloaded_paths.append((remote_path, local_path))

    monkeypatch.setattr(vm, "ssh_exec", _fake_ssh_exec)
    monkeypatch.setattr(vm, "scp_download", _fake_scp_download)

    output = vm.scp_download_dir("/remote/out", tmp_path / "local")
    assert output
    assert any("-type l" in cmd and "find -L" in cmd for cmd in captured_cmds)
    assert downloaded_paths[0][0].endswith("chair.glb")
    assert downloaded_paths[0][1].is_file()


def test_run_reconstruction_fails_when_background_missing_and_required(
    monkeypatch,
    tmp_path: Path,
) -> None:
    from tools.regen3d_runner import runner as runner_module

    input_image = tmp_path / "input.jpg"
    input_image.write_bytes(b"mock")
    output_dir = tmp_path / "regen3d"

    runner = Regen3DRunner(
        config=Regen3DConfig(gemini_api_key="", labels=["chair"], require_background=True),
        verbose=False,
    )

    monkeypatch.setattr(runner, "_ensure_vm_ready", lambda: None)
    monkeypatch.setattr(runner, "_setup_remote", lambda: None)
    monkeypatch.setattr(runner, "_upload_input_image", lambda *_args, **_kwargs: "/remote/input.jpg")
    monkeypatch.setattr(runner, "_generate_and_upload_config", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(runner, "_execute_pipeline", lambda *_args, **_kwargs: (0, "", ""))
    monkeypatch.setattr(runner, "_count_remote_masks", lambda *_args, **_kwargs: 0)
    monkeypatch.setattr(runner, "_create_glb_symlinks", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(runner, "_download_outputs", lambda *_args, **_kwargs: None)

    def _fake_harvest(**kwargs):
        return HarvestResult(
            target_dir=kwargs["target_dir"],
            objects_count=1,
            has_background=False,
            has_camera=True,
            has_depth=False,
            warnings=["No background mesh found"],
        )

    monkeypatch.setattr(runner_module, "harvest_regen3d_native_output", _fake_harvest)

    result = runner.run_reconstruction(
        input_image=input_image,
        scene_id="scene",
        output_dir=output_dir,
    )

    assert not result.success
    assert "Background mesh not produced" in (result.error or "")


def test_run_reconstruction_sam3_mask_guard_retries_with_stricter_labels(
    monkeypatch,
    tmp_path: Path,
) -> None:
    from tools.regen3d_runner import runner as runner_module

    input_image = tmp_path / "input.jpg"
    input_image.write_bytes(b"mock")
    output_dir = tmp_path / "regen3d"

    cfg = Regen3DConfig(
        gemini_api_key="",
        labels=[
            "stacked washer and dryer",
            "control knob",
            "digital display panel",
            "woven basket",
            "wall",
            "floor",
            "detergent bottle",
        ],
        seg_backend="sam3",
        quality_mode="quality",
        max_masks=5,
    )
    runner = Regen3DRunner(config=cfg, verbose=False)

    monkeypatch.setattr(runner, "_ensure_vm_ready", lambda: None)
    monkeypatch.setattr(runner, "_setup_remote", lambda: None)
    monkeypatch.setattr(runner, "_upload_input_image", lambda *_args, **_kwargs: "/remote/input.jpg")
    monkeypatch.setattr(runner, "_generate_and_upload_config", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(runner, "_execute_pipeline", lambda *_args, **_kwargs: (0, "", ""))
    monkeypatch.setattr(runner, "_count_remote_masks", lambda *_args, **_kwargs: 0)
    monkeypatch.setattr(runner, "_create_glb_symlinks", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(runner, "_download_outputs", lambda *_args, **_kwargs: None)

    cleared = {"count": 0}
    monkeypatch.setattr(
        runner,
        "_clear_remote_segmentation_outputs",
        lambda *_args, **_kwargs: cleared.__setitem__("count", cleared["count"] + 1),
    )

    calls = {"count": 0}

    def _fake_sam3(*_args, **_kwargs):
        calls["count"] += 1
        if calls["count"] == 1:
            return {"mask_count": 12, "fallback_used": False, "error": None}
        return {"mask_count": 3, "fallback_used": False, "error": None}

    monkeypatch.setattr(runner, "_run_sam3_segmentation", _fake_sam3)

    def _fake_harvest(**kwargs):
        return HarvestResult(
            target_dir=kwargs["target_dir"],
            objects_count=2,
            has_background=True,
            has_camera=True,
            has_depth=False,
            warnings=[],
        )

    monkeypatch.setattr(runner_module, "harvest_regen3d_native_output", _fake_harvest)

    result = runner.run_reconstruction(
        input_image=input_image,
        scene_id="scene",
        output_dir=output_dir,
    )

    assert result.success
    assert calls["count"] == 2
    assert cleared["count"] == 1
    assert len(runner.config.labels) < 7


def test_run_reconstruction_sam3_mask_guard_fails_if_retry_still_too_high(
    monkeypatch,
    tmp_path: Path,
) -> None:
    input_image = tmp_path / "input.jpg"
    input_image.write_bytes(b"mock")
    output_dir = tmp_path / "regen3d"

    runner = Regen3DRunner(
        config=Regen3DConfig(
            gemini_api_key="",
            labels=["washer", "dryer", "basket", "floor", "wall", "knob"],
            seg_backend="sam3",
            quality_mode="quality",
            max_masks=4,
        ),
        verbose=False,
    )

    monkeypatch.setattr(runner, "_ensure_vm_ready", lambda: None)
    monkeypatch.setattr(runner, "_setup_remote", lambda: None)
    monkeypatch.setattr(runner, "_upload_input_image", lambda *_args, **_kwargs: "/remote/input.jpg")
    monkeypatch.setattr(runner, "_generate_and_upload_config", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(runner, "_clear_remote_segmentation_outputs", lambda *_args, **_kwargs: None)

    calls = {"count": 0}

    def _fake_sam3(*_args, **_kwargs):
        calls["count"] += 1
        if calls["count"] == 1:
            return {"mask_count": 10, "fallback_used": False, "error": None}
        return {"mask_count": 8, "fallback_used": False, "error": None}

    monkeypatch.setattr(runner, "_run_sam3_segmentation", _fake_sam3)

    result = runner.run_reconstruction(
        input_image=input_image,
        scene_id="scene",
        output_dir=output_dir,
    )

    assert not result.success
    assert "too many masks" in (result.error or "").lower()


def test_run_reconstruction_grounded_sam_mask_guard_fails_fast(
    monkeypatch,
    tmp_path: Path,
) -> None:
    input_image = tmp_path / "input.jpg"
    input_image.write_bytes(b"mock")
    output_dir = tmp_path / "regen3d"

    runner = Regen3DRunner(
        config=Regen3DConfig(
            gemini_api_key="",
            labels=["chair"],
            seg_backend="grounded_sam",
            quality_mode="quality",
            max_masks=2,
            steps=[1],
        ),
        verbose=False,
    )
    monkeypatch.setattr(runner, "_ensure_vm_ready", lambda: None)
    monkeypatch.setattr(runner, "_setup_remote", lambda: None)
    monkeypatch.setattr(runner, "_upload_input_image", lambda *_args, **_kwargs: "/remote/input.jpg")
    monkeypatch.setattr(runner, "_generate_and_upload_config", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(runner, "_execute_pipeline", lambda *_args, **_kwargs: (0, "", ""))
    monkeypatch.setattr(runner, "_count_remote_masks", lambda *_args, **_kwargs: 9)

    result = runner.run_reconstruction(
        input_image=input_image,
        scene_id="scene",
        output_dir=output_dir,
    )

    assert not result.success
    assert "too many masks" in (result.error or "").lower()


def test_vm_executor_retries_on_ssh_exit_255_when_check_false(monkeypatch) -> None:
    vm = VMExecutor(
        VMConfig(host="example-host", zone="example-zone", max_ssh_retries=3, retry_backoff_s=0.0),
        verbose=False,
    )

    attempts = {"count": 0}

    def _fake_exec_capture(_cmd, _timeout, _check):
        attempts["count"] += 1
        if attempts["count"] == 1:
            return 255, "", "temporary ssh error"
        return 0, "ok", ""

    monkeypatch.setattr(vm, "_build_ssh_cmd", lambda *_args, **_kwargs: ["ssh"])
    monkeypatch.setattr(vm, "_exec_capture", _fake_exec_capture)

    rc, stdout, stderr = vm.ssh_exec("echo ok", stream_logs=False, check=False)
    assert attempts["count"] == 2
    assert rc == 0
    assert stdout == "ok"
    assert stderr == ""


def test_vm_executor_retries_on_sshconnectionerror(monkeypatch) -> None:
    vm = VMExecutor(
        VMConfig(host="example-host", zone="example-zone", max_ssh_retries=3, retry_backoff_s=0.0),
        verbose=False,
    )

    attempts = {"count": 0}

    def _fake_exec_capture(_cmd, _timeout, _check):
        attempts["count"] += 1
        if attempts["count"] == 1:
            raise SSHConnectionError("transient ssh failure")
        return 0, "ok", ""

    monkeypatch.setattr(vm, "_build_ssh_cmd", lambda *_args, **_kwargs: ["ssh"])
    monkeypatch.setattr(vm, "_exec_capture", _fake_exec_capture)

    rc, stdout, stderr = vm.ssh_exec("echo ok", stream_logs=False, check=True)
    assert attempts["count"] == 2
    assert rc == 0
    assert stdout == "ok"
    assert stderr == ""


def test_vm_executor_sanitizes_gemini_key_in_command_log() -> None:
    raw = "echo GEMINI_API_KEY=AIzaRealSecretToken1234567890 && run"
    masked = VMExecutor._sanitize_command_for_log(raw)
    assert "GEMINI_API_KEY=<REDACTED>" in masked
    assert "AIzaRealSecretToken1234567890" not in masked


def test_vm_executor_sanitizes_hf_token_in_command_log() -> None:
    # Use a dummy token shape; never embed real HF tokens in the repo.
    raw = "export HF_TOKEN=hf_test_secret_abcdefghijklmnopqrstuvwxyz && run"
    masked = VMExecutor._sanitize_command_for_log(raw)
    assert "HF_TOKEN=<REDACTED>" in masked
    assert "hf_test_secret_abcdefghijklmnopqrstuvwxyz" not in masked


def test_regen3d_reconstruct_loads_env_defaults(monkeypatch, tmp_path: Path) -> None:
    import tools.regen3d_runner as regen3d_pkg
    import tools.run_local_pipeline as pipeline_module

    scene_dir = tmp_path / "scene"
    input_dir = scene_dir / "input"
    input_dir.mkdir(parents=True, exist_ok=True)
    (input_dir / "room.jpg").write_bytes(b"mock")

    dotenv_calls: list[tuple[Path, bool]] = []

    def _fake_load_dotenv(*args, **kwargs):
        path = kwargs.get("dotenv_path") if "dotenv_path" in kwargs else args[0]
        override = kwargs.get("override", False)
        dotenv_calls.append((Path(path), override))
        return True

    monkeypatch.setattr(pipeline_module, "load_dotenv", _fake_load_dotenv)

    fake_config = regen3d_pkg.Regen3DConfig(
        vm_host="vm-host",
        vm_zone="us-test1-a",
        repo_path="/home/test/3D-RE-GEN",
        steps=[1, 2, 3],
        gemini_api_key="",
    )
    monkeypatch.setattr(
        regen3d_pkg.Regen3DConfig,
        "from_env",
        classmethod(lambda cls: fake_config),
    )

    class _FakeRunner:
        def __init__(self, config, verbose):
            self.config = config
            self.verbose = verbose

        def run_reconstruction(self, input_image, scene_id, output_dir, environment_type):
            return SimpleNamespace(
                success=True,
                scene_id=scene_id,
                objects_count=2,
                duration_seconds=0.1,
                output_dir=output_dir,
                error=None,
            )

    monkeypatch.setattr(regen3d_pkg, "Regen3DRunner", _FakeRunner)

    runner = pipeline_module.LocalPipelineRunner(
        scene_dir=scene_dir,
        verbose=False,
        skip_interactive=True,
        environment_type="kitchen",
        enable_dwm=False,
        enable_dream2flow=False,
    )

    result = runner._run_regen3d_reconstruct()
    assert result.success
    assert dotenv_calls

    expected_env_path = Path(pipeline_module.__file__).resolve().parents[1] / "configs" / "regen3d_reconstruct.env"
    loaded_path, override = dotenv_calls[-1]
    assert loaded_path == expected_env_path
    assert override is False
