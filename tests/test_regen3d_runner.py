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

    runner = Regen3DRunner(config=Regen3DConfig(gemini_api_key=""), verbose=False)

    monkeypatch.setattr(runner, "_ensure_vm_ready", lambda: None)
    monkeypatch.setattr(runner, "_setup_remote", lambda: None)
    monkeypatch.setattr(runner, "_upload_input_image", lambda *_args, **_kwargs: "/remote/input.jpg")
    monkeypatch.setattr(runner, "_generate_and_upload_config", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(runner, "_execute_pipeline", lambda *_args, **_kwargs: (0, "", ""))

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


def test_harvest_fails_when_object_transforms_are_missing(tmp_path: Path) -> None:
    native_dir = tmp_path / "native"
    target_dir = tmp_path / "regen3d"
    _write_native_output(native_dir, ["chair"], include_transforms=False)

    with pytest.raises(ValueError, match="chair"):
        harvest_regen3d_native_output(native_dir, target_dir, scene_id="scene")


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
