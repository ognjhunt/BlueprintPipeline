import importlib.util
import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest
from tools.lerobot_format import LeRobotExportFormat


class _FakeArrowSchema:
    def __init__(self, names: list[str]) -> None:
        self.names = names


class _FakeArrowTable:
    def __init__(self, data: dict, schema: _FakeArrowSchema | None = None) -> None:
        self._data = data
        self.schema = schema or _FakeArrowSchema(list(data.keys()))
        values = list(data.values())
        self.num_rows = len(values[0]) if values else 0


class _FakeParquetRegistry:
    def __init__(self) -> None:
        self.files: dict[str, dict[str, object]] = {}


class _FakeParquetWriter:
    def __init__(self, path: Path, schema: _FakeArrowSchema, compression: str | None = None) -> None:
        self.path = Path(path)
        self.schema = schema
        self.num_rows = 0
        self.num_row_groups = 0

    def write_table(self, table: _FakeArrowTable) -> None:
        self.num_rows += table.num_rows
        self.num_row_groups += 1
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_bytes(b"FAKEPARQUET")

    def close(self) -> None:
        _FAKE_PARQUET_REGISTRY.files[str(self.path)] = {
            "num_rows": self.num_rows,
            "num_row_groups": self.num_row_groups,
            "schema_names": self.schema.names,
        }


class _FakeParquetFile:
    def __init__(self, path: Path) -> None:
        record = _FAKE_PARQUET_REGISTRY.files.get(str(path))
        if record is None:
            raise FileNotFoundError(f"Fake parquet metadata missing for {path}")
        self.metadata = types.SimpleNamespace(
            num_rows=record["num_rows"],
            num_row_groups=record["num_row_groups"],
        )
        self.schema_arrow = types.SimpleNamespace(names=record["schema_names"])


_FAKE_PARQUET_REGISTRY = _FakeParquetRegistry()


def _install_fake_pyarrow(monkeypatch: pytest.MonkeyPatch) -> bool:
    if importlib.util.find_spec("pyarrow") is not None:
        return False
    fake_pa = types.ModuleType("pyarrow")
    fake_pa.Codec = type("Codec", (), {"is_available": staticmethod(lambda name: True)})
    fake_pa.int64 = lambda: "int64"
    fake_pa.float64 = lambda: "float64"
    fake_pa.float32 = lambda: "float32"
    fake_pa.string = lambda: "string"
    fake_pa.list_ = lambda _inner: "list"
    fake_pa.array = lambda values, type=None: list(values)
    fake_pa.schema = lambda fields: _FakeArrowSchema([name for name, _ in fields])
    fake_pa.table = lambda data, schema=None: _FakeArrowTable(data, schema=schema)

    fake_pq = types.ModuleType("pyarrow.parquet")
    fake_pq.ParquetWriter = _FakeParquetWriter
    fake_pq.ParquetFile = _FakeParquetFile

    def _write_table(table: _FakeArrowTable, path: Path, compression: str | None = None) -> None:
        writer = _FakeParquetWriter(path, schema=table.schema, compression=compression)
        writer.write_table(table)
        writer.close()

    fake_pq.write_table = _write_table
    monkeypatch.setitem(sys.modules, "pyarrow", fake_pa)
    monkeypatch.setitem(sys.modules, "pyarrow.parquet", fake_pq)
    return True


@pytest.mark.unit
def test_lerobot_exporter_writes_metadata_and_data(load_job_module, tmp_path: Path) -> None:
    trajectory_solver = load_job_module("episode_generation", "trajectory_solver.py")
    lerobot_exporter = load_job_module("episode_generation", "lerobot_exporter.py")

    robot_config = trajectory_solver.ROBOT_CONFIGS["franka"]
    states = [
        trajectory_solver.JointState(
            frame_idx=0,
            timestamp=0.0,
            joint_positions=robot_config.default_joint_positions.copy(),
            joint_velocities=np.zeros(robot_config.num_joints),
            gripper_position=0.0,
            ee_position=np.array([0.4, 0.0, 0.7]),
            phase=trajectory_solver.MotionPhase.APPROACH,
        ),
        trajectory_solver.JointState(
            frame_idx=1,
            timestamp=1.0 / 30.0,
            joint_positions=robot_config.default_joint_positions.copy() + 0.01,
            joint_velocities=np.zeros(robot_config.num_joints),
            gripper_position=0.02,
            ee_position=np.array([0.45, 0.05, 0.72]),
            phase=trajectory_solver.MotionPhase.PLACE,
        ),
    ]

    trajectory = trajectory_solver.JointTrajectory(
        trajectory_id="traj_test",
        robot_type="franka",
        robot_config=robot_config,
        states=states,
        source_plan_id="plan_001",
        fps=30.0,
        total_duration=1.0 / 30.0,
    )

    config = lerobot_exporter.LeRobotDatasetConfig(
        dataset_name="unit_test_dataset",
        robot_type="franka",
        output_dir=tmp_path / "lerobot_dataset",
    )
    exporter = lerobot_exporter.LeRobotExporter(config, verbose=False)
    exporter.add_episode(trajectory, "Pick and place test")

    output_dir = exporter.finalize()

    meta_dir = output_dir / "meta"
    data_dir = output_dir / "data"
    assert meta_dir.exists()
    assert data_dir.exists()
    assert (meta_dir / "info.json").exists()
    assert (meta_dir / "tasks.jsonl").exists()
    assert (meta_dir / "episodes.jsonl").exists()

    info = json.loads((meta_dir / "info.json").read_text())
    assert info["total_episodes"] == 1
    assert info["total_tasks"] == 1
    assert info["data_pack"]["tier"] == "core"

    chunk_dir = data_dir / "chunk-000"
    assert chunk_dir.exists()

    if lerobot_exporter.HAVE_PYARROW:
        assert (chunk_dir / "episode_000000.parquet").exists()
    else:
        assert (chunk_dir / "episode_000000.json").exists()


@pytest.mark.unit
def test_lerobot_exporter_requires_complete_episodes(load_job_module, tmp_path: Path) -> None:
    trajectory_solver = load_job_module("episode_generation", "trajectory_solver.py")
    lerobot_exporter = load_job_module("episode_generation", "lerobot_exporter.py")

    robot_config = trajectory_solver.ROBOT_CONFIGS["franka"]
    states = [
        trajectory_solver.JointState(
            frame_idx=0,
            timestamp=0.0,
            joint_positions=robot_config.default_joint_positions.copy(),
            joint_velocities=np.zeros(robot_config.num_joints),
            gripper_position=0.0,
            ee_position=np.array([0.4, 0.0, 0.7]),
            phase=trajectory_solver.MotionPhase.APPROACH,
        ),
        trajectory_solver.JointState(
            frame_idx=1,
            timestamp=1.0 / 30.0,
            joint_positions=robot_config.default_joint_positions.copy() + 0.01,
            joint_velocities=np.zeros(robot_config.num_joints),
            gripper_position=0.02,
            ee_position=np.array([0.45, 0.05, 0.72]),
            phase=trajectory_solver.MotionPhase.PLACE,
        ),
    ]

    trajectory = trajectory_solver.JointTrajectory(
        trajectory_id="traj_test",
        robot_type="franka",
        robot_config=robot_config,
        states=states,
        source_plan_id="plan_001",
        fps=30.0,
        total_duration=1.0 / 30.0,
    )

    config = lerobot_exporter.LeRobotDatasetConfig(
        dataset_name="unit_test_dataset",
        robot_type="franka",
        output_dir=tmp_path / "lerobot_dataset",
        require_complete_episodes=True,
    )
    exporter = lerobot_exporter.LeRobotExporter(config, verbose=False)
    exporter.add_episode(trajectory, "Pick and place test")
    exporter.add_episode(trajectory, "")

    with pytest.raises(ValueError, match="Incomplete episodes detected"):
        exporter.finalize()


@pytest.mark.unit
def test_lerobot_exporter_writes_v3_layout(load_job_module, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    trajectory_solver = load_job_module("episode_generation", "trajectory_solver.py")
    _install_fake_pyarrow(monkeypatch)
    lerobot_exporter = load_job_module("episode_generation", "lerobot_exporter.py")

    robot_config = trajectory_solver.ROBOT_CONFIGS["franka"]
    states = [
        trajectory_solver.JointState(
            frame_idx=0,
            timestamp=0.0,
            joint_positions=robot_config.default_joint_positions.copy(),
            joint_velocities=np.zeros(robot_config.num_joints),
            gripper_position=0.0,
            ee_position=np.array([0.4, 0.0, 0.7]),
            phase=trajectory_solver.MotionPhase.APPROACH,
        ),
        trajectory_solver.JointState(
            frame_idx=1,
            timestamp=1.0 / 30.0,
            joint_positions=robot_config.default_joint_positions.copy() + 0.01,
            joint_velocities=np.zeros(robot_config.num_joints),
            gripper_position=0.02,
            ee_position=np.array([0.45, 0.05, 0.72]),
            phase=trajectory_solver.MotionPhase.PLACE,
        ),
    ]

    trajectory = trajectory_solver.JointTrajectory(
        trajectory_id="traj_test",
        robot_type="franka",
        robot_config=robot_config,
        states=states,
        source_plan_id="plan_001",
        fps=30.0,
        total_duration=1.0 / 30.0,
    )

    config = lerobot_exporter.LeRobotDatasetConfig(
        dataset_name="unit_test_dataset",
        robot_type="franka",
        output_dir=tmp_path / "lerobot_dataset",
        export_format=LeRobotExportFormat.LEROBOT_V3,
    )
    exporter = lerobot_exporter.LeRobotExporter(config, verbose=False)
    exporter.add_episode(trajectory, "Pick and place test")
    exporter.add_episode(trajectory, "Pick and place test 2")

    output_dir = exporter.finalize()
    meta_dir = output_dir / "meta"
    data_dir = output_dir / "data" / "chunk-000"

    info = json.loads((meta_dir / "info.json").read_text())
    assert info["export_format"] == LeRobotExportFormat.LEROBOT_V3.value
    assert info["version"] == "3.0"
    # v3 uses file-based naming per official LeRobot v3.0 spec
    assert (data_dir / "file-0000.parquet").exists()

    # Episode metadata stored in meta/episodes/
    episodes_meta_dir = meta_dir / "episodes" / "chunk-000"
    assert episodes_meta_dir.exists()
    assert (episodes_meta_dir / "file-0000.parquet").exists()


@pytest.mark.optional_dep
def test_lerobot_exporter_writes_v3_parquet_metadata(load_job_module, tmp_path: Path) -> None:
    pytest.importorskip("pyarrow")
    pytest.importorskip("pyarrow.parquet")
    trajectory_solver = load_job_module("episode_generation", "trajectory_solver.py")
    lerobot_exporter = load_job_module("episode_generation", "lerobot_exporter.py")

    robot_config = trajectory_solver.ROBOT_CONFIGS["franka"]
    states = [
        trajectory_solver.JointState(
            frame_idx=0,
            timestamp=0.0,
            joint_positions=robot_config.default_joint_positions.copy(),
            joint_velocities=np.zeros(robot_config.num_joints),
            gripper_position=0.0,
            ee_position=np.array([0.4, 0.0, 0.7]),
            phase=trajectory_solver.MotionPhase.APPROACH,
        ),
        trajectory_solver.JointState(
            frame_idx=1,
            timestamp=1.0 / 30.0,
            joint_positions=robot_config.default_joint_positions.copy() + 0.01,
            joint_velocities=np.zeros(robot_config.num_joints),
            gripper_position=0.02,
            ee_position=np.array([0.45, 0.05, 0.72]),
            phase=trajectory_solver.MotionPhase.PLACE,
        ),
    ]

    trajectory = trajectory_solver.JointTrajectory(
        trajectory_id="traj_test",
        robot_type="franka",
        robot_config=robot_config,
        states=states,
        source_plan_id="plan_001",
        fps=30.0,
        total_duration=1.0 / 30.0,
    )

    config = lerobot_exporter.LeRobotDatasetConfig(
        dataset_name="unit_test_dataset",
        robot_type="franka",
        output_dir=tmp_path / "lerobot_dataset",
        export_format=LeRobotExportFormat.LEROBOT_V3,
    )
    exporter = lerobot_exporter.LeRobotExporter(config, verbose=False)
    exporter.add_episode(trajectory, "Pick and place test")
    exporter.add_episode(trajectory, "Pick and place test 2")

    output_dir = exporter.finalize()
    meta_dir = output_dir / "meta" / "episodes" / "chunk-000"

    import pyarrow.parquet as pq

    episodes_meta = pq.read_table(meta_dir / "file-0000.parquet")
    assert "episode_index" in episodes_meta.column_names
    assert "num_frames" in episodes_meta.column_names
    assert episodes_meta.num_rows == 2
