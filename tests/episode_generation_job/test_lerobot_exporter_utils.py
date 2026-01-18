import pytest


@pytest.mark.unit
def test_lerobot_exporter_sanitizes_paths(load_job_module, tmp_path):
    lerobot_exporter = load_job_module("episode_generation", "lerobot_exporter.py")
    config = lerobot_exporter.LeRobotDatasetConfig(dataset_name="demo")
    exporter = lerobot_exporter.LeRobotExporter(config, verbose=False)

    message = f"Failed to open {tmp_path}/secret.txt"
    sanitized = exporter._sanitize_error_message(message)

    assert "<redacted-path>" in sanitized
    assert str(tmp_path) not in sanitized


@pytest.mark.unit
def test_checksum_manifest_tracks_relative_paths(load_job_module, tmp_path):
    lerobot_exporter = load_job_module("episode_generation", "lerobot_exporter.py")
    config = lerobot_exporter.LeRobotDatasetConfig(dataset_name="demo")
    exporter = lerobot_exporter.LeRobotExporter(config, verbose=False)

    exporter.output_dir = tmp_path
    file_path = tmp_path / "meta" / "info.json"
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text("data")

    exporter._record_checksum(file_path)

    assert "meta/info.json" in exporter.checksum_manifest
    assert exporter._checksum_manifest_hash() is not None
