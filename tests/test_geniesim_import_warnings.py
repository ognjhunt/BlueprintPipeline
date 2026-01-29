from __future__ import annotations

import logging
from pathlib import Path

import pytest


def test_resolve_upload_file_list_warns_on_bad_json(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    load_job_module,
) -> None:
    module = load_job_module("geniesim_import", "import_from_geniesim.py")

    output_dir = tmp_path / "output"
    dataset_info_path = output_dir / "lerobot" / "dataset_info.json"
    dataset_info_path.parent.mkdir(parents=True)
    dataset_info_path.write_text("{not-json")

    caplog.set_level(logging.WARNING)
    with pytest.raises(ValueError, match="Failed to load JSON file"):
        module._resolve_upload_file_list(output_dir, ["episode_000001"])


def test_parquet_validation_logs_shape_warnings_once(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    load_job_module,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    pd = pytest.importorskip("pandas")

    module = load_job_module("geniesim_import", "import_from_geniesim.py")

    parquet_path = tmp_path / "episode_000000.parquet"
    df = pd.DataFrame(
        {
            "observation": ["bad_obs"],
            "action": ["bad_action"],
        }
    )
    table = pa.Table.from_pandas(df)
    pq.write_table(table, parquet_path)

    original_array = module.np.array

    def guarded_array(value, *args, **kwargs):
        if value in {"bad_obs", "bad_action"}:
            raise ValueError("invalid shape payload")
        return original_array(value, *args, **kwargs)

    monkeypatch.setattr(module.np, "array", guarded_array)

    caplog.set_level(logging.WARNING)
    results = module._stream_parquet_validation(
        parquet_path,
        require_parquet_validation=True,
        episode_index=0,
    )

    assert "errors" in results
    warning_messages = [record.message for record in caplog.records]
    assert any("observation" in message and "episode_index=0" in message for message in warning_messages)
    assert any("action" in message and "episode_index=0" in message for message in warning_messages)
