import sys

import pytest


def test_deliver_flag_injects_dataset_delivery(monkeypatch, tmp_path):
    from tools import run_local_pipeline

    captured = {}

    def fake_run(self, steps=None, **kwargs):
        captured["steps"] = steps
        return True

    monkeypatch.setattr(run_local_pipeline.LocalPipelineRunner, "run", fake_run)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_local_pipeline.py",
            "--scene-dir",
            str(tmp_path),
            "--steps",
            "genie-sim-import",
            "--deliver",
        ],
    )

    with pytest.raises(SystemExit) as excinfo:
        run_local_pipeline.main()

    assert excinfo.value.code == 0
    assert captured["steps"] == [
        run_local_pipeline.PipelineStep.GENIESIM_IMPORT,
        run_local_pipeline.PipelineStep.DATASET_DELIVERY,
    ]
