import json

from tools.batch_processing.parallel_runner import SceneResult, SceneStatus
from tools.scene_batch_reporting import _summarize_batch_results


def test_summarize_batch_results_writes_dlq(tmp_path):
    reports_dir = tmp_path / "reports"
    dlq_path = tmp_path / "custom_dlq.json"
    report_path = reports_dir / "scene-1" / "quality_gate_report.json"
    report_path.parent.mkdir(parents=True)
    report_path.write_text(json.dumps({
        "results": [
            {
                "gate_id": "gate-alpha",
                "checkpoint": "scene_ready",
                "severity": "error",
                "message": "Missing output",
                "passed": False,
                "recommendations": ["rerun pipeline"],
            },
        ],
    }))

    results = [
        SceneResult(
            scene_id="scene-1",
            status=SceneStatus.FAILED,
            error="boom",
            metadata={
                "scene_dir": "/scenes/scene-1",
                "quality_gate_report": str(report_path),
                "attempt": 2,
            },
        ),
        SceneResult(
            scene_id="scene-2",
            status=SceneStatus.CANCELLED,
            error="cancelled",
            metadata={
                "scene_dir": "/scenes/scene-2",
                "attempt": 1,
            },
        ),
        SceneResult(
            scene_id="scene-3",
            status=SceneStatus.SUCCESS,
            metadata={"scene_dir": "/scenes/scene-3"},
        ),
    ]

    _summarize_batch_results(results, reports_dir, dlq_path=dlq_path)

    dlq_payload = json.loads(dlq_path.read_text())
    assert {entry["scene_id"] for entry in dlq_payload} == {"scene-1", "scene-2"}

    by_scene = {entry["scene_id"]: entry for entry in dlq_payload}
    scene_one = by_scene["scene-1"]
    assert scene_one["scene_dir"] == "/scenes/scene-1"
    assert scene_one["status"] == "failed"
    assert scene_one["error"] == "boom"
    assert scene_one["attempts"] == 2
    assert scene_one["quality_gate_failures"] == [
        {
            "gate_id": "gate-alpha",
            "checkpoint": "scene_ready",
            "severity": "error",
            "message": "Missing output",
            "recommendations": ["rerun pipeline"],
        },
    ]

    scene_two = by_scene["scene-2"]
    assert scene_two["scene_dir"] == "/scenes/scene-2"
    assert scene_two["status"] == "cancelled"
    assert scene_two["error"] == "cancelled"
    assert scene_two["attempts"] == 1
    assert scene_two["quality_gate_failures"] == []
