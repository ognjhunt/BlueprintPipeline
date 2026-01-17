import json

import tools.error_handling.dead_letter as dead_letter


def test_local_dead_letter_queue_parse_failure_logs_and_skips(tmp_path, caplog):
    queue = dead_letter.LocalDeadLetterQueue(directory=str(tmp_path))
    bad_path = tmp_path / "pending" / "bad.json"
    bad_path.write_text("not-json")

    caplog.set_level("ERROR")
    messages = queue.get_pending()

    assert messages == []
    assert any("Failed to load" in record.message for record in caplog.records)


def test_local_dead_letter_queue_update_status_missing_returns_false(tmp_path):
    queue = dead_letter.LocalDeadLetterQueue(directory=str(tmp_path))

    assert queue.update_status("missing", "resolved") is False


def test_dead_letter_message_round_trip():
    message = dead_letter.DeadLetterMessage(
        scene_id="scene-1",
        job_type="job",
        step="step",
        error_type="RuntimeError",
        error_message="boom",
        metadata={"one": "two"},
    )

    data = json.loads(message.to_json())
    rehydrated = dead_letter.DeadLetterMessage.from_dict(data)

    assert rehydrated.scene_id == "scene-1"
    assert rehydrated.metadata["one"] == "two"
