import json

import pytest

import tools.error_handling.partial_failure as partial_failure


class TestPartialFailure:
    def test_process_with_partial_failure_surfaces_failures(self, tmp_path):
        items = ["a", "b", "c"]

        def process_fn(item):
            if item == "b":
                raise RuntimeError("bad item")
            return item.upper()

        progress_file = tmp_path / "progress.json"

        with pytest.raises(partial_failure.PartialFailureError) as exc_info:
            partial_failure.process_with_partial_failure(
                items,
                process_fn,
                min_success_rate=1.0,
                save_progress=True,
                progress_file=progress_file,
                item_id_fn=lambda value: f"item-{value}",
            )

        result = exc_info.value.result
        assert result.successful == ["A", "C"]
        assert result.failure_count == 1
        assert result.failed[0]["item_id"] == "item-b"
        assert result.failed[0]["error_type"] == "RuntimeError"
        assert progress_file.exists()

    def test_handler_writes_failure_report(self, tmp_path):
        output_dir = tmp_path / "outputs"
        report_path = tmp_path / "reports" / "failures.json"
        handler = partial_failure.PartialFailureHandler(
            min_success_rate=0.0,
            save_successful=True,
            output_dir=output_dir,
            failure_report_path=report_path,
        )

        def process_fn(item):
            if item == 2:
                raise ValueError("nope")
            return item

        result = handler.process_batch([1, 2], process_fn, item_id_fn=str, batch_name="batch")

        assert result.success_count == 1
        assert result.failure_count == 1
        assert report_path.exists()

        report = json.loads(report_path.read_text())
        assert report["total_failures"] == 1
        assert report["failures"][0]["error_type"] == "ValueError"
