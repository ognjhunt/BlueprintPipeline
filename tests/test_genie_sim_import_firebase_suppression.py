from __future__ import annotations

def test_firebase_upload_suppression_when_partial_uploads_disabled(
    load_job_module,
) -> None:
    module = load_job_module("geniesim_import", "import_from_geniesim.py")

    suppressed, reason, log_partial = module._resolve_firebase_upload_suppression(
        partial_failure=True,
        allow_partial_firebase_uploads=False,
        fail_on_partial_error=True,
        quality_gate_failures=[],
    )

    assert suppressed is True
    assert reason == "partial_failure_fail_on_partial_error"
    assert log_partial is False


def test_firebase_upload_suppression_when_partial_uploads_allowed(
    load_job_module,
) -> None:
    module = load_job_module("geniesim_import", "import_from_geniesim.py")

    suppressed, reason, log_partial = module._resolve_firebase_upload_suppression(
        partial_failure=True,
        allow_partial_firebase_uploads=True,
        fail_on_partial_error=True,
        quality_gate_failures=[],
    )

    assert suppressed is False
    assert reason is None
    assert log_partial is True
