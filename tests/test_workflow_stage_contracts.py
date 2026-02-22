from __future__ import annotations

from pathlib import Path


def _assert_ordered_markers(text: str, markers: list[str]) -> None:
    positions = [text.index(marker) for marker in markers]
    assert positions == sorted(positions), f"Expected ordered markers: {markers}"


def test_usd_assembly_stage2_order_includes_interactive_phase() -> None:
    workflow = Path("workflows/usd-assembly-pipeline.yaml").read_text(encoding="utf-8")

    _assert_ordered_markers(
        workflow,
        [
            "- run_convert_job:",
            "- run_simready_job:",
            "- run_interactive_job:",
            "- run_usd_job:",
            "- run_replicator_job:",
            "- run_isaac_lab_job:",
        ],
    )


def test_variation_pipeline_stage3_order_includes_isaac_refresh() -> None:
    workflow = Path("workflows/variation-assets-pipeline.yaml").read_text(encoding="utf-8")

    _assert_ordered_markers(
        workflow,
        [
            "- run_variation_gen_job:",
            "- run_simready_job:",
            "- run_isaac_refresh_job:",
            "- ensure_at_least_one_isaac_pass:",
        ],
    )


def test_source_orchestrator_text_path_wires_text_jobs_before_downstream_stages() -> None:
    workflow = Path("workflows/source-orchestrator.yaml").read_text(encoding="utf-8")

    _assert_ordered_markers(
        workflow,
        [
            "- run_text_stage1_for_child:",
            "- run_downstream_for_child:",
            "- run_stage2:",
            "- run_stage3:",
            "- run_stage4:",
        ],
    )


def test_source_orchestrator_has_text_only_request_contract() -> None:
    workflow = Path("workflows/source-orchestrator.yaml").read_text(encoding="utf-8")

    assert 'requestSourceMode != "text"' in workflow
    assert 'requestTextBackend != "scenesmith" and requestTextBackend != "sage" and requestTextBackend != "hybrid_serial"' in workflow
    assert 'text.replace_all(default(sys.get_env("TEXT_BACKEND_ALLOWLIST"), "scenesmith,sage,hybrid_serial"), " ", "")' in workflow


def test_source_orchestrator_strips_image_path_compat() -> None:
    workflow = Path("workflows/source-orchestrator.yaml").read_text(encoding="utf-8")

    assert "run_image_path_compat:" not in workflow
    assert "imagePathMode" not in workflow
    assert "wait_for_gcs_marker:" not in workflow
    assert "wait_for_any_gcs_marker:" not in workflow


def test_source_orchestrator_stage2_params_exclude_image_mode() -> None:
    workflow = Path("workflows/source-orchestrator.yaml").read_text(encoding="utf-8")

    assert "params: [projectId, region, bucket, sceneId, arenaExportRequired, useGeniesimOverrideRaw]" in workflow


def test_source_orchestrator_stage2_trigger_uses_stage1_marker() -> None:
    workflow = Path("workflows/source-orchestrator.yaml").read_text(encoding="utf-8")

    assert 'name: \'${scenePrefix + "/assets/.stage1_complete"}\'' in workflow


def test_setup_source_orchestrator_trigger_has_new_backend_defaults() -> None:
    script = Path("workflows/setup-source-orchestrator-trigger.sh").read_text(encoding="utf-8")

    assert 'TEXT_BACKEND_ALLOWLIST=${TEXT_BACKEND_ALLOWLIST:-"scenesmith,sage,hybrid_serial"}' in script
    assert "IMAGE_PATH_MODE" not in script
    assert "TEXT_GEN_ENABLE_IMAGE_FALLBACK" not in script


def test_setup_all_triggers_no_longer_invokes_image_setup() -> None:
    script = Path("workflows/setup-all-triggers.sh").read_text(encoding="utf-8")

    assert "setup-image-trigger.sh" not in script


def test_setup_usd_assembly_trigger_uses_stage1_marker_regex() -> None:
    script = Path("workflows/setup-usd-assembly-trigger.sh").read_text(encoding="utf-8")

    assert "assets/.stage1_complete" in script
