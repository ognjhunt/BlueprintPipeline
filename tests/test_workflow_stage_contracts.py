from __future__ import annotations

from pathlib import Path


def _assert_ordered_markers(text: str, markers: list[str]) -> None:
    positions = [text.index(marker) for marker in markers]
    assert positions == sorted(positions), f"Expected ordered markers: {markers}"


def test_usd_assembly_stage2_order_includes_interactive_phase():
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


def test_usd_assembly_enforces_strict_defaults_and_failure_paths():
    workflow = Path("workflows/usd-assembly-pipeline.yaml").read_text(encoding="utf-8")

    assert 'interactiveFailurePolicy: \'${default(map.get(event, "interactive_failure_policy"), default(sys.get_env("INTERACTIVE_FAILURE_POLICY"), "hybrid_strict"))}\'' in workflow
    assert 'requireReplicator: \'${default(map.get(event, "require_replicator"), default(sys.get_env("REQUIRE_REPLICATOR"), "true")) == "true"}\'' in workflow
    assert 'requireIsaac: \'${default(map.get(event, "require_isaac"), default(sys.get_env("REQUIRE_ISAAC"), "true")) == "true"}\'' in workflow
    assert "- replicator_failure_policy_switch:" in workflow
    assert "- release_lock_after_required_replicator_failure:" in workflow
    assert "- raise_required_replicator_error:" in workflow


def test_usd_assembly_hybrid_strict_gate_uses_required_articulation_and_markers():
    workflow = Path("workflows/usd-assembly-pipeline.yaml").read_text(encoding="utf-8")

    assert "requiredArticulationCount" in workflow
    assert "interactiveRequiredFailureCount" in workflow
    assert "interactiveMarkerMissing" in workflow
    assert (
        '${interactiveFailurePolicy == "hybrid_strict" and (interactiveMarkerMissing or (requiredArticulationCount > 0'
        in workflow
    )


def test_variation_pipeline_stage3_order_includes_isaac_refresh():
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


def test_variation_pipeline_refresh_env_and_markers_contract():
    workflow = Path("workflows/variation-assets-pipeline.yaml").read_text(encoding="utf-8")

    assert 'enableStage3IsaacRefresh: \'${default(map.get(event, "enable_stage3_isaac_refresh"), default(sys.get_env("ENABLE_STAGE3_ISAAC_REFRESH"), "true")) == "true"}\'' in workflow
    assert 'requireIsaac: \'${default(map.get(event, "require_isaac"), default(sys.get_env("REQUIRE_ISAAC"), "true")) == "true"}\'' in workflow
    assert "- name: ISAAC_REFRESH_ONLY" in workflow
    assert "- name: VARIATION_ASSETS_PREFIX" in workflow
    assert "- verify_isaac_refresh_marker:" in workflow
    assert "isaacLabRefreshCompleteObject" in workflow
    assert "- fail_missing_required_isaac:" in workflow


def test_orchestrator_passes_strict_defaults_into_stage2_and_stage3_events():
    workflow = Path("workflows/image-to-scene-orchestrator.yaml").read_text(
        encoding="utf-8"
    )

    assert "interactive_failure_policy: \"hybrid_strict\"" in workflow
    assert "require_replicator: true" in workflow
    assert "require_isaac: true" in workflow
    assert "enable_stage3_isaac_refresh: true" in workflow


def test_orchestrator_stage1_explicitly_runs_reconstruction_steps_only():
    workflow = Path("workflows/image-to-scene-orchestrator.yaml").read_text(
        encoding="utf-8"
    )

    assert (
        "run_pipeline_gcs.sh ${SCENE_ID_Q} ${BUCKET_Q} ${OBJECT_NAME_Q} "
        "${OBJECT_GENERATION_Q} regen3d-reconstruct,regen3d"
    ) in workflow


def test_source_orchestrator_text_path_wires_text_jobs_before_downstream_stages():
    workflow = Path("workflows/source-orchestrator.yaml").read_text(encoding="utf-8")

    _assert_ordered_markers(
        workflow,
        [
            "- run_text_gen_job:",
            "- run_text_adapter_job:",
            "- run_downstream_for_child:",
            "- run_stage2:",
            "- run_stage3:",
            "- run_stage4:",
        ],
    )


def test_source_orchestrator_preserves_strict_stage2_stage3_contracts():
    workflow = Path("workflows/source-orchestrator.yaml").read_text(encoding="utf-8")

    assert 'interactive_failure_policy: "hybrid_strict"' in workflow
    assert "require_replicator: true" in workflow
    assert "require_isaac: true" in workflow
    assert "enable_stage3_isaac_refresh: true" in workflow


def test_source_orchestrator_trigger_filter_targets_scene_request_json():
    workflow = Path("workflows/source-orchestrator.yaml").read_text(encoding="utf-8")

    assert 'text.match_regex(object, "^scenes/[^/]+/prompts/scene_request\\\\.json$")' in workflow
