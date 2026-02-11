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
            "- run_text_stage1_for_child:",
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


def test_source_orchestrator_fallback_path_delegates_to_image_orchestrator():
    """Verify that the text path exception handler can fall back to image compatibility path."""
    workflow = Path("workflows/source-orchestrator.yaml").read_text(encoding="utf-8")

    # The fallback step must exist and call run_image_path_compat
    assert "- fallback_to_image_mode:" in workflow
    assert "call: run_image_path_compat" in workflow

    # The fallback must be inside the text seed error handler
    assert "- maybe_fallback_switch:" in workflow

    # The image compatibility sub-workflow must be defined
    assert "run_image_path_compat:" in workflow
    assert "image-to-scene-orchestrator" in workflow
    assert "image-to-scene-pipeline" in workflow


def test_source_orchestrator_multi_seed_writes_variants_index():
    """Verify that multi-seed fanout writes a variants index."""
    workflow = Path("workflows/source-orchestrator.yaml").read_text(encoding="utf-8")

    # Variants index logic: only write when seedCount > 1
    assert "- write_variants_index:" in workflow
    assert "- write_variants_index_object:" in workflow
    assert "variantsIndexJson" in workflow
    assert "variantsEntriesJson" in workflow
    assert '${seedCount > 1}' in workflow

    # Verify variant entry accumulation in the seed loop
    assert "- append_variant_entry:" in workflow
    assert "variantEntryJson" in workflow


def test_source_orchestrator_multi_seed_child_request_uses_non_trigger_path():
    workflow = Path("workflows/source-orchestrator.yaml").read_text(encoding="utf-8")

    assert 'childRequestObject: \'${if(seedCount > 1, childScenePrefix + "/internal/scene_request.generated.json", object)}\'' in workflow
    assert "/internal/scene_request.generated.json" in workflow
    assert "- maybe_write_child_request:" in workflow
    assert "shouldWriteChildRequest" in workflow


def test_source_orchestrator_runtime_branches_include_vm_and_cloudrun():
    workflow = Path("workflows/source-orchestrator.yaml").read_text(encoding="utf-8")

    assert "run_text_stage1_for_child:" in workflow
    assert '${textGenRuntime == "vm"}' in workflow
    assert '${textGenRuntime == "cloudrun"}' in workflow
    assert "run_text_jobs_on_vm:" in workflow
    assert "run_text_jobs_on_cloudrun:" in workflow
    assert "[vm_transient]" in workflow
    assert "TEXT_ASSET_RETRIEVAL_ENABLED" in workflow
    assert "TEXT_ASSET_LIBRARY_PREFIXES" in workflow
    assert "TEXT_ASSET_GENERATION_ENABLED" in workflow
    assert "TEXT_ASSET_GENERATION_PROVIDER" in workflow
    assert "TEXT_ASSET_GENERATION_PROVIDER_CHAIN" in workflow
    assert "TEXT_ASSET_RETRIEVAL_MODE" in workflow
    assert "TEXT_ASSET_ANN_ENABLED" in workflow
    assert "TEXT_ASSET_ANN_TOP_K" in workflow
    assert "TEXT_ASSET_ANN_MIN_SCORE" in workflow
    assert "TEXT_ASSET_ANN_MAX_RERANK" in workflow
    assert "TEXT_ASSET_ANN_NAMESPACE" in workflow
    assert "TEXT_ASSET_LEXICAL_FALLBACK_ENABLED" in workflow
    assert "TEXT_ASSET_ROLLOUT_STATE_PREFIX" in workflow
    assert "TEXT_ASSET_EMBEDDING_QUEUE_PREFIX" in workflow
    assert "VECTOR_STORE_PROVIDER" in workflow
    assert "VERTEX_INDEX_ENDPOINT" in workflow
    assert "TEXT_SAM3D_TEXT_ENDPOINTS" in workflow
    assert "TEXT_HUNYUAN_TEXT_ENDPOINTS" in workflow


def test_source_orchestrator_image_mode_branches_include_orchestrator_and_legacy_chain():
    workflow = Path("workflows/source-orchestrator.yaml").read_text(encoding="utf-8")

    assert "run_image_path_compat:" in workflow
    assert '${imagePathMode == "orchestrator"}' in workflow
    assert '${imagePathMode == "legacy_chain"}' in workflow
    assert "wait_for_gcs_marker:" in workflow
    assert "IMAGE_PATH_MODE" in workflow


def test_source_orchestrator_request_sanity_checks_and_lock_dedupe_present():
    workflow = Path("workflows/source-orchestrator.yaml").read_text(encoding="utf-8")

    assert "requestSchemaVersion" in workflow
    assert "requestSceneId" in workflow
    assert "seedCount" in workflow
    assert "qualityTier" in workflow
    assert "- acquire_lock:" in workflow
    assert "ifGenerationMatch: 0" in workflow
    assert "source-orchestrator-" in workflow
    assert "release_source_lock_if_needed:" in workflow


def test_source_orchestrator_image_mode_validates_and_delegates():
    """Verify that image mode validates the image URI and delegates to image compatibility dispatch."""
    workflow = Path("workflows/source-orchestrator.yaml").read_text(encoding="utf-8")

    # Image mode routing
    assert "- run_image_mode:" in workflow
    assert "- delegate_image_mode:" in workflow
    assert "call: run_image_path_compat" in workflow
    assert "- invalid_image_path:" in workflow

    # Image URI regex validation
    assert "([Pp][Nn][Gg]|[Jj][Pp][Ee]?[Gg])" in workflow


def test_source_orchestrator_stage5_strict_toggle_present():
    workflow = Path("workflows/source-orchestrator.yaml").read_text(encoding="utf-8")

    assert "ARENA_EXPORT_REQUIRED" in workflow
    assert "arenaExportRequired" in workflow
    assert "- run_stage5_requirement_switch:" in workflow
    assert "- run_stage5_required:" in workflow
    assert "- run_stage5_non_blocking:" in workflow
    assert "arena_required" in workflow


def test_text_autonomy_daily_workflow_contract_has_lock_pause_emit_wait_state():
    workflow = Path("workflows/text-autonomy-daily.yaml").read_text(encoding="utf-8")

    assert "- acquire_lock:" in workflow
    assert "ifGenerationMatch: 0" in workflow
    assert "TEXT_DAILY_PAUSE_AFTER_CONSEC_FAILS" in workflow
    assert "text-request-emitter-job" in workflow
    assert "wait_for_scene_terminal:" in workflow
    assert ".source_orchestrator_complete" in workflow
    assert ".source_orchestrator_failed" in workflow
    assert "consecutive_failures" in workflow
    assert "run_summary.json" in workflow
    assert "newConsecutiveFailures" in workflow
    assert "newConsecutiveFailures >= pauseAfterConsecutiveFails" in workflow


def test_setup_text_autonomy_scheduler_has_expected_schedule_defaults():
    script = Path("workflows/setup-text-autonomy-scheduler.sh").read_text(encoding="utf-8")

    assert 'SCHEDULER_SCHEDULE=${TEXT_AUTONOMY_SCHEDULER_CRON:-"0 9 * * *"}' in script
    assert 'SCHEDULER_TIMEZONE=${TEXT_AUTONOMY_TIMEZONE:-"America/New_York"}' in script
    assert 'TEXT_DAILY_QUOTA=${TEXT_DAILY_QUOTA:-"1"}' in script
    assert "TEXT_AUTONOMY_STATE_PREFIX" in script
    assert "setup-text-autonomy-scheduler.sh" in script


def test_setup_source_orchestrator_trigger_includes_text_asset_retrieval_envs():
    script = Path("workflows/setup-source-orchestrator-trigger.sh").read_text(encoding="utf-8")

    assert "TEXT_ASSET_RETRIEVAL_ENABLED" in script
    assert "TEXT_ASSET_LIBRARY_PREFIXES" in script
    assert "TEXT_ASSET_LIBRARY_MAX_FILES" in script
    assert "TEXT_ASSET_LIBRARY_MIN_SCORE" in script
    assert "TEXT_ASSET_RETRIEVAL_MODE" in script
    assert "TEXT_ASSET_ANN_ENABLED" in script
    assert "TEXT_ASSET_ANN_TOP_K" in script
    assert "TEXT_ASSET_ANN_MIN_SCORE" in script
    assert "TEXT_ASSET_ANN_MAX_RERANK" in script
    assert "TEXT_ASSET_ANN_NAMESPACE" in script
    assert "TEXT_ASSET_LEXICAL_FALLBACK_ENABLED" in script
    assert "TEXT_ASSET_ROLLOUT_STATE_PREFIX" in script
    assert "TEXT_ASSET_EMBEDDING_QUEUE_PREFIX" in script
    assert "TEXT_ASSET_EMBEDDING_PROCESSED_PREFIX" in script
    assert "TEXT_ASSET_EMBEDDING_FAILED_PREFIX" in script
    assert "TEXT_ASSET_EMBEDDING_MODEL" in script
    assert "VECTOR_STORE_PROVIDER" in script
    assert "VECTOR_STORE_PROJECT_ID" in script
    assert "VECTOR_STORE_LOCATION" in script
    assert "VECTOR_STORE_NAMESPACE" in script
    assert "VECTOR_STORE_DIMENSION" in script
    assert "VERTEX_INDEX_ENDPOINT" in script
    assert "VERTEX_DEPLOYED_INDEX_ID" in script
    assert "TEXT_ASSET_CATALOG_ENABLED" in script
    assert "TEXT_ASSET_REPLICATION_ENABLED" in script
    assert "TEXT_ASSET_REPLICATION_QUEUE_PREFIX" in script
    assert "TEXT_ASSET_REPLICATION_TARGET" in script
    assert "TEXT_ASSET_REPLICATION_TARGET_PREFIX" in script
    assert "TEXT_ASSET_GENERATION_ENABLED" in script
    assert "TEXT_ASSET_GENERATION_PROVIDER" in script
    assert "TEXT_ASSET_GENERATION_PROVIDER_CHAIN" in script
    assert "TEXT_SAM3D_API_HOST" in script
    assert "TEXT_SAM3D_TEXT_ENDPOINTS" in script
    assert "TEXT_HUNYUAN_API_HOST" in script
    assert "TEXT_HUNYUAN_TEXT_ENDPOINTS" in script


def test_asset_replication_workflow_and_trigger_contract():
    workflow = Path("workflows/asset-replication-pipeline.yaml").read_text(encoding="utf-8")
    setup_script = Path("workflows/setup-asset-replication-trigger.sh").read_text(encoding="utf-8")
    master_script = Path("workflows/setup-all-triggers.sh").read_text(encoding="utf-8")

    assert "asset-replication-job" in workflow
    assert "automation/asset_replication/queue" in workflow
    assert "QUEUE_OBJECT" in workflow
    assert "B2_S3_ENDPOINT" not in workflow
    assert "B2_BUCKET" not in workflow
    assert "B2_APPLICATION_KEY" not in workflow

    assert "asset-replication-pipeline" in setup_script
    assert "asset-replication-queue-trigger" in setup_script
    assert "B2_S3_ENDPOINT" in setup_script
    assert "B2_BUCKET" in setup_script
    assert "B2_KEY_ID_SECRET" in setup_script
    assert "B2_APPLICATION_KEY_SECRET" in setup_script
    assert "--update-secrets" in setup_script

    assert "ENABLE_ASSET_REPLICATION" in master_script
    assert "setup-asset-replication-trigger.sh" in master_script


def test_asset_embedding_workflow_and_trigger_contract():
    workflow = Path("workflows/asset-embedding-pipeline.yaml").read_text(encoding="utf-8")
    setup_script = Path("workflows/setup-asset-embedding-trigger.sh").read_text(encoding="utf-8")
    master_script = Path("workflows/setup-all-triggers.sh").read_text(encoding="utf-8")

    assert "asset-embedding-job" in workflow
    assert "automation/asset_embedding/queue" in workflow
    assert "QUEUE_OBJECT" in workflow
    assert "TEXT_ASSET_EMBEDDING_QUEUE_PREFIX" in workflow

    assert "asset-embedding-pipeline" in setup_script
    assert "asset-embedding-queue-trigger" in setup_script
    assert "ASSET_EMBEDDING_JOB_NAME" in setup_script
    assert "VECTOR_STORE_PROVIDER" in setup_script
    assert "VERTEX_INDEX_ENDPOINT" in setup_script
    assert "VERTEX_DEPLOYED_INDEX_ID" in setup_script
    assert "OPENAI_API_KEY_SECRET" in setup_script
    assert "--update-secrets" in setup_script

    assert "ENABLE_ASSET_EMBEDDING" in master_script
    assert "setup-asset-embedding-trigger.sh" in master_script
