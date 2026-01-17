# Public API Inventory (Initial)

This inventory lists top-level public functions (non-underscore) discovered via AST parsing.
It is intended as a starting point for API coverage and may be expanded with classes and methods later.

## episode-generation-job/

### episode-generation-job/collision_aware_planner.py

- `enhance_motion_plan_with_collision_avoidance`
- `is_curobo_available`

### episode-generation-job/cpgen_augmenter.py

- `augment_episode`

### episode-generation-job/curobo_planner.py

- `create_curobo_planner`
- `is_curobo_available`

### episode-generation-job/data_pack_config.py

- `create_core_pack`
- `create_full_pack`
- `create_plus_pack`
- `data_pack_from_string`
- `get_data_pack_config`
- `get_leorbot_feature_config`
- `get_tier_comparison`

### episode-generation-job/generate_episodes.py

- `main`
- `run_episode_generation_job`

### episode-generation-job/isaac_sim_enforcement.py

- `check_production_ready`
- `enforce_isaac_sim_for_production`
- `get_data_quality_level`
- `get_environment_capabilities`
- `print_environment_report`
- `require_isaac_sim`

### episode-generation-job/isaac_sim_integration.py

- `get_availability_status`
- `get_isaac_sim_session`
- `is_curobo_available`
- `is_isaac_lab_available`
- `is_isaac_sim_available`
- `is_physx_available`
- `is_replicator_available`
- `print_availability_report`

### episode-generation-job/lerobot_exporter.py

- `export_trajectories_to_lerobot`

### episode-generation-job/motion_planner.py

- `plan_drawer_open`
- `plan_pick_place`

### episode-generation-job/policy_config_loader.py

- `load_motion_planner_timing`
- `load_policy_config`
- `load_validation_thresholds`

### episode-generation-job/pytorch_dataloaders.py

- `blueprint_collate_fn`
- `count_episodes`
- `create_blueprint_dataloader`
- `get_dataset_info`

### episode-generation-job/quality_certificate.py

- `compute_episode_data_hash`

### episode-generation-job/quality_constants.py

- `get_quality_thresholds`
- `get_training_suitability_level`
- `meets_minimum_quality`

### episode-generation-job/reward_computation.py

- `compute_dense_rewards`
- `compute_episode_reward`

### episode-generation-job/sensor_data_capture.py

- `check_sensor_capture_environment`
- `create_sensor_capture`
- `get_capture_mode_from_env`
- `require_isaac_sim_or_fail`

### episode-generation-job/sim_validator.py

- `filter_valid_episodes`
- `validate_episode`

### episode-generation-job/task_specifier.py

- `specify_task`

### episode-generation-job/trajectory_solver.py

- `get_robot_config`
- `get_robot_info`
- `list_supported_robots`
- `solve_trajectory`

### episode-generation-job/usd_scene_scan.py

- `discover_camera_prim_specs`
- `discover_robot_prim_paths`
- `get_usd_stage`
- `resolve_robot_prim_paths`

## genie-sim-export-job/

### genie-sim-export-job/default_audio_narration.py

- `create_default_audio_narration_exporter`

### genie-sim-export-job/default_embodiment_transfer.py

- `compute_action_space_compatibility`
- `compute_kinematic_similarity`
- `compute_workspace_overlap`
- `create_default_embodiment_transfer_exporter`

### genie-sim-export-job/default_generalization_analyzer.py

- `create_default_generalization_analyzer_exporter`

### genie-sim-export-job/default_language_annotations.py

- `create_default_language_annotations_exporter`

### genie-sim-export-job/default_policy_leaderboard.py

- `create_default_policy_leaderboard_exporter`

### genie-sim-export-job/default_premium_analytics.py

- `create_default_premium_analytics_exporter`

### genie-sim-export-job/default_sim2real_fidelity.py

- `create_default_sim2real_fidelity_exporter`

### genie-sim-export-job/default_sim2real_validation.py

- `create_default_sim2real_validation_exporter`

### genie-sim-export-job/default_tactile_sensor_sim.py

- `create_default_tactile_sensor_exporter`

### genie-sim-export-job/default_trajectory_optimality.py

- `create_default_trajectory_optimality_exporter`

### genie-sim-export-job/export_to_geniesim.py

- `main`
- `parse_bool`
- `run_geniesim_export_job`

### genie-sim-export-job/geniesim_client.py

- `main`
- `safe_extract_tar`
- `validate_robot_type`

## genie-sim-import-job/

### genie-sim-import-job/import_from_geniesim.py

- `convert_to_lerobot`
- `main`
- `run_import_job`
- `run_local_import_job`

### genie-sim-import-job/import_manifest_utils.py

- `build_checksums_map`
- `build_directory_checksums`
- `build_file_inventory`
- `collect_provenance`
- `compute_manifest_checksum`
- `compute_sha256`
- `get_episode_file_paths`
- `get_git_sha`
- `get_lerobot_metadata_paths`
- `get_pipeline_version`
- `get_upstream_versions`
- `iter_files_sorted`
- `snapshot_env`

### genie-sim-import-job/quality_config.py

- `load_quality_config`
- `resolve_min_quality_score`

### genie-sim-import-job/verify_import_manifest.py

- `main`
- `verify_manifest`

## genie-sim-submit-job/

### genie-sim-submit-job/submit_to_geniesim.py

- `main`

## tools/

### tools/arena_integration/affordances.py

- `detect_affordances`
- `detect_affordances_heuristic`
- `detect_affordances_llm`

### tools/arena_integration/arena_exporter.py

- `export_scene_to_arena`

### tools/arena_integration/components.py

- `get_registry`

### tools/arena_integration/composite_tasks.py

- `build_composite_task`
- `evaluate_composite_task`

### tools/arena_integration/evaluation_runner.py

- `run_arena_evaluation`

### tools/arena_integration/groot_integration.py

- `evaluate_groot_on_arena`
- `load_groot_for_arena`

### tools/arena_integration/hub_registration.py

- `register_with_hub`

### tools/arena_integration/lerobot_hub.py

- `generate_hub_spec`
- `publish_to_hub`

### tools/arena_integration/mimic_integration.py

- `augment_genie_sim_episodes`

### tools/arena_integration/parallel_evaluation.py

- `estimate_evaluation_time`
- `run_parallel_evaluation`

### tools/arena_integration/task_mapping.py

- `get_arena_tasks_for_affordances`

### tools/articulation/detector.py

- `detect_scene_articulations`

### tools/articulation_wiring/wiring.py

- `find_articulated_assets`
- `parse_urdf`
- `update_manifest_with_articulation`
- `validate_urdf`
- `validate_urdf_for_simulation`
- `wire_articulation_to_scene`

### tools/asset_catalog/image_captioning.py

- `caption_thumbnail`

### tools/checkpoint/retention_cleanup.py

- `cleanup_scene`
- `load_retention_policy`
- `main`
- `parse_args`

### tools/checkpoint/store.py

- `checkpoint_dir`
- `checkpoint_path`
- `load_checkpoint`
- `should_skip_step`
- `write_checkpoint`

### tools/config/seed_manager.py

- `configure_pipeline_seed`
- `get_pipeline_seed`
- `set_global_seed`

### tools/cost_tracking/estimate.py

- `estimate_gpu_costs`
- `format_estimate_summary`
- `load_estimate_config`
- `main`
- `resolve_steps_for_scene`

### tools/cost_tracking/tracker.py

- `get_cost_tracker`

### tools/error_handling/dead_letter.py

- `get_dead_letter_queue`

### tools/error_handling/errors.py

- `classify_exception`

### tools/error_handling/job_wrapper.py

- `publish_failure`
- `run_job_with_dead_letter_queue`

### tools/error_handling/partial_failure.py

- `process_with_partial_failure`
- `save_successful_items`

### tools/error_handling/retry.py

- `async_retry_with_backoff`
- `calculate_delay`
- `retry_with_backoff`
- `should_retry`

### tools/error_handling/timeout.py

- `monitored_timeout`
- `timeout`
- `timeout_thread`
- `with_timeout`

### tools/external_services/service_client.py

- `create_gcs_client`
- `create_gemini_client`
- `create_genie_sim_client`
- `create_particulate_client`

### tools/geniesim_adapter/asset_index.py

- `build_asset_index`

### tools/geniesim_adapter/exporter.py

- `export_to_geniesim`

### tools/geniesim_adapter/geniesim_grpc_pb2_grpc.py

- `add_GenieSimServiceServicer_to_server`
- `create_channel`
- `is_grpc_available`

### tools/geniesim_adapter/geniesim_healthcheck.py

- `main`

### tools/geniesim_adapter/geniesim_server.py

- `main`
- `parse_args`
- `run_health_check`
- `serve`

### tools/geniesim_adapter/local_framework.py

- `build_geniesim_preflight_report`
- `check_geniesim_availability`
- `format_geniesim_preflight_failure`
- `main`
- `run_geniesim_preflight`
- `run_geniesim_preflight_or_exit`
- `run_local_data_collection`

### tools/geniesim_adapter/multi_robot_config.py

- `get_geniesim_robot_config`
- `get_robot_spec`
- `save_multi_robot_config`

### tools/geniesim_adapter/scene_graph.py

- `convert_manifest_to_scene_graph`

### tools/geniesim_adapter/task_config.py

- `generate_task_config`

### tools/inventory_enrichment/enricher.py

- `enrich_inventory_file`
- `get_inventory_enricher`

### tools/isaac_lab_tasks/env_config.py

- `get_isaac_asset_path`

### tools/isaac_lab_tasks/multi_robot.py

- `create_dual_arm_config`
- `create_robot_fleet_config`
- `generate_multi_robot_env_config`
- `generate_multi_robot_reward_code`

### tools/isaac_lab_tasks/policy_templates.py

- `generate_reward_functions_code`
- `generate_termination_functions_code`
- `get_policy_template`
- `get_required_observations`
- `get_reward_weights`

### tools/isaac_lab_tasks/run_training.py

- `main`
- `rollout_sanity_check`

### tools/isaac_lab_tasks/runtime_validator.py

- `validate_generated_package`

### tools/isaac_lab_tasks/task_generator.py

- `load_physics_profiles`
- `load_robot_embodiments`
- `select_physics_profile`

### tools/job_registry/registry.py

- `get_registry`

### tools/llm_client/client.py

- `create_llm_client`
- `get_default_provider`

### tools/material_transfer/material_transfer.py

- `apply_materials_to_usd`
- `create_material_manifest`
- `extract_materials_from_glb`
- `infer_material_type`
- `transfer_materials`

### tools/mesh_processing/lod_generator.py

- `apply_lod_to_usd`
- `generate_lod_chain`

### tools/mesh_processing/texture_compression.py

- `compress_texture`
- `compress_textures_batch`
- `get_recommended_format`

### tools/metrics/pipeline_analytics.py

- `get_dashboard_data`
- `get_success_summary`
- `get_tracker`
- `track_customer_feedback`
- `track_pipeline_run`
- `track_scene_delivery`
- `track_training_outcome`
- `update_pipeline_status`

### tools/metrics/pipeline_metrics.py

- `get_metrics`
- `reset_metrics`

### tools/nvidia_pack_indexer.py

- `build_asset_document`
- `build_embedding`
- `build_embedding_text`
- `index_pack`
- `index_regen3d_assets`
- `load_pack_assets`
- `main`
- `map_type_to_role`
- `parse_args`
- `safe_slug`

### tools/performance/parallel_processing.py

- `process_in_batches`
- `process_parallel_multiprocess`
- `process_parallel_threaded`

### tools/performance/streaming_json.py

- `process_large_manifest`
- `stream_json_array`
- `stream_manifest_objects`

### tools/pipeline_selector/selector.py

- `get_active_pipeline_mode`
- `get_data_generation_backend`
- `is_geniesim_enabled`
- `select_pipeline`
- `should_skip_deprecated_job`

### tools/qa_validation/validator.py

- `run_qa_validation`

### tools/quality_gates/ai_qa_context.py

- `generate_qa_context`

### tools/quality_gates/notification_service.py

- `send_email_notification`
- `send_sms_notification`

### tools/quality_gates/sli_gate_runner.py

- `main`

### tools/quality_reports/asset_provenance_generator.py

- `generate_asset_provenance`

### tools/quality_reports/scene_report_generator.py

- `generate_scene_report`

### tools/regen3d_adapter/adapter.py

- `layout_from_regen3d`
- `manifest_from_regen3d`

### tools/run_first_10_scenes.py

- `create_test_scenes`
- `main`

### tools/run_full_isaacsim_pipeline.py

- `initialize_isaac_sim`
- `main`

### tools/scale_authority/authority.py

- `apply_scale_to_layout`
- `apply_scale_to_manifest`
- `validate_scale`

### tools/scene_manifest/loader.py

- `load_manifest`
- `load_manifest_or_scene_assets`

### tools/scene_manifest/validate_manifest.py

- `load_json`
- `main`
- `manifest_from_blueprint_recipe`
- `manifest_from_scene_assets`
- `normalize_sim_role`
- `parse_args`
- `validate_manifest`
- `write_output`

### tools/secrets/secret_manager.py

- `create_secret`
- `get_global_secret_cache`
- `get_secret`
- `get_secret_or_env`
- `load_pipeline_secrets`
- `update_secret`

### tools/sim2real/experiments.py

- `analyze_experiment`
- `get_aggregate_stats`
- `get_validator`
- `log_real_world_result`
- `register_experiment`

### tools/sim2real/metrics.py

- `compute_confidence_interval`
- `compute_failure_mode_distribution`
- `compute_policy_divergence`
- `compute_sample_size_for_power`
- `compute_success_rate`
- `compute_timing_ratio`
- `compute_transfer_gap`
- `interpret_transfer_quality`

### tools/smoke_manifest_pipeline.py

- `main`
- `smoke_manifest_to_usd`
- `smoke_replicator_bundle`
- `smoke_simready`

### tools/startup_validation.py

- `print_validation_report`
- `validate_all_credentials`
- `validate_and_fail_fast`
- `validate_gcs_credentials`
- `validate_gemini_credentials`
- `validate_genie_sim_credentials`

### tools/storage_layout/paths.py

- `get_asset_index_path`
- `get_asset_path`
- `get_episodes_path`
- `get_geniesim_path`
- `get_isaac_lab_path`
- `get_layout_path`
- `get_manifest_path`
- `get_replicator_path`
- `get_scene_graph_path`
- `get_scene_paths`
- `get_usd_path`
- `validate_scene_structure`

### tools/sync_requirements_pins.py

- `load_pins`
- `sync_requirements`

### tools/tracing/tracer.py

- `extract_trace_context`
- `get_current_span`
- `get_tracer`
- `init_tracing`
- `inject_trace_context`
- `set_trace_attribute`
- `set_trace_error`
- `trace_function`
- `trace_job`

### tools/validation/config_schemas.py

- `load_and_validate_env_config`
- `load_and_validate_manifest`

### tools/validation/entrypoint_checks.py

- `validate_required_env_vars`
- `validate_scene_manifest`

### tools/validation/input_validation.py

- `sanitize_path`
- `sanitize_string`
- `validate_category`
- `validate_description`
- `validate_dimensions`
- `validate_enum`
- `validate_numeric`
- `validate_object_id`
- `validate_rgb_color`
- `validate_scene_id`
- `validate_url`

### tools/variation_contract/contract.py

- `create_variation_manifest`
- `get_asset_paths`
- `standardize_asset_name`
- `validate_variation_manifest`

### tools/verify_workflow_triggers_and_dryrun.py

- `check_patterns`
- `dry_run_regen3d_pipeline`
- `main`
- `verify_geniesim_export_trigger`
- `verify_usd_assembly`

### tools/workflow/failure_markers.py

- `write_failure_marker`

