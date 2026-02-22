import builtins


def test_default_steps_without_selector_excludes_dream2flow(monkeypatch, tmp_path):
    """Ensure Dream2Flow steps stay gated when pipeline selector is unavailable."""
    real_import = builtins.__import__

    def blocked_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "tools.pipeline_selector.selector":
            raise ImportError("blocked for test")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", blocked_import)

    from tools.run_local_pipeline import LocalPipelineRunner, PipelineStep

    runner = LocalPipelineRunner(
        scene_dir=tmp_path,
        verbose=False,
        skip_interactive=True,
        environment_type="kitchen",
        enable_dwm=False,
        enable_dream2flow=False,
    )

    steps = runner._resolve_default_steps()

    assert steps == [
        PipelineStep.TEXT_SCENE_GEN,
        PipelineStep.TEXT_SCENE_ADAPTER,
        PipelineStep.SCALE,
        PipelineStep.INTERACTIVE,
        PipelineStep.SIMREADY,
        PipelineStep.USD,
        PipelineStep.REPLICATOR,
        PipelineStep.GENIESIM_EXPORT,
        PipelineStep.GENIESIM_SUBMIT,
        PipelineStep.GENIESIM_IMPORT,
    ]
    assert PipelineStep.DREAM2FLOW not in steps
    assert PipelineStep.DREAM2FLOW_INFERENCE not in steps


def test_dream2flow_steps_require_explicit_flag(monkeypatch, tmp_path):
    """Ensure Dream2Flow steps only appear when explicitly enabled."""
    real_import = builtins.__import__

    def blocked_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "tools.pipeline_selector.selector":
            raise ImportError("blocked for test")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", blocked_import)

    from tools.run_local_pipeline import LocalPipelineRunner, PipelineStep

    runner = LocalPipelineRunner(
        scene_dir=tmp_path,
        verbose=False,
        skip_interactive=True,
        environment_type="kitchen",
        enable_dwm=False,
        enable_dream2flow=True,
    )

    steps = runner._resolve_default_steps()

    assert steps[-2:] == [
        PipelineStep.DREAM2FLOW,
        PipelineStep.DREAM2FLOW_INFERENCE,
    ]


def test_production_enables_checkpoint_hashes(monkeypatch, tmp_path):
    monkeypatch.setenv("PIPELINE_ENV", "production")
    monkeypatch.setenv("BP_QUALITY_HUMAN_APPROVAL_NOTIFICATION_CHANNELS", "#test-approvals")
    monkeypatch.delenv("BP_CHECKPOINT_HASHES", raising=False)

    from tools.run_local_pipeline import LocalPipelineRunner

    runner = LocalPipelineRunner(
        scene_dir=tmp_path,
        verbose=False,
        skip_interactive=True,
        environment_type="kitchen",
        enable_dwm=False,
        enable_dream2flow=False,
    )

    assert runner.enable_checkpoint_hashes is True


def test_episode_generation_job_maps_to_local_step_without_unsupported_warning(tmp_path):
    from tools.run_local_pipeline import LocalPipelineRunner, PipelineStep

    runner = LocalPipelineRunner(
        scene_dir=tmp_path,
        verbose=False,
        skip_interactive=True,
        environment_type="kitchen",
        enable_dwm=False,
        enable_dream2flow=False,
    )
    log_messages = []
    runner.log = lambda msg, level="INFO": log_messages.append((level, msg))  # type: ignore[assignment]

    steps = runner._map_jobs_to_steps(["episode-generation-job"])

    assert steps == [PipelineStep.EPISODE_GENERATION]
    assert not any("unsupported local job" in msg for _level, msg in log_messages)


def test_episode_generation_step_dependency_and_expected_outputs(tmp_path):
    from tools.run_local_pipeline import LocalPipelineRunner, PipelineStep

    runner = LocalPipelineRunner(
        scene_dir=tmp_path,
        verbose=False,
        skip_interactive=True,
        environment_type="kitchen",
        enable_dwm=False,
        enable_dream2flow=False,
    )

    deps = runner._resolve_step_dependencies(
        [PipelineStep.ISAAC_LAB, PipelineStep.EPISODE_GENERATION]
    )
    assert deps[PipelineStep.EPISODE_GENERATION] == [PipelineStep.ISAAC_LAB]

    expected_outputs = runner._expected_output_paths(PipelineStep.EPISODE_GENERATION)
    assert runner.episodes_dir / ".episodes_complete" in expected_outputs
    assert runner.episodes_dir / "quality" / "validation_report.json" in expected_outputs
