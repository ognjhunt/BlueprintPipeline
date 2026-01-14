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
        PipelineStep.REGEN3D,
        PipelineStep.SIMREADY,
        PipelineStep.USD,
        PipelineStep.REPLICATOR,
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
