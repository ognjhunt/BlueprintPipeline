import importlib


def test_pipeline_metrics_import():
    importlib.import_module("tools.metrics.pipeline_metrics")
