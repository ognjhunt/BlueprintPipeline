"""Contract checks for base classes and their implementations."""

from __future__ import annotations

import inspect
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "episode-generation-job"))

from motion_planner import (
    CuRoboPlannerBackend,
    IKPlannerBackend,
    OmplPlannerBackend,
    PlannerBackend,
)
from tools.arena_integration.mimic_integration import (
    AugmentationTransform,
    ConstraintPreservingAugmentation,
    NoiseInjection,
    SpatialPerturbation,
    TemporalScaling,
)
from tools.asset_catalog.vector_store import (
    BaseVectorStore,
    InMemoryVectorStore,
    PgVectorStore,
)
from tools.audio.tts_providers import (
    AzureTTSProvider,
    BaseTTSProvider,
    GoogleTTSProvider,
    LocalTTSProvider,
    MockTTSProvider,
    OpenAITTSProvider,
)
from tools.checkpoint.store import (
    BaseCheckpointStore,
    GCSCheckpointStore,
    LocalCheckpointStore,
)


def _assert_concrete(
    cls: type,
    required_methods: list[str],
    base_cls: type | None = None,
) -> None:
    assert not inspect.isabstract(cls), f"{cls.__name__} should be concrete"
    for method_name in required_methods:
        method = getattr(cls, method_name)
        assert callable(method), f"{cls.__name__}.{method_name} should be callable"
        assert not getattr(method, "__isabstractmethod__", False)
        if base_cls is not None:
            assert getattr(cls, method_name) is not getattr(base_cls, method_name)


def test_planner_backend_implementations_are_concrete() -> None:
    # Source: episode-generation-job/motion_planner.py
    for backend in [CuRoboPlannerBackend, OmplPlannerBackend, IKPlannerBackend]:
        _assert_concrete(backend, ["plan"], PlannerBackend)


def test_checkpoint_store_implementations_are_concrete() -> None:
    # Source: tools/checkpoint/store.py
    for store in [LocalCheckpointStore, GCSCheckpointStore]:
        _assert_concrete(store, ["write_checkpoint", "load_checkpoint"], BaseCheckpointStore)


def test_vector_store_implementations_are_concrete() -> None:
    # Source: tools/asset_catalog/vector_store.py
    for store in [InMemoryVectorStore, PgVectorStore]:
        _assert_concrete(store, ["upsert", "query", "fetch", "list"], BaseVectorStore)


def test_tts_provider_implementations_are_concrete() -> None:
    # Source: tools/audio/tts_providers.py
    for provider in [
        GoogleTTSProvider,
        OpenAITTSProvider,
        AzureTTSProvider,
        LocalTTSProvider,
        MockTTSProvider,
    ]:
        _assert_concrete(provider, ["generate_audio"], BaseTTSProvider)


def test_augmentation_transform_implementations_are_concrete() -> None:
    # Source: tools/arena_integration/mimic_integration.py
    for transform in [
        SpatialPerturbation,
        TemporalScaling,
        NoiseInjection,
        ConstraintPreservingAugmentation,
    ]:
        _assert_concrete(transform, ["apply"], AugmentationTransform)
