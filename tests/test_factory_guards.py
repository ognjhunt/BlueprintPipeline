from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
EPISODE_JOB_DIR = REPO_ROOT / "episode-generation-job"
GENIESIM_ADAPTER_DIR = REPO_ROOT / "tools" / "geniesim_adapter"

for path in (REPO_ROOT, EPISODE_JOB_DIR, GENIESIM_ADAPTER_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from motion_planner import PlannerBackend
from tools.arena_integration.mimic_integration import AugmentationTransform, MimicConfig
from tools.asset_catalog.vector_store import (
    BaseVectorStore,
    InMemoryVectorStore,
    VectorStoreClient,
    VectorStoreConfig,
)
from tools.llm_client.client import LLMClient, LLMProvider, LLMResponse

import geniesim_grpc_pb2_grpc
import geniesim_server
from tools.geniesim_adapter import geniesim_grpc_pb2_grpc as _grpc_module


class DummyClient(LLMClient):
    provider = LLMProvider.ANTHROPIC

    @property
    def default_model(self) -> str:
        return "dummy"

    def generate(self, prompt: str, **kwargs) -> LLMResponse:
        return LLMResponse(text="ok", provider=self.provider, model=self.model)

    def _generate_image(self, prompt: str, size: str = "1024x1024", **kwargs) -> LLMResponse:
        return LLMResponse(text="image", provider=self.provider, model=self.model)


def test_planner_backend_is_abstract() -> None:
    with pytest.raises(TypeError):
        PlannerBackend()


def test_augmentation_transform_is_abstract() -> None:
    with pytest.raises(TypeError):
        AugmentationTransform(MimicConfig())


def test_vector_store_client_returns_concrete_store() -> None:
    client = VectorStoreClient(VectorStoreConfig(provider="in-memory"))
    assert isinstance(client.store, InMemoryVectorStore)
    assert type(client.store) is not BaseVectorStore


def test_generate_image_returns_structured_failure_for_unsupported_provider() -> None:
    client = DummyClient()
    response = client.generate_image("draw a cube")

    assert response.error_message
    assert response.images == []


def test_geniesim_server_local_servicer_is_concrete() -> None:
    assert issubclass(
        geniesim_server.GenieSimLocalServicer,
        _grpc_module.SimObservationServiceServicer,
    )
