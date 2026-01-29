import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.geniesim_adapter.asset_index import (
    AssetIndexBuilder,
    CATEGORY_MAPPING,
    GenieSimAsset,
    SemanticDescriptionGenerator,
)


@pytest.mark.unit
def test_semantic_description_generator_rich_text():
    generator = SemanticDescriptionGenerator()
    obj = {
        "category": "cup",
        "name": "Aqua",
        "description": "a ceramic travel cup",
        "dimensions_est": {"width": 0.1, "depth": 0.2, "height": 0.3},
        "physics_hints": {"material_type": "ceramic"},
        "semantics": {"affordances": ["Grasp", {"type": "pour"}]},
        "sim_role": "articulated_furniture",
    }

    description = generator.generate(obj)

    assert "A cup" in description
    assert "named Aqua" in description
    assert "a ceramic travel cup" in description
    assert "0.10m x 0.20m x 0.30m" in description
    assert "made of ceramic" in description
    assert "can be grasp, pour" in description
    assert "with moving parts" in description


@pytest.mark.unit
def test_asset_index_builder_defaults_unknown_category_and_mass():
    builder = AssetIndexBuilder(generate_embeddings=False, verbose=False)
    obj = {
        "id": "asset-1",
        "category": "weird",
        "sim_role": "manipulable_object",
        "asset": {"path": "assets/item.usdz", "source": "external_nc"},
        "dimensions_est": {"width": 0.2, "depth": 0.1, "height": 0.3},
        "physics_hints": {"material_type": "wood"},
    }

    asset = builder._build_asset(obj, usd_base_path="/data")

    assert asset is not None
    assert asset.categories == CATEGORY_MAPPING["object"]
    assert asset.commercial_ok is False
    assert asset.collision_hull_path == "/data/assets/item_collision.usda"
    assert pytest.approx(asset.mass, rel=1e-6) == 0.2 * 0.1 * 0.3 * 700.0


@pytest.mark.unit
def test_placeholder_embedding_is_deterministic():
    builder = AssetIndexBuilder(generate_embeddings=False, verbose=False)

    embedding_a = builder._get_placeholder_embedding("hello")
    embedding_b = builder._get_placeholder_embedding("hello")
    embedding_c = builder._get_placeholder_embedding("world")

    assert len(embedding_a) == 2048
    assert embedding_a == embedding_b
    assert embedding_a != embedding_c
    assert np.all(np.array(embedding_a) <= 1.0)
    assert np.all(np.array(embedding_a) >= -1.0)


@pytest.mark.unit
def test_resize_embedding_passthrough():
    builder = AssetIndexBuilder(generate_embeddings=False, verbose=False, embedding_dim=768)
    embedding = [float(i) / 768 for i in range(768)]

    resized = builder._resize_embedding(embedding)

    assert resized == embedding


@pytest.mark.unit
def test_resize_embedding_shrink_is_deterministic():
    builder = AssetIndexBuilder(generate_embeddings=False, verbose=False, embedding_dim=768)
    embedding = [float(i) for i in range(2048)]

    resized = builder._resize_embedding(embedding)
    resized_again = builder._resize_embedding(embedding)

    original_indices = np.linspace(0.0, 1.0, num=len(embedding))
    target_indices = np.linspace(0.0, 1.0, num=builder.embedding_dim)
    expected = np.interp(target_indices, original_indices, embedding).tolist()

    assert resized == expected
    assert resized_again == expected


@pytest.mark.unit
def test_resize_embedding_grow_pads_zeros():
    builder = AssetIndexBuilder(generate_embeddings=False, verbose=False, embedding_dim=768)
    embedding = [0.5, 1.5]

    resized = builder._resize_embedding(embedding)

    assert resized[:2] == [0.5, 1.5]
    assert resized[2:] == [0.0] * 766
    assert len(resized) == 768


@pytest.mark.unit
def test_resize_embedding_empty_is_zeros():
    builder = AssetIndexBuilder(generate_embeddings=False, verbose=False, embedding_dim=768)

    resized = builder._resize_embedding([])

    assert resized == [0.0] * 768


@pytest.mark.unit
def test_production_disallows_placeholder_embeddings(monkeypatch):
    monkeypatch.setenv("PIPELINE_ENV", "production")
    builder = AssetIndexBuilder(
        generate_embeddings=True,
        verbose=False,
        require_embeddings=False,
    )
    asset = GenieSimAsset(
        asset_id="asset-1",
        usd_path="/data/asset.usdz",
        semantic_description="test asset",
        categories=["general"],
    )

    def _mock_embedding(_text: str):
        return [0.0] * builder.embedding_dim, "placeholder", "mock placeholder"

    monkeypatch.setattr(builder, "_get_embedding_with_status", _mock_embedding)

    with pytest.raises(RuntimeError, match="Configure an embedding provider"):
        builder._generate_embeddings([asset])
