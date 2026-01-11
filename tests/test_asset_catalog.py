"""Tests for asset catalog modules."""

import json
import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from tools.asset_catalog.catalog_builder import AssetCatalogBuilder, AssetMetadata
from tools.asset_catalog.asset_matcher import AssetMatcher, AssetMatch
from tools.asset_catalog.ingestion import AssetIngestionConfig, AssetIngestionPipeline
from tools.asset_catalog.embeddings import EmbeddingGenerator
from tools.asset_catalog.vector_store import VectorStore


class TestAssetMetadata:
    """Test AssetMetadata class."""

    def test_asset_metadata_creation(self):
        """Test creating asset metadata."""
        metadata = AssetMetadata(
            asset_id="chair_001",
            name="Office Chair",
            category="furniture",
            description="A modern office chair",
        )
        assert metadata.asset_id == "chair_001"
        assert metadata.name == "Office Chair"
        assert metadata.category == "furniture"

    def test_asset_metadata_with_tags(self):
        """Test asset metadata with tags."""
        metadata = AssetMetadata(
            asset_id="chair_001",
            name="Office Chair",
            category="furniture",
            tags=["modern", "office", "seating"],
        )
        assert "modern" in metadata.tags
        assert len(metadata.tags) == 3

    def test_asset_metadata_to_dict(self):
        """Test serializing asset metadata."""
        metadata = AssetMetadata(
            asset_id="chair_001",
            name="Office Chair",
            category="furniture",
        )
        metadata_dict = metadata.to_dict()
        assert metadata_dict["asset_id"] == "chair_001"
        assert metadata_dict["name"] == "Office Chair"


class TestAssetCatalogBuilder:
    """Test AssetCatalogBuilder class."""

    @pytest.fixture
    def temp_catalog_dir(self):
        """Create temporary directory for catalog."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    def test_catalog_builder_init(self, temp_catalog_dir):
        """Test initializing catalog builder."""
        builder = AssetCatalogBuilder(catalog_dir=str(temp_catalog_dir))
        assert builder.catalog_dir == Path(temp_catalog_dir)
        assert len(builder.assets) == 0

    def test_add_asset(self, temp_catalog_dir):
        """Test adding asset to catalog."""
        builder = AssetCatalogBuilder(catalog_dir=str(temp_catalog_dir))

        metadata = AssetMetadata(
            asset_id="chair_001",
            name="Office Chair",
            category="furniture",
        )

        builder.add_asset(metadata)
        assert len(builder.assets) == 1
        assert builder.assets["chair_001"] == metadata

    def test_build_catalog(self, temp_catalog_dir):
        """Test building catalog."""
        builder = AssetCatalogBuilder(catalog_dir=str(temp_catalog_dir))

        # Add multiple assets
        for i in range(3):
            metadata = AssetMetadata(
                asset_id=f"asset_{i:03d}",
                name=f"Asset {i}",
                category="test",
            )
            builder.add_asset(metadata)

        # Build catalog
        catalog = builder.build()
        assert len(catalog["assets"]) == 3
        assert catalog["metadata"]["total_assets"] == 3

    def test_save_catalog(self, temp_catalog_dir):
        """Test saving catalog to file."""
        builder = AssetCatalogBuilder(catalog_dir=str(temp_catalog_dir))

        metadata = AssetMetadata(
            asset_id="chair_001",
            name="Office Chair",
            category="furniture",
        )
        builder.add_asset(metadata)

        # Save catalog
        catalog_file = temp_catalog_dir / "catalog.json"
        builder.save(catalog_file)

        assert catalog_file.exists()

        # Verify contents
        with open(catalog_file) as f:
            saved_catalog = json.load(f)
        assert len(saved_catalog["assets"]) == 1


class TestAssetMatcher:
    """Test AssetMatcher class."""

    def test_asset_matcher_init(self):
        """Test initializing asset matcher."""
        matcher = AssetMatcher()
        assert matcher is not None

    def test_match_by_category(self):
        """Test matching assets by category."""
        matcher = AssetMatcher()

        assets = [
            AssetMetadata(asset_id="chair_001", category="furniture"),
            AssetMetadata(asset_id="table_001", category="furniture"),
            AssetMetadata(asset_id="lamp_001", category="lighting"),
        ]

        matches = matcher.match_by_category(assets, "furniture")
        assert len(matches) == 2

    def test_match_by_tags(self):
        """Test matching assets by tags."""
        matcher = AssetMatcher()

        assets = [
            AssetMetadata(
                asset_id="chair_001",
                tags=["modern", "office"],
            ),
            AssetMetadata(
                asset_id="table_001",
                tags=["modern", "wood"],
            ),
            AssetMetadata(
                asset_id="lamp_001",
                tags=["vintage"],
            ),
        ]

        matches = matcher.match_by_tags(assets, ["modern"])
        assert len(matches) == 2

    def test_match_by_description(self):
        """Test matching assets by description."""
        matcher = AssetMatcher()

        assets = [
            AssetMetadata(
                asset_id="chair_001",
                description="A comfortable office chair with wheels",
            ),
            AssetMetadata(
                asset_id="table_001",
                description="A wooden dining table",
            ),
        ]

        matches = matcher.match_by_description(assets, "office")
        assert len(matches) >= 1


class TestAssetIngestionPipeline:
    """Test AssetIngestionPipeline class."""

    @pytest.fixture
    def temp_ingestion_dir(self):
        """Create temporary directory for ingestion."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    def test_ingestion_config_creation(self):
        """Test creating ingestion config."""
        config = AssetIngestionConfig(
            source_dir="/path/to/assets",
            output_dir="/path/to/output",
        )
        assert config.source_dir == "/path/to/assets"
        assert config.output_dir == "/path/to/output"

    def test_ingestion_config_validation(self):
        """Test validating ingestion config."""
        config = AssetIngestionConfig(
            source_dir="/nonexistent/path",
            output_dir="/tmp/output",
            validate_paths=False,  # Skip validation for test
        )
        assert config.source_dir == "/nonexistent/path"

    def test_ingestion_pipeline_init(self, temp_ingestion_dir):
        """Test initializing ingestion pipeline."""
        config = AssetIngestionConfig(
            source_dir=str(temp_ingestion_dir),
            output_dir=str(temp_ingestion_dir / "output"),
            validate_paths=False,
        )
        pipeline = AssetIngestionPipeline(config)
        assert pipeline.config == config


class TestEmbeddingGenerator:
    """Test EmbeddingGenerator class."""

    @patch("tools.asset_catalog.embeddings.HuggingFaceEmbeddings")
    def test_embedding_generator_init(self, mock_embeddings):
        """Test initializing embedding generator."""
        generator = EmbeddingGenerator(model_name="test-model")
        assert generator.model_name == "test-model"

    @patch("tools.asset_catalog.embeddings.HuggingFaceEmbeddings")
    def test_generate_embedding(self, mock_embeddings_class):
        """Test generating embedding."""
        mock_embeddings = MagicMock()
        mock_embeddings.embed_query.return_value = [0.1, 0.2, 0.3]
        mock_embeddings_class.return_value = mock_embeddings

        generator = EmbeddingGenerator(model_name="test-model")
        embedding = generator.generate("Test description")

        assert len(embedding) == 3
        assert embedding[0] == 0.1

    @patch("tools.asset_catalog.embeddings.HuggingFaceEmbeddings")
    def test_generate_batch_embeddings(self, mock_embeddings_class):
        """Test generating batch embeddings."""
        mock_embeddings = MagicMock()
        mock_embeddings.embed_documents.return_value = [
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6],
        ]
        mock_embeddings_class.return_value = mock_embeddings

        generator = EmbeddingGenerator(model_name="test-model")
        embeddings = generator.generate_batch(["Text 1", "Text 2"])

        assert len(embeddings) == 2
        assert len(embeddings[0]) == 3


class TestVectorStore:
    """Test VectorStore class."""

    @pytest.fixture
    def temp_vector_store_dir(self):
        """Create temporary directory for vector store."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    def test_vector_store_init(self, temp_vector_store_dir):
        """Test initializing vector store."""
        store = VectorStore(persistence_dir=str(temp_vector_store_dir))
        assert store is not None

    def test_add_vector(self, temp_vector_store_dir):
        """Test adding vector to store."""
        store = VectorStore(persistence_dir=str(temp_vector_store_dir))

        store.add(
            vector_id="vec_001",
            vector=[0.1, 0.2, 0.3],
            metadata={"name": "test_vector"},
        )

        assert store.size() >= 1

    def test_search_vectors(self, temp_vector_store_dir):
        """Test searching vectors."""
        store = VectorStore(persistence_dir=str(temp_vector_store_dir))

        # Add vectors
        store.add("vec_001", [0.1, 0.2, 0.3], {"name": "v1"})
        store.add("vec_002", [0.11, 0.21, 0.31], {"name": "v2"})
        store.add("vec_003", [0.5, 0.6, 0.7], {"name": "v3"})

        # Search for similar vectors
        query = [0.1, 0.2, 0.3]
        results = store.search(query, top_k=2)

        assert len(results) <= 2
        # First result should be the exact match
        if results:
            assert results[0][0] == "vec_001"


class TestAssetCatalogIntegration:
    """Integration tests for asset catalog."""

    @pytest.fixture
    def temp_catalog_dir(self):
        """Create temporary directory for catalog."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    def test_full_catalog_workflow(self, temp_catalog_dir):
        """Test complete catalog workflow."""
        # Create catalog
        builder = AssetCatalogBuilder(catalog_dir=str(temp_catalog_dir))

        # Add assets
        assets_data = [
            {"id": "chair_001", "name": "Office Chair", "category": "furniture", "tags": ["modern"]},
            {"id": "table_001", "name": "Dining Table", "category": "furniture", "tags": ["wood"]},
            {"id": "lamp_001", "name": "Table Lamp", "category": "lighting", "tags": ["modern"]},
        ]

        for asset_data in assets_data:
            metadata = AssetMetadata(
                asset_id=asset_data["id"],
                name=asset_data["name"],
                category=asset_data["category"],
                tags=asset_data["tags"],
            )
            builder.add_asset(metadata)

        # Build and save
        catalog = builder.build()
        assert catalog["metadata"]["total_assets"] == 3

        # Save to file
        catalog_file = temp_catalog_dir / "catalog.json"
        builder.save(catalog_file)
        assert catalog_file.exists()

        # Load and verify
        with open(catalog_file) as f:
            loaded_catalog = json.load(f)
        assert len(loaded_catalog["assets"]) == 3

        # Test matching
        matcher = AssetMatcher()
        assets = [
            AssetMetadata(
                asset_id=asset_data["id"],
                name=asset_data["name"],
                category=asset_data["category"],
                tags=asset_data["tags"],
            )
            for asset_data in assets_data
        ]

        furniture = matcher.match_by_category(assets, "furniture")
        assert len(furniture) == 2

        modern = matcher.match_by_tags(assets, ["modern"])
        assert len(modern) == 2
