"""
Asset Embeddings - Vector embeddings for semantic asset search.

This module provides embedding-based asset search using:
- Text embeddings for descriptions and tags
- Image embeddings for thumbnails
- Combined multimodal embeddings

Supports various backends:
- Local sentence-transformers
- Cloud APIs (Vertex AI, OpenAI)
"""

import importlib.util
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from .vector_store import VectorRecord, VectorStoreClient, VectorStoreConfig


@dataclass
class EmbeddingConfig:
    """Configuration for embedding generation."""

    model_name: str = "all-MiniLM-L6-v2"
    dimension: int = 384
    backend: str = "sentence-transformers"  # or "vertex-ai", "openai"
    api_key: Optional[str] = None
    project_id: Optional[str] = None
    image_model_name: Optional[str] = None
    image_dimension: Optional[int] = None
    image_backend: Optional[str] = None


class AssetEmbeddings:
    """
    Manages vector embeddings for asset search.

    Usage:
        embeddings = AssetEmbeddings(config)
        embeddings.build_index(catalog)
        matches = embeddings.search("modern wooden dining table", top_k=10)
    """

    def __init__(
        self,
        config: Optional[EmbeddingConfig] = None,
        vector_store: Optional[VectorStoreClient] = None,
        vector_store_config: Optional[VectorStoreConfig] = None,
        collection: str = "asset-embeddings",
    ):
        self.config = config or EmbeddingConfig()
        self.embeddings: dict[str, np.ndarray] = {}
        self.thumbnail_embeddings: dict[str, np.ndarray] = {}
        self.asset_ids: list[str] = []
        self.thumbnail_asset_ids: list[str] = []
        self.index_matrix: Optional[np.ndarray] = None
        self.thumbnail_index_matrix: Optional[np.ndarray] = None
        self._model = None
        self._client = None
        self._image_model = None
        self._image_client = None
        self.collection = collection
        self.vector_store = vector_store or (
            VectorStoreClient(vector_store_config) if vector_store_config else None
        )

    def _load_model(self) -> None:
        """Load the embedding model."""
        if self._model is not None:
            return

        if self.config.backend == "sentence-transformers":
            try:
                from sentence_transformers import SentenceTransformer

                self._model = SentenceTransformer(self.config.model_name)
            except ImportError:
                print("Warning: sentence-transformers not installed. Using random embeddings.")
                self._model = "stub"
        elif self.config.backend in {"vertex-ai", "gemini"}:
            try:
                import google.generativeai as genai

                if not self.config.api_key:
                    raise ValueError("API key required for Vertex/Gemini embeddings")

                genai.configure(api_key=self.config.api_key, client_options=None)
                self._client = genai
                self._model = self.config.model_name or "models/text-embedding-004"
            except Exception as exc:  # pragma: no cover - dependency not always installed
                raise RuntimeError(f"Failed to initialize Vertex/Gemini embeddings: {exc}") from exc
        elif self.config.backend == "openai":
            try:
                from openai import OpenAI

                if not self.config.api_key:
                    raise ValueError("API key required for OpenAI embeddings")

                self._client = OpenAI(api_key=self.config.api_key)
                self._model = self.config.model_name or "text-embedding-3-small"
            except Exception as exc:  # pragma: no cover - dependency not always installed
                raise RuntimeError(f"Failed to initialize OpenAI embeddings: {exc}") from exc
        else:
            self._model = "stub"

    def _load_image_model(self) -> None:
        """Load the image embedding model."""
        if self._image_model is not None:
            return

        backend = self.config.image_backend or self.config.backend
        model_name = self.config.image_model_name or self.config.model_name

        if backend == "sentence-transformers":
            st_spec = importlib.util.find_spec("sentence_transformers")
            if st_spec is None:
                print("Warning: sentence-transformers not installed. Using random image embeddings.")
                self._image_model = "stub"
                return

            from sentence_transformers import SentenceTransformer

            self._image_model = SentenceTransformer(model_name)
        elif backend in {"vertex-ai", "gemini"}:
            genai_spec = importlib.util.find_spec("google.generativeai")
            if genai_spec is None:
                raise RuntimeError("google.generativeai is required for Gemini image embeddings")

            import google.generativeai as genai

            if not self.config.api_key:
                raise ValueError("API key required for Vertex/Gemini embeddings")

            genai.configure(api_key=self.config.api_key, client_options=None)
            self._image_client = genai
            self._image_model = model_name or "models/image-embedding-001"
        elif backend == "openai":
            openai_spec = importlib.util.find_spec("openai")
            if openai_spec is None:
                raise RuntimeError("openai package is required for OpenAI image embeddings")

            from openai import OpenAI

            if not self.config.api_key:
                raise ValueError("API key required for OpenAI embeddings")

            self._image_client = OpenAI(api_key=self.config.api_key)
            self._image_model = model_name or "text-embedding-3-small"
        else:
            self._image_model = "stub"

    def embed_text(self, text: str) -> np.ndarray:
        """Generate embedding for text."""
        self._load_model()

        if self._model == "stub" or self._model is None:
            # Return random embedding for testing
            np.random.seed(hash(text) % 2 ** 32)
            return np.random.randn(self.config.dimension).astype(np.float32)

        if hasattr(self._model, "encode"):
            return self._model.encode(text, convert_to_numpy=True)

        if self.config.backend in {"vertex-ai", "gemini"} and self._client:
            response = self._client.embed_content(model=self._model, content=text)
            embedding = response.get("embedding") if isinstance(response, dict) else getattr(response, "embedding", None)
            if embedding is None:
                raise RuntimeError("Vertex/Gemini embedding response missing 'embedding'")
            return np.array(embedding, dtype=np.float32)

        if self.config.backend == "openai" and self._client:
            response = self._client.embeddings.create(model=self._model, input=text)
            data = response.data[0].embedding
            return np.array(data, dtype=np.float32)

        # Fallback
        np.random.seed(hash(text) % 2 ** 32)
        return np.random.randn(self.config.dimension).astype(np.float32)

    def embed_image(self, image_path: str) -> np.ndarray:
        """Generate embedding for an image thumbnail."""
        self._load_image_model()

        image_path_obj = Path(image_path)
        if not image_path_obj.exists():
            raise FileNotFoundError(f"Thumbnail not found: {image_path}")

        backend = self.config.image_backend or self.config.backend

        if self._image_model == "stub" or self._image_model is None:
            np.random.seed(hash(image_path_obj.name) % 2 ** 32)
            dim = self.config.image_dimension or self.config.dimension
            return np.random.randn(dim).astype(np.float32)

        if backend == "sentence-transformers":
            pil_spec = importlib.util.find_spec("PIL.Image")
            if pil_spec is None:
                print("Warning: Pillow is required for image embeddings. Using random vectors instead.")
            else:
                from PIL import Image

                img = Image.open(image_path_obj)
                emb = self._image_model.encode(img, convert_to_numpy=True)
                return np.asarray(emb, dtype=np.float32)

        if backend in {"vertex-ai", "gemini"} and self._image_client:
            response = self._image_client.embed_content(
                model=self._image_model,
                content={"parts": [{"inline_data": {"mime_type": "image/png", "data": image_path_obj.read_bytes()}}]},
            )
            embedding = response.get("embedding") if isinstance(response, dict) else getattr(response, "embedding", None)
            if embedding is None:
                raise RuntimeError("Vertex/Gemini image embedding response missing 'embedding'")
            return np.array(embedding, dtype=np.float32)

        if backend == "openai" and self._image_client:
            response = self._image_client.embeddings.create(model=self._image_model, input=image_path_obj.read_bytes())
            data = response.data[0].embedding
            return np.array(data, dtype=np.float32)

        np.random.seed(hash(image_path_obj.name) % 2 ** 32)
        dim = self.config.image_dimension or self.config.dimension
        return np.random.randn(dim).astype(np.float32)

    def build_index(self, catalog: Any, batch_size: int = 32, thumbnail_root: Optional[str] = None) -> None:
        """
        Build embedding index for catalog assets.

        Args:
            catalog: AssetCatalog to index
            batch_size: Batch size for embedding generation
        """
        self._load_model()

        texts = []
        self.asset_ids = []
        self.thumbnail_asset_ids = []
        self.thumbnail_embeddings = {}
        self.thumbnail_index_matrix = None

        vector_records: list[VectorRecord] = []
        thumbnail_records: list[VectorRecord] = []

        for asset in catalog.assets:
            text_parts = [
                asset.display_name or "",
                getattr(asset, "description", "") or "",
                asset.category,
                asset.subcategory or "",
                " ".join(asset.tags),
            ]
            text = " ".join(filter(None, text_parts))
            texts.append(text)
            self.asset_ids.append(asset.asset_id)

        all_embeddings: list[np.ndarray] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            batch_embeddings = [self.embed_text(t) for t in batch]
            all_embeddings.extend(batch_embeddings)

        if all_embeddings:
            self.config.dimension = int(all_embeddings[0].shape[-1])

        self.index_matrix = np.vstack(all_embeddings) if all_embeddings else None

        for asset_id, emb in zip(self.asset_ids, all_embeddings):
            self.embeddings[asset_id] = emb
            vector_records.append(
                VectorRecord(
                    id=self._vector_id(asset_id, "text"),
                    embedding=emb,
                    metadata={"asset_id": asset_id, "kind": "text", "collection": self.collection},
                )
            )

        thumb_root_path = Path(thumbnail_root) if thumbnail_root else None
        thumb_vectors: list[np.ndarray] = []
        for asset, asset_id in zip(catalog.assets, self.asset_ids):
            if not asset.thumbnail_path:
                continue

            thumb_path = Path(asset.thumbnail_path)
            if thumb_root_path and not thumb_path.is_absolute():
                thumb_path = thumb_root_path / asset.thumbnail_path

            if not thumb_path.exists():
                continue

            thumb_emb = self.embed_image(str(thumb_path))
            if thumb_emb is None:
                continue

            if self.config.image_dimension is None:
                self.config.image_dimension = int(thumb_emb.shape[-1])

            self.thumbnail_embeddings[asset_id] = thumb_emb
            self.thumbnail_asset_ids.append(asset_id)
            thumb_vectors.append(thumb_emb)
            thumbnail_records.append(
                VectorRecord(
                    id=self._vector_id(asset_id, "thumbnail"),
                    embedding=thumb_emb,
                    metadata={"asset_id": asset_id, "kind": "thumbnail", "collection": self.collection},
                )
            )

        if thumb_vectors:
            self.thumbnail_index_matrix = np.vstack(thumb_vectors)

        if self.vector_store and vector_records:
            self.vector_store.upsert(vector_records, namespace=self.collection)
        if self.vector_store and thumbnail_records:
            self.vector_store.upsert(thumbnail_records, namespace=self.collection)

    def search(self, query: str, top_k: int = 10, threshold: float = 0.0) -> list[tuple[str, float]]:
        """
        Search for assets by semantic similarity.

        Args:
            query: Search query text
            top_k: Number of results to return
            threshold: Minimum similarity threshold

        Returns:
            List of (asset_id, similarity_score) tuples
        """
        query_emb = self.embed_text(query)

        if self.vector_store:
            matches = self.vector_store.query(query_emb, top_k=top_k, namespace=self.collection, filter_metadata={"kind": "text"})
            return [
                (rec.metadata.get("asset_id", rec.id.split(":" + "text")[0]), float(rec.score or 0.0))
                for rec in matches
                if (rec.score or 0.0) >= threshold
            ]

        if self.index_matrix is None or len(self.asset_ids) == 0:
            return []

        query_norm = query_emb / (np.linalg.norm(query_emb) + 1e-8)
        index_norms = self.index_matrix / (np.linalg.norm(self.index_matrix, axis=1, keepdims=True) + 1e-8)
        similarities = np.dot(index_norms, query_norm)

        top_indices = np.argsort(similarities)[::-1][:top_k]

        results = []
        for idx in top_indices:
            score = float(similarities[idx])
            if score >= threshold:
                results.append((self.asset_ids[idx], score))

        return results

    def search_by_image(self, image_path: str, top_k: int = 10, threshold: float = 0.0, thumbnail_root: Optional[str] = None) -> list[tuple[str, float]]:
        """Search for assets by thumbnail similarity."""
        thumb_path = Path(image_path)
        if thumbnail_root:
            thumb_path = Path(thumbnail_root) / image_path

        query_emb = self.embed_image(str(thumb_path))

        if self.vector_store:
            matches = self.vector_store.query(query_emb, top_k=top_k, namespace=self.collection, filter_metadata={"kind": "thumbnail"})
            return [
                (rec.metadata.get("asset_id", rec.id.split(":" + "thumbnail")[0]), float(rec.score or 0.0))
                for rec in matches
                if (rec.score or 0.0) >= threshold
            ]

        if self.thumbnail_index_matrix is None or not self.thumbnail_asset_ids:
            return []

        query_norm = query_emb / (np.linalg.norm(query_emb) + 1e-8)
        index_norms = self.thumbnail_index_matrix / (np.linalg.norm(self.thumbnail_index_matrix, axis=1, keepdims=True) + 1e-8)
        similarities = np.dot(index_norms, query_norm)

        top_indices = np.argsort(similarities)[::-1][:top_k]

        results = []
        for idx in top_indices:
            score = float(similarities[idx])
            if score >= threshold:
                results.append((self.thumbnail_asset_ids[idx], score))

        return results

    def _vector_id(self, asset_id: str, kind: str) -> str:
        return f"{asset_id}:{kind}"

    def get_embedding(self, asset_id: str, kind: str = "text") -> Optional[np.ndarray]:
        """Fetch an embedding for a single asset, pulling from the vector store when necessary."""

        cache: Dict[str, np.ndarray] = self.embeddings if kind == "text" else self.thumbnail_embeddings
        if asset_id in cache:
            return cache[asset_id]

        if not self.vector_store:
            return None

        records = self.vector_store.fetch([self._vector_id(asset_id, kind)], namespace=self.collection)
        if not records:
            return None

        embedding = records[0].embedding
        cache[asset_id] = embedding
        return embedding

    def load_from_vector_store(self, include_thumbnails: bool = True) -> None:
        """Populate local caches and matrices from a configured vector store."""

        if not self.vector_store:
            raise ValueError("Vector store is not configured")

        text_records = self.vector_store.list(namespace=self.collection, filter_metadata={"kind": "text"})
        self.asset_ids = []
        self.embeddings = {}

        for rec in text_records:
            asset_id = rec.metadata.get("asset_id", rec.id.split(":" + "text")[0])
            self.asset_ids.append(asset_id)
            self.embeddings[asset_id] = rec.embedding
            if self.config.dimension is None:
                self.config.dimension = int(rec.embedding.shape[-1])

        if self.asset_ids:
            self.index_matrix = np.vstack([self.embeddings[aid] for aid in self.asset_ids])

        if not include_thumbnails:
            return

        thumbnail_records = self.vector_store.list(namespace=self.collection, filter_metadata={"kind": "thumbnail"})
        self.thumbnail_asset_ids = []
        self.thumbnail_embeddings = {}

        for rec in thumbnail_records:
            asset_id = rec.metadata.get("asset_id", rec.id.split(":" + "thumbnail")[0])
            self.thumbnail_asset_ids.append(asset_id)
            self.thumbnail_embeddings[asset_id] = rec.embedding
            if self.config.image_dimension is None:
                self.config.image_dimension = int(rec.embedding.shape[-1])

        if self.thumbnail_asset_ids:
            self.thumbnail_index_matrix = np.vstack(
                [self.thumbnail_embeddings[aid] for aid in self.thumbnail_asset_ids if aid in self.thumbnail_embeddings]
            )

    def save(self, path: str) -> None:
        """Save embeddings to file."""
        data = {
            "config": {
                "model_name": self.config.model_name,
                "dimension": self.config.dimension,
                "backend": self.config.backend,
                "image_model_name": self.config.image_model_name,
                "image_dimension": self.config.image_dimension,
                "image_backend": self.config.image_backend,
            },
            "asset_ids": self.asset_ids,
            "embeddings": {k: v.tolist() for k, v in self.embeddings.items()},
            "thumbnail_embeddings": {k: v.tolist() for k, v in self.thumbnail_embeddings.items()},
            "thumbnail_asset_ids": self.thumbnail_asset_ids,
        }
        with open(path, "w") as f:
            json.dump(data, f)

    def load(self, path: str) -> None:
        """Load embeddings from file."""
        with open(path) as f:
            data = json.load(f)

        self.config = EmbeddingConfig(**data["config"])
        self.asset_ids = data["asset_ids"]
        self.embeddings = {k: np.array(v, dtype=np.float32) for k, v in data["embeddings"].items()}
        self.thumbnail_embeddings = {k: np.array(v, dtype=np.float32) for k, v in data.get("thumbnail_embeddings", {}).items()}
        self.thumbnail_asset_ids = data.get("thumbnail_asset_ids", list(self.thumbnail_embeddings.keys()))

        # Rebuild index matrix
        if self.asset_ids:
            self.index_matrix = np.vstack([self.embeddings[aid] for aid in self.asset_ids if aid in self.embeddings])
        if self.thumbnail_asset_ids:
            self.thumbnail_index_matrix = np.vstack(
                [self.thumbnail_embeddings[aid] for aid in self.thumbnail_asset_ids if aid in self.thumbnail_embeddings]
            )
