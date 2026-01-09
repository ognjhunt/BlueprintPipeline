"""
Streaming JSON parser for large manifests.

Provides memory-efficient parsing of large JSON files by processing
objects incrementally rather than loading the entire file into memory.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Union

logger = logging.getLogger(__name__)


class StreamingJSONError(Exception):
    """Raised when streaming JSON parsing fails."""
    pass


def stream_json_array(
    file_path: Union[str, Path],
    array_path: str = "objects",
    batch_size: int = 100,
) -> Generator[List[Dict[str, Any]], None, None]:
    """
    Stream parse a JSON array without loading entire file into memory.

    Yields batches of items from the specified array path.

    Args:
        file_path: Path to JSON file
        array_path: JSON path to array (e.g., "objects", "items.data")
        batch_size: Number of items per batch

    Yields:
        Batches of parsed objects

    Raises:
        StreamingJSONError: If parsing fails

    Example:
        # Process objects in batches of 100
        for batch in stream_json_array("scene_manifest.json", "objects", 100):
            for obj in batch:
                process_object(obj)
    """
    try:
        import ijson
    except ImportError:
        logger.error(
            "ijson not installed. Install with: pip install ijson\n"
            "Falling back to standard json.load (may use more memory)"
        )
        # Fallback to standard parsing
        yield from _fallback_json_array(file_path, array_path, batch_size)
        return

    try:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        batch = []

        with open(path, 'rb') as f:
            # Parse array items one at a time
            items = ijson.items(f, f'{array_path}.item')

            for item in items:
                batch.append(item)

                if len(batch) >= batch_size:
                    yield batch
                    batch = []

            # Yield remaining items
            if batch:
                yield batch

    except Exception as e:
        raise StreamingJSONError(
            f"Failed to stream parse JSON array at '{array_path}': {e}"
        ) from e


def _fallback_json_array(
    file_path: Union[str, Path],
    array_path: str,
    batch_size: int,
) -> Generator[List[Dict[str, Any]], None, None]:
    """Fallback to standard JSON parsing when ijson not available."""
    path = Path(file_path)

    with open(path) as f:
        data = json.load(f)

    # Navigate to array path
    obj = data
    for key in array_path.split('.'):
        obj = obj[key]

    if not isinstance(obj, list):
        raise StreamingJSONError(
            f"Path '{array_path}' does not point to an array"
        )

    # Yield in batches
    for i in range(0, len(obj), batch_size):
        yield obj[i:i + batch_size]


def stream_manifest_objects(
    manifest_path: Union[str, Path],
    batch_size: int = 100,
) -> Generator[List[Dict[str, Any]], None, None]:
    """
    Stream parse objects from a scene manifest.

    Args:
        manifest_path: Path to scene_manifest.json
        batch_size: Number of objects per batch

    Yields:
        Batches of object dictionaries

    Example:
        total_objects = 0
        for batch in stream_manifest_objects("scene_manifest.json"):
            total_objects += len(batch)
            for obj in batch:
                print(f"Processing object: {obj['id']}")
    """
    yield from stream_json_array(manifest_path, "objects", batch_size)


class StreamingManifestParser:
    """
    Streaming parser for scene manifests.

    Provides utilities for processing large manifests without loading
    everything into memory.

    Example:
        parser = StreamingManifestParser("scene_manifest.json")

        # Get metadata without loading objects
        print(f"Scene ID: {parser.get_scene_id()}")
        print(f"Version: {parser.get_version()}")

        # Process objects in batches
        for batch in parser.stream_objects(batch_size=50):
            process_batch(batch)
    """

    def __init__(self, manifest_path: Union[str, Path]):
        self.manifest_path = Path(manifest_path)
        self._metadata: Optional[Dict[str, Any]] = None

    def _ensure_metadata(self) -> None:
        """Load manifest metadata (everything except objects array)."""
        if self._metadata is not None:
            return

        try:
            import ijson
            use_streaming = True
        except ImportError:
            use_streaming = False

        if use_streaming:
            self._metadata = {}
            with open(self.manifest_path, 'rb') as f:
                # Parse top-level fields (skip objects array)
                parser = ijson.parse(f)
                current_path = []

                for prefix, event, value in parser:
                    # Track current path
                    if event == 'map_key':
                        if current_path and current_path[-1] == 'map':
                            current_path.pop()
                        current_path.append(value)
                    elif event == 'start_map':
                        current_path.append('map')
                    elif event == 'end_map':
                        if current_path:
                            current_path.pop()
                    elif event == 'start_array':
                        # Skip objects array
                        if current_path and current_path[-1] == 'objects':
                            break
                        current_path.append('array')
                    elif event in ('string', 'number', 'boolean', 'null'):
                        # Store value
                        if current_path and current_path[-1] != 'objects':
                            key = '.'.join(str(p) for p in current_path if p != 'map' and p != 'array')
                            self._metadata[key] = value
        else:
            # Fallback: load entire manifest but only use metadata
            with open(self.manifest_path) as f:
                data = json.load(f)
                self._metadata = {k: v for k, v in data.items() if k != 'objects'}

    def get_version(self) -> str:
        """Get manifest version."""
        self._ensure_metadata()
        return self._metadata.get('version', 'unknown')

    def get_scene_id(self) -> str:
        """Get scene ID."""
        self._ensure_metadata()
        return self._metadata.get('scene_id', 'unknown')

    def get_metadata(self) -> Dict[str, Any]:
        """Get all metadata (excluding objects)."""
        self._ensure_metadata()
        return dict(self._metadata)

    def stream_objects(
        self,
        batch_size: int = 100,
    ) -> Generator[List[Dict[str, Any]], None, None]:
        """
        Stream objects from manifest in batches.

        Args:
            batch_size: Number of objects per batch

        Yields:
            Batches of object dictionaries
        """
        yield from stream_manifest_objects(self.manifest_path, batch_size)

    def count_objects(self) -> int:
        """Count total number of objects (requires full scan)."""
        count = 0
        for batch in self.stream_objects():
            count += len(batch)
        return count

    def filter_objects(
        self,
        predicate: callable,
        batch_size: int = 100,
    ) -> Generator[Dict[str, Any], None, None]:
        """
        Stream objects that match a predicate.

        Args:
            predicate: Function that takes an object dict and returns bool
            batch_size: Batch size for streaming

        Yields:
            Objects that match predicate

        Example:
            # Find all manipulable objects
            parser = StreamingManifestParser("manifest.json")
            for obj in parser.filter_objects(
                lambda o: o.get("sim_role") == "manipulable_object"
            ):
                print(f"Found: {obj['id']}")
        """
        for batch in self.stream_objects(batch_size):
            for obj in batch:
                if predicate(obj):
                    yield obj

    def map_objects(
        self,
        mapper: callable,
        batch_size: int = 100,
    ) -> Generator[Any, None, None]:
        """
        Apply a mapping function to all objects.

        Args:
            mapper: Function that transforms an object
            batch_size: Batch size for streaming

        Yields:
            Mapped objects

        Example:
            # Extract object IDs
            parser = StreamingManifestParser("manifest.json")
            object_ids = list(parser.map_objects(lambda o: o['id']))
        """
        for batch in self.stream_objects(batch_size):
            for obj in batch:
                yield mapper(obj)


def process_large_manifest(
    manifest_path: Union[str, Path],
    processor: callable,
    batch_size: int = 100,
    max_objects: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Process a large manifest file with a custom processor function.

    Args:
        manifest_path: Path to manifest file
        processor: Function that takes a batch of objects and returns results
        batch_size: Number of objects per batch
        max_objects: Maximum number of objects to process (None = all)

    Returns:
        Dictionary with processing results

    Example:
        def count_by_category(batch):
            counts = {}
            for obj in batch:
                cat = obj.get("category", "unknown")
                counts[cat] = counts.get(cat, 0) + 1
            return counts

        results = process_large_manifest(
            "scene_manifest.json",
            count_by_category,
            batch_size=100,
        )
    """
    parser = StreamingManifestParser(manifest_path)

    logger.info(
        f"Processing manifest: {manifest_path} "
        f"(scene_id: {parser.get_scene_id()}, version: {parser.get_version()})"
    )

    all_results = []
    processed_count = 0

    for batch in parser.stream_objects(batch_size):
        # Check if we've reached max_objects
        if max_objects and processed_count >= max_objects:
            break

        # Trim batch if needed
        if max_objects:
            remaining = max_objects - processed_count
            batch = batch[:remaining]

        # Process batch
        try:
            result = processor(batch)
            all_results.append(result)
            processed_count += len(batch)

            logger.debug(
                f"Processed batch: {len(batch)} objects "
                f"(total: {processed_count})"
            )

        except Exception as e:
            logger.error(f"Error processing batch: {e}")
            raise

    logger.info(f"Processed {processed_count} objects")

    return {
        "processed_count": processed_count,
        "batch_results": all_results,
        "scene_id": parser.get_scene_id(),
        "version": parser.get_version(),
    }
