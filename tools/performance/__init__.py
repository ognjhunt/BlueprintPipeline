"""Performance optimization utilities."""

from .streaming_json import (
    StreamingJSONError,
    stream_json_array,
    stream_manifest_objects,
    StreamingManifestParser,
    process_large_manifest,
)
from .parallel_processing import (
    ParallelResult,
    process_parallel_threaded,
    process_parallel_multiprocess,
    process_in_batches,
    ParallelProcessor,
)

__all__ = [
    # Streaming JSON
    "StreamingJSONError",
    "stream_json_array",
    "stream_manifest_objects",
    "StreamingManifestParser",
    "process_large_manifest",
    # Parallel processing
    "ParallelResult",
    "process_parallel_threaded",
    "process_parallel_multiprocess",
    "process_in_batches",
    "ParallelProcessor",
]
