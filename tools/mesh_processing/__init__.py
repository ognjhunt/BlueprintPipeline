"""
Mesh Processing Tools for BlueprintPipeline.

Provides:
- LOD (Level of Detail) generation for multi-resolution meshes
- Texture compression for GPU-friendly formats
- Mesh optimization and simplification
"""

from tools.mesh_processing.lod_generator import (
    LODConfig,
    LODLevel,
    LODResult,
    generate_lod_chain,
    apply_lod_to_usd,
    LODGenerator,
)

from tools.mesh_processing.texture_compression import (
    CompressionFormat,
    CompressionResult,
    compress_texture,
    compress_textures_batch,
    TextureCompressor,
)

__all__ = [
    # LOD Generation
    "LODConfig",
    "LODLevel",
    "LODResult",
    "generate_lod_chain",
    "apply_lod_to_usd",
    "LODGenerator",
    # Texture Compression
    "CompressionFormat",
    "CompressionResult",
    "compress_texture",
    "compress_textures_batch",
    "TextureCompressor",
]
