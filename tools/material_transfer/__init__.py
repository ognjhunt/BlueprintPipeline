"""Material and texture transfer utilities."""

from .material_transfer import (
    MaterialType,
    TextureInfo,
    PBRMaterial,
    extract_materials_from_glb,
    apply_materials_to_usd,
    create_material_manifest,
    MaterialTransferPipeline,
    transfer_materials,
    infer_material_type,
    MATERIAL_PHYSICS,
)

__all__ = [
    "MaterialType",
    "TextureInfo",
    "PBRMaterial",
    "extract_materials_from_glb",
    "apply_materials_to_usd",
    "create_material_manifest",
    "MaterialTransferPipeline",
    "transfer_materials",
    "infer_material_type",
    "MATERIAL_PHYSICS",
]
