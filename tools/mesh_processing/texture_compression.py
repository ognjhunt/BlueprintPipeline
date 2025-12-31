"""
Texture Compression for GPU-Friendly Formats.

Compresses textures to GPU-native formats for efficient rendering:
- BC/DXT (Desktop GPUs - DirectX/Vulkan)
- ASTC (Mobile GPUs - Android/iOS)
- ETC2 (OpenGL ES 3.0+)
- KTX2 container format with basis universal compression

This is critical for:
- Reduced GPU memory usage (4-8x smaller than uncompressed)
- Faster texture loading and streaming
- Better runtime performance
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


class CompressionFormat(str, Enum):
    """Supported texture compression formats."""
    # Desktop (BC/DXT family)
    BC1 = "bc1"  # DXT1 - RGB, 4bpp, no alpha
    BC3 = "bc3"  # DXT5 - RGBA, 8bpp, interpolated alpha
    BC4 = "bc4"  # Single channel, 4bpp (normals, height)
    BC5 = "bc5"  # Two channel, 8bpp (normal maps XY)
    BC7 = "bc7"  # High quality RGBA, 8bpp

    # Mobile (ASTC)
    ASTC_4x4 = "astc_4x4"  # Highest quality, 8bpp
    ASTC_6x6 = "astc_6x6"  # Medium quality, 3.56bpp
    ASTC_8x8 = "astc_8x8"  # Lower quality, 2bpp

    # OpenGL ES (ETC)
    ETC2_RGB = "etc2_rgb"  # RGB, 4bpp
    ETC2_RGBA = "etc2_rgba"  # RGBA, 8bpp

    # Universal (Basis)
    BASIS_UASTC = "basis_uastc"  # High quality universal
    BASIS_ETC1S = "basis_etc1s"  # Smaller size universal

    # Keep original
    NONE = "none"


@dataclass
class CompressionResult:
    """Result of texture compression."""
    source_path: Path
    output_path: Optional[Path]
    format: CompressionFormat
    original_size: int  # bytes
    compressed_size: int  # bytes
    compression_ratio: float
    success: bool
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": str(self.source_path),
            "output": str(self.output_path) if self.output_path else None,
            "format": self.format.value,
            "original_size": self.original_size,
            "compressed_size": self.compressed_size,
            "compression_ratio": self.compression_ratio,
            "success": self.success,
            "error": self.error,
        }


def get_recommended_format(
    texture_path: Path,
    texture_type: str = "color",
    target_platform: str = "desktop",
) -> CompressionFormat:
    """
    Get recommended compression format based on texture type and platform.

    Args:
        texture_path: Path to texture
        texture_type: "color", "normal", "metallic_roughness", "occlusion", "emissive"
        target_platform: "desktop", "mobile", "universal"

    Returns:
        Recommended CompressionFormat
    """
    # Check if texture has alpha
    has_alpha = _texture_has_alpha(texture_path)

    if target_platform == "desktop":
        if texture_type == "normal":
            return CompressionFormat.BC5  # Best for normal maps
        elif texture_type in ("metallic_roughness", "occlusion"):
            return CompressionFormat.BC4 if not has_alpha else CompressionFormat.BC3
        else:
            return CompressionFormat.BC7 if has_alpha else CompressionFormat.BC1

    elif target_platform == "mobile":
        if texture_type == "normal":
            return CompressionFormat.ASTC_4x4
        else:
            return CompressionFormat.ASTC_6x6

    else:  # universal
        return CompressionFormat.BASIS_UASTC


def _texture_has_alpha(texture_path: Path) -> bool:
    """Check if texture has an alpha channel with non-trivial values."""
    try:
        from PIL import Image
        img = Image.open(texture_path)
        if img.mode == "RGBA":
            # Check if alpha channel is actually used
            alpha = img.split()[-1]
            extrema = alpha.getextrema()
            return extrema[0] < 255  # If min alpha < 255, alpha is used
        return False
    except Exception:
        # If we can't check, assume no alpha
        return texture_path.suffix.lower() == ".png"


def compress_texture(
    texture_path: Path,
    output_path: Path,
    format: CompressionFormat,
    quality: str = "high",
) -> CompressionResult:
    """
    Compress a texture to the specified format.

    Args:
        texture_path: Input texture path
        output_path: Output texture path
        format: Target compression format
        quality: "low", "medium", "high"

    Returns:
        CompressionResult with compression details
    """
    texture_path = Path(texture_path)
    output_path = Path(output_path)

    original_size = texture_path.stat().st_size if texture_path.is_file() else 0

    result = CompressionResult(
        source_path=texture_path,
        output_path=None,
        format=format,
        original_size=original_size,
        compressed_size=0,
        compression_ratio=1.0,
        success=False,
    )

    if format == CompressionFormat.NONE:
        # Just copy the file
        output_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(texture_path, output_path)
        result.output_path = output_path
        result.compressed_size = output_path.stat().st_size
        result.compression_ratio = 1.0
        result.success = True
        return result

    # Try different compression tools
    success = False
    error = None

    # Try using external tools first (better quality)
    if not success and _tool_available("texconv"):
        success, error = _compress_with_texconv(
            texture_path, output_path, format, quality
        )

    if not success and _tool_available("astcenc"):
        success, error = _compress_with_astcenc(
            texture_path, output_path, format, quality
        )

    if not success and _tool_available("basisu"):
        success, error = _compress_with_basisu(
            texture_path, output_path, format, quality
        )

    # Fallback to PIL-based compression (limited formats)
    if not success:
        success, error = _compress_with_pil(
            texture_path, output_path, format, quality
        )

    if success and output_path.is_file():
        result.output_path = output_path
        result.compressed_size = output_path.stat().st_size
        result.compression_ratio = original_size / max(1, result.compressed_size)
        result.success = True
        print(f"[COMPRESS] {texture_path.name} -> {format.value}: "
              f"{original_size/1024:.1f}KB -> {result.compressed_size/1024:.1f}KB "
              f"({result.compression_ratio:.1f}x)")
    else:
        result.error = error or "Compression failed"
        # Copy original as fallback
        output_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(texture_path, output_path)
        result.output_path = output_path
        result.compressed_size = output_path.stat().st_size
        result.compression_ratio = 1.0
        print(f"[COMPRESS] {texture_path.name}: fallback to original ({result.error})")

    return result


def _tool_available(tool_name: str) -> bool:
    """Check if an external tool is available."""
    return shutil.which(tool_name) is not None


def _compress_with_texconv(
    input_path: Path,
    output_path: Path,
    format: CompressionFormat,
    quality: str,
) -> Tuple[bool, Optional[str]]:
    """Compress using DirectXTex texconv tool."""
    format_map = {
        CompressionFormat.BC1: "BC1_UNORM",
        CompressionFormat.BC3: "BC3_UNORM",
        CompressionFormat.BC4: "BC4_UNORM",
        CompressionFormat.BC5: "BC5_UNORM",
        CompressionFormat.BC7: "BC7_UNORM",
    }

    if format not in format_map:
        return False, f"texconv doesn't support {format.value}"

    output_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "texconv",
        "-f", format_map[format],
        "-o", str(output_path.parent),
        "-y",  # Overwrite
        str(input_path),
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if result.returncode == 0:
            # Rename output to desired name
            expected_output = output_path.parent / f"{input_path.stem}.dds"
            if expected_output.is_file():
                expected_output.rename(output_path)
                return True, None
        return False, result.stderr
    except Exception as e:
        return False, str(e)


def _compress_with_astcenc(
    input_path: Path,
    output_path: Path,
    format: CompressionFormat,
    quality: str,
) -> Tuple[bool, Optional[str]]:
    """Compress using ARM ASTC encoder."""
    block_size_map = {
        CompressionFormat.ASTC_4x4: "4x4",
        CompressionFormat.ASTC_6x6: "6x6",
        CompressionFormat.ASTC_8x8: "8x8",
    }

    if format not in block_size_map:
        return False, f"astcenc doesn't support {format.value}"

    quality_map = {"low": "-fast", "medium": "-medium", "high": "-thorough"}

    output_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "astcenc",
        "-cl",  # Compress LDR
        str(input_path),
        str(output_path),
        block_size_map[format],
        quality_map.get(quality, "-medium"),
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        return result.returncode == 0, result.stderr if result.returncode != 0 else None
    except Exception as e:
        return False, str(e)


def _compress_with_basisu(
    input_path: Path,
    output_path: Path,
    format: CompressionFormat,
    quality: str,
) -> Tuple[bool, Optional[str]]:
    """Compress using Basis Universal encoder."""
    if format not in (CompressionFormat.BASIS_UASTC, CompressionFormat.BASIS_ETC1S):
        return False, f"basisu doesn't support {format.value}"

    quality_map = {"low": "128", "medium": "192", "high": "255"}

    output_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = ["basisu", str(input_path), "-output_file", str(output_path)]

    if format == CompressionFormat.BASIS_UASTC:
        cmd.extend(["-uastc", "-uastc_level", "2"])
    else:
        cmd.extend(["-comp_level", "2"])

    cmd.extend(["-quality", quality_map.get(quality, "192")])

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        return result.returncode == 0, result.stderr if result.returncode != 0 else None
    except Exception as e:
        return False, str(e)


def _compress_with_pil(
    input_path: Path,
    output_path: Path,
    format: CompressionFormat,
    quality: str,
) -> Tuple[bool, Optional[str]]:
    """
    Fallback compression using PIL.

    Note: PIL can't create BC/ASTC/ETC formats directly, so this
    falls back to optimized PNG/JPEG for those cases.
    """
    try:
        from PIL import Image
    except ImportError:
        return False, "PIL not available"

    try:
        img = Image.open(input_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # For GPU formats we can't create, output optimized PNG
        if format in (CompressionFormat.BC1, CompressionFormat.BC3,
                      CompressionFormat.BC4, CompressionFormat.BC5,
                      CompressionFormat.BC7, CompressionFormat.ASTC_4x4,
                      CompressionFormat.ASTC_6x6, CompressionFormat.ASTC_8x8,
                      CompressionFormat.ETC2_RGB, CompressionFormat.ETC2_RGBA):
            # Save as optimized PNG (better than nothing)
            output_path = output_path.with_suffix(".png")
            img.save(output_path, "PNG", optimize=True)
            return True, None

        # For Basis formats, save as high-quality JPEG
        if format in (CompressionFormat.BASIS_UASTC, CompressionFormat.BASIS_ETC1S):
            if img.mode == "RGBA":
                img = img.convert("RGB")
            output_path = output_path.with_suffix(".jpg")
            quality_val = {"low": 70, "medium": 85, "high": 95}.get(quality, 85)
            img.save(output_path, "JPEG", quality=quality_val, optimize=True)
            return True, None

        return False, f"Unsupported format: {format.value}"

    except Exception as e:
        return False, str(e)


def compress_textures_batch(
    texture_paths: List[Path],
    output_dir: Path,
    format: Optional[CompressionFormat] = None,
    target_platform: str = "desktop",
    quality: str = "high",
) -> List[CompressionResult]:
    """
    Compress multiple textures.

    Args:
        texture_paths: List of input texture paths
        output_dir: Output directory for compressed textures
        format: Compression format (auto-detected if None)
        target_platform: "desktop", "mobile", "universal"
        quality: "low", "medium", "high"

    Returns:
        List of CompressionResult
    """
    results = []

    for i, tex_path in enumerate(texture_paths):
        print(f"[COMPRESS] Processing {i+1}/{len(texture_paths)}: {tex_path.name}")

        # Determine format if not specified
        tex_format = format
        if tex_format is None:
            # Guess texture type from filename
            name_lower = tex_path.stem.lower()
            if "normal" in name_lower:
                tex_type = "normal"
            elif "metallic" in name_lower or "roughness" in name_lower:
                tex_type = "metallic_roughness"
            elif "occlusion" in name_lower or "ao" in name_lower:
                tex_type = "occlusion"
            elif "emissive" in name_lower:
                tex_type = "emissive"
            else:
                tex_type = "color"

            tex_format = get_recommended_format(tex_path, tex_type, target_platform)

        # Determine output path
        output_suffix = _get_format_extension(tex_format)
        output_path = output_dir / f"{tex_path.stem}{output_suffix}"

        result = compress_texture(tex_path, output_path, tex_format, quality)
        results.append(result)

    return results


def _get_format_extension(format: CompressionFormat) -> str:
    """Get file extension for compression format."""
    if format in (CompressionFormat.BC1, CompressionFormat.BC3,
                  CompressionFormat.BC4, CompressionFormat.BC5,
                  CompressionFormat.BC7):
        return ".dds"
    elif format in (CompressionFormat.ASTC_4x4, CompressionFormat.ASTC_6x6,
                    CompressionFormat.ASTC_8x8):
        return ".astc"
    elif format in (CompressionFormat.ETC2_RGB, CompressionFormat.ETC2_RGBA):
        return ".ktx"
    elif format in (CompressionFormat.BASIS_UASTC, CompressionFormat.BASIS_ETC1S):
        return ".ktx2"
    else:
        return ".png"


class TextureCompressor:
    """
    Texture compression pipeline for batch processing.

    Usage:
        compressor = TextureCompressor(output_dir, target_platform="desktop")
        results = compressor.compress_all(texture_paths)
    """

    def __init__(
        self,
        output_dir: Path,
        target_platform: str = "desktop",
        quality: str = "high",
    ):
        self.output_dir = Path(output_dir)
        self.target_platform = target_platform
        self.quality = quality
        self.results: List[CompressionResult] = []

    def compress_texture(
        self,
        texture_path: Path,
        texture_type: str = "color",
    ) -> CompressionResult:
        """Compress a single texture with auto-detected format."""
        format = get_recommended_format(
            texture_path, texture_type, self.target_platform
        )
        output_suffix = _get_format_extension(format)
        output_path = self.output_dir / f"{texture_path.stem}{output_suffix}"

        result = compress_texture(texture_path, output_path, format, self.quality)
        self.results.append(result)
        return result

    def compress_all(
        self,
        texture_paths: List[Path],
        format: Optional[CompressionFormat] = None,
    ) -> List[CompressionResult]:
        """Compress all textures."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        results = compress_textures_batch(
            texture_paths,
            self.output_dir,
            format=format,
            target_platform=self.target_platform,
            quality=self.quality,
        )
        self.results.extend(results)
        return results

    def save_summary(self, summary_path: Optional[Path] = None) -> Path:
        """Save compression summary."""
        if summary_path is None:
            summary_path = self.output_dir / "compression_summary.json"

        total_original = sum(r.original_size for r in self.results)
        total_compressed = sum(r.compressed_size for r in self.results)

        summary = {
            "total_textures": len(self.results),
            "successful": sum(1 for r in self.results if r.success),
            "failed": sum(1 for r in self.results if not r.success),
            "total_original_size_mb": total_original / (1024 * 1024),
            "total_compressed_size_mb": total_compressed / (1024 * 1024),
            "overall_compression_ratio": total_original / max(1, total_compressed),
            "target_platform": self.target_platform,
            "quality": self.quality,
            "results": [r.to_dict() for r in self.results],
        }

        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, indent=2))
        return summary_path
