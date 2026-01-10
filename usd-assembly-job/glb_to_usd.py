#!/usr/bin/env python3
"""
glb_to_usd.py - Convert GLB files to USD/USDZ format.

Uses pygltflib to parse GLB and pxr (OpenUSD) to create USD output.
This is a clean, modern implementation that avoids the unmaintained
kcoley/gltf2usd library and its Python 2 compatibility issues.

Supports:
  - Triangle meshes with positions, normals, UVs
  - Materials with baseColorFactor and baseColorTexture
  - Embedded textures (extracted to disk for USDZ packaging)
  - Node hierarchy with transforms

Usage:
  python glb_to_usd.py input.glb output.usdz
  python glb_to_usd.py input.glb output.usda  # For debugging
"""

from __future__ import annotations

import argparse
import base64
import os
import shutil
import struct
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    from pygltflib import GLTF2
except ImportError:
    print("ERROR: pygltflib is required. Install with: pip install pygltflib")
    sys.exit(1)

try:
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade, UsdUtils, Vt
except ImportError:
    print("ERROR: usd-core is required. Install with: pip install usd-core")
    sys.exit(1)


# -----------------------------------------------------------------------------
# GLB/GLTF Data Extraction
# -----------------------------------------------------------------------------


class GLBReader:
    """
    Reads and decodes data from a GLTF2 object loaded via pygltflib.
    """

    def __init__(self, gltf: GLTF2, gltf_path: Path):
        self.gltf = gltf
        self.gltf_path = gltf_path
        self.gltf_dir = gltf_path.parent
        self._buffer_cache: Dict[int, bytes] = {}

    def get_buffer_data(self, buffer_index: int) -> bytes:
        """Load and cache buffer data."""
        if buffer_index in self._buffer_cache:
            return self._buffer_cache[buffer_index]

        buffer = self.gltf.buffers[buffer_index]

        # Check if data is embedded in the GLTF2 object (GLB binary chunk)
        if hasattr(self.gltf, "_glb_data") and self.gltf._glb_data:
            data = self.gltf._glb_data
        elif buffer.uri:
            if buffer.uri.startswith("data:"):
                # Base64 data URI
                _, encoded = buffer.uri.split(",", 1)
                data = base64.b64decode(encoded)
            else:
                # External file
                buffer_path = self.gltf_dir / buffer.uri
                data = buffer_path.read_bytes()
        else:
            # Try to get from pygltflib's internal buffer
            # pygltflib stores binary data in a special way for GLB files
            blob = self.gltf.binary_blob()
            if blob:
                data = blob
            else:
                raise ValueError(f"Cannot load buffer {buffer_index}: no data source")

        self._buffer_cache[buffer_index] = data
        return data

    def get_accessor_data(self, accessor_index: int) -> np.ndarray:
        """
        Extract typed array data from an accessor.
        Returns a numpy array with the appropriate shape.
        """
        accessor = self.gltf.accessors[accessor_index]
        buffer_view = self.gltf.bufferViews[accessor.bufferView]
        buffer_data = self.get_buffer_data(buffer_view.buffer)

        # Calculate byte offsets
        byte_offset = (buffer_view.byteOffset or 0) + (accessor.byteOffset or 0)

        # Component type to numpy dtype
        component_types = {
            5120: np.int8,
            5121: np.uint8,
            5122: np.int16,
            5123: np.uint16,
            5125: np.uint32,
            5126: np.float32,
        }
        dtype = component_types.get(accessor.componentType, np.float32)

        # Type to component count
        type_counts = {
            "SCALAR": 1,
            "VEC2": 2,
            "VEC3": 3,
            "VEC4": 4,
            "MAT2": 4,
            "MAT3": 9,
            "MAT4": 16,
        }
        num_components = type_counts.get(accessor.type, 1)

        # Handle byte stride
        byte_stride = buffer_view.byteStride
        element_size = np.dtype(dtype).itemsize * num_components

        if byte_stride and byte_stride != element_size:
            # Strided access - need to read element by element
            result = np.zeros((accessor.count, num_components), dtype=dtype)
            for i in range(accessor.count):
                start = byte_offset + i * byte_stride
                end = start + element_size
                chunk = buffer_data[start:end]
                result[i] = np.frombuffer(chunk, dtype=dtype, count=num_components)
        else:
            # Contiguous access
            byte_length = accessor.count * element_size
            data_slice = buffer_data[byte_offset : byte_offset + byte_length]
            result = np.frombuffer(data_slice, dtype=dtype).reshape(
                accessor.count, num_components
            )

        # For scalars, flatten to 1D
        if num_components == 1:
            result = result.flatten()

        return result.copy()  # Return a copy to ensure it's writable

    def get_image_data(self, image_index: int) -> Tuple[bytes, str]:
        """
        Extract image data and determine its format.
        Returns (image_bytes, extension).
        """
        image = self.gltf.images[image_index]

        if image.bufferView is not None:
            # Image embedded in buffer
            buffer_view = self.gltf.bufferViews[image.bufferView]
            buffer_data = self.get_buffer_data(buffer_view.buffer)
            byte_offset = buffer_view.byteOffset or 0
            byte_length = buffer_view.byteLength
            image_data = buffer_data[byte_offset : byte_offset + byte_length]
        elif image.uri:
            if image.uri.startswith("data:"):
                # Data URI
                header, encoded = image.uri.split(",", 1)
                image_data = base64.b64decode(encoded)
            else:
                # External file
                image_path = self.gltf_dir / image.uri
                image_data = image_path.read_bytes()
        else:
            raise ValueError(f"Cannot load image {image_index}: no data source")

        # Determine extension from mime type or magic bytes
        ext = ".png"  # default
        if image.mimeType:
            mime_to_ext = {
                "image/png": ".png",
                "image/jpeg": ".jpg",
                "image/jpg": ".jpg",
                "image/webp": ".webp",
            }
            ext = mime_to_ext.get(image.mimeType, ".png")
        elif len(image_data) >= 4:
            # Check magic bytes
            if image_data[:4] == b"\x89PNG":
                ext = ".png"
            elif image_data[:2] == b"\xff\xd8":
                ext = ".jpg"

        return image_data, ext


# -----------------------------------------------------------------------------
# USD Stage Building
# -----------------------------------------------------------------------------


def sanitize_name(name: str, prefix: str = "prim") -> str:
    """Convert a name to a valid USD identifier."""
    if not name:
        return prefix

    # Replace invalid characters
    result = ""
    for c in name:
        if c.isalnum() or c == "_":
            result += c
        else:
            result += "_"

    # Ensure it doesn't start with a digit
    if result and result[0].isdigit():
        result = prefix + "_" + result

    return result or prefix


def gltf_matrix_to_usd(matrix: List[float]) -> Gf.Matrix4d:
    """
    Convert a glTF column-major 4x4 matrix to USD's Gf.Matrix4d.
    glTF stores matrices in column-major order; USD uses row-major.
    """
    m = np.array(matrix, dtype=np.float64).reshape(4, 4).T
    return Gf.Matrix4d(*m.flatten().tolist())


def gltf_transform_to_matrix(
    translation: Optional[List[float]] = None,
    rotation: Optional[List[float]] = None,
    scale: Optional[List[float]] = None,
) -> Gf.Matrix4d:
    """
    Build a USD matrix from glTF TRS components.
    Rotation is a quaternion [x, y, z, w].
    """
    mat = Gf.Matrix4d(1.0)

    if scale:
        scale_mat = Gf.Matrix4d(1.0)
        scale_mat.SetScale(Gf.Vec3d(*scale))
        mat = mat * scale_mat

    if rotation:
        # glTF quaternion is [x, y, z, w]
        quat = Gf.Quatd(rotation[3], Gf.Vec3d(rotation[0], rotation[1], rotation[2]))
        rot_mat = Gf.Matrix4d(1.0)
        rot_mat.SetRotate(quat)
        mat = mat * rot_mat

    if translation:
        trans_mat = Gf.Matrix4d(1.0)
        trans_mat.SetTranslate(Gf.Vec3d(*translation))
        mat = mat * trans_mat

    return mat


class USDBuilder:
    """Builds a USD stage from parsed GLB data."""

    def __init__(self, stage: Usd.Stage, reader: GLBReader, texture_dir: Path):
        self.stage = stage
        self.reader = reader
        self.gltf = reader.gltf
        self.texture_dir = texture_dir
        self.material_cache: Dict[int, str] = {}
        self.texture_cache: Dict[int, str] = {}

    def build(self) -> None:
        """Build the complete USD stage from the glTF."""
        # Set up stage metadata
        UsdGeom.SetStageUpAxis(self.stage, UsdGeom.Tokens.y)
        UsdGeom.SetStageMetersPerUnit(self.stage, 1.0)

        # Create root xform
        root_path = "/Root"
        root_xform = UsdGeom.Xform.Define(self.stage, root_path)

        # CRITICAL: Set the default prim so that external references work correctly.
        # Without a default prim, when scene.usda references this USDZ via
        # `prepend references = @asset.usdz@`, USD won't know which prim to bring in,
        # resulting in empty geometry.
        self.stage.SetDefaultPrim(root_xform.GetPrim())

        # Process default scene or all scenes
        if self.gltf.scene is not None and self.gltf.scenes:
            scene = self.gltf.scenes[self.gltf.scene]
            if scene.nodes:
                for node_index in scene.nodes:
                    self._process_node(node_index, root_path)
        elif self.gltf.nodes:
            # No explicit scene, process all root nodes
            for i, node in enumerate(self.gltf.nodes):
                # Check if this node is a root (not a child of another node)
                is_child = False
                for other_node in self.gltf.nodes:
                    if other_node.children and i in other_node.children:
                        is_child = True
                        break
                if not is_child:
                    self._process_node(i, root_path)

    def _filter_degenerate_triangles(
        self, indices: list, points: list, mesh_name: str, epsilon: float = 1e-6
    ) -> list:
        """
        Filter out degenerate triangles from the index list.

        GAP-GEOMETRY-001 FIX: Detect and remove degenerate triangles that would cause
        rendering or physics issues.

        A triangle is degenerate if:
        - Any two vertices are the same (duplicate indices)
        - All three vertices are collinear (zero area)

        Args:
            indices: List of vertex indices (must be divisible by 3)
            points: List of Gf.Vec3f points
            mesh_name: Name for logging
            epsilon: Tolerance for zero-area check

        Returns:
            Filtered index list with degenerate triangles removed
        """
        if len(indices) % 3 != 0:
            logger.warning(f"Mesh '{mesh_name}': Index count {len(indices)} not divisible by 3")
            return indices

        filtered_indices = []
        degenerate_count = 0

        for i in range(0, len(indices), 3):
            i0, i1, i2 = indices[i], indices[i + 1], indices[i + 2]

            # Check for duplicate indices
            if i0 == i1 or i1 == i2 or i0 == i2:
                degenerate_count += 1
                continue

            # Check bounds
            if i0 >= len(points) or i1 >= len(points) or i2 >= len(points):
                degenerate_count += 1
                logger.warning(
                    f"Mesh '{mesh_name}': Triangle {i//3} has out-of-bounds index "
                    f"({i0}, {i1}, {i2}) >= {len(points)}"
                )
                continue

            # Check for zero-area triangle (collinear vertices)
            p0, p1, p2 = points[i0], points[i1], points[i2]

            # Calculate triangle area using cross product
            edge1 = Gf.Vec3f(p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2])
            edge2 = Gf.Vec3f(p2[0] - p0[0], p2[1] - p0[1], p2[2] - p0[2])

            # Cross product magnitude = 2 * area
            cross = Gf.Vec3f(
                edge1[1] * edge2[2] - edge1[2] * edge2[1],
                edge1[2] * edge2[0] - edge1[0] * edge2[2],
                edge1[0] * edge2[1] - edge1[1] * edge2[0],
            )
            area = 0.5 * (cross[0] ** 2 + cross[1] ** 2 + cross[2] ** 2) ** 0.5

            if area < epsilon:
                degenerate_count += 1
                continue

            # Triangle is valid
            filtered_indices.extend([i0, i1, i2])

        if degenerate_count > 0:
            logger.warning(
                f"Mesh '{mesh_name}': Filtered {degenerate_count} degenerate triangles "
                f"({degenerate_count * 100 // (len(indices) // 3)}% of total)"
            )

        return filtered_indices

    def _process_node(self, node_index: int, parent_path: str) -> None:
        """Recursively process a glTF node and its children."""
        node = self.gltf.nodes[node_index]
        node_name = sanitize_name(node.name or f"node_{node_index}", "node")
        node_path = f"{parent_path}/{node_name}"

        # Ensure unique path
        counter = 0
        base_path = node_path
        while self.stage.GetPrimAtPath(node_path):
            counter += 1
            node_path = f"{base_path}_{counter}"

        # Create xform for this node
        xform = UsdGeom.Xform.Define(self.stage, node_path)

        # Apply transform
        if node.matrix:
            mat = gltf_matrix_to_usd(node.matrix)
            xform.MakeMatrixXform().Set(mat)
        else:
            mat = gltf_transform_to_matrix(
                translation=node.translation,
                rotation=node.rotation,
                scale=node.scale,
            )
            if mat != Gf.Matrix4d(1.0):
                xform.MakeMatrixXform().Set(mat)

        # Process mesh if present
        if node.mesh is not None:
            self._process_mesh(node.mesh, node_path)

        # Process children
        if node.children:
            for child_index in node.children:
                self._process_node(child_index, node_path)

    def _process_mesh(self, mesh_index: int, parent_path: str) -> None:
        """Process a glTF mesh and create USD geometry."""
        mesh = self.gltf.meshes[mesh_index]
        mesh_name = sanitize_name(mesh.name or f"mesh_{mesh_index}", "mesh")

        for prim_idx, primitive in enumerate(mesh.primitives):
            # Create a unique name for each primitive
            prim_name = f"{mesh_name}" if len(mesh.primitives) == 1 else f"{mesh_name}_{prim_idx}"
            prim_path = f"{parent_path}/{prim_name}"

            # Ensure unique path
            counter = 0
            base_path = prim_path
            while self.stage.GetPrimAtPath(prim_path):
                counter += 1
                prim_path = f"{base_path}_{counter}"

            self._create_mesh_prim(primitive, prim_path)

    def _create_mesh_prim(self, primitive: Any, prim_path: str) -> None:
        """Create a USD Mesh prim from a glTF primitive."""
        attrs = primitive.attributes

        # Get positions (required)
        if attrs.POSITION is None:
            print(f"[WARN] Skipping primitive at {prim_path}: no POSITION attribute")
            return

        positions = self.reader.get_accessor_data(attrs.POSITION)

        # Create the mesh
        usd_mesh = UsdGeom.Mesh.Define(self.stage, prim_path)

        # Set points - CRITICAL FIX: Convert numpy array to Python list to get native float types
        # The USD Gf.Vec3f constructor requires Python floats, not numpy.float32
        points = [Gf.Vec3f(float(p[0]), float(p[1]), float(p[2])) for p in positions]
        usd_mesh.CreatePointsAttr(points)

        # Get and set indices
        if primitive.indices is not None:
            indices = self.reader.get_accessor_data(primitive.indices)
            indices_list = indices.astype(np.int32).tolist()

            # GAP-GEOMETRY-001 FIX: Filter degenerate triangles
            indices_list = self._filter_degenerate_triangles(indices_list, points, prim_path)

            usd_mesh.CreateFaceVertexIndicesAttr(Vt.IntArray(indices_list))

            # Calculate face vertex counts (assuming triangles)
            num_faces = len(indices_list) // 3
            face_counts = [3] * num_faces
            usd_mesh.CreateFaceVertexCountsAttr(Vt.IntArray(face_counts))
        else:
            # No indices - assume triangles with sequential vertices
            num_verts = len(positions)
            indices_list = list(range(num_verts))

            # GAP-GEOMETRY-001 FIX: Filter degenerate triangles
            indices_list = self._filter_degenerate_triangles(indices_list, points, prim_path)

            usd_mesh.CreateFaceVertexIndicesAttr(Vt.IntArray(indices_list))

            num_faces = len(indices_list) // 3
            face_counts = [3] * num_faces
            usd_mesh.CreateFaceVertexCountsAttr(Vt.IntArray(face_counts))

        # Set normals - CRITICAL FIX: Convert to native Python floats
        if attrs.NORMAL is not None:
            normals = self.reader.get_accessor_data(attrs.NORMAL)
            normal_vecs = [Gf.Vec3f(float(n[0]), float(n[1]), float(n[2])) for n in normals]
            usd_mesh.CreateNormalsAttr(normal_vecs)
            usd_mesh.SetNormalsInterpolation(UsdGeom.Tokens.vertex)

        # Set UVs (texture coordinates) - CRITICAL FIX: Convert to native Python floats
        if attrs.TEXCOORD_0 is not None:
            uvs = self.reader.get_accessor_data(attrs.TEXCOORD_0)
            # Flip V coordinate (glTF uses top-left origin, USD uses bottom-left)
            uvs[:, 1] = 1.0 - uvs[:, 1]
            uv_vecs = [Gf.Vec2f(float(uv[0]), float(uv[1])) for uv in uvs]

            # Create primvar for UVs
            primvar_api = UsdGeom.PrimvarsAPI(usd_mesh)
            uv_primvar = primvar_api.CreatePrimvar(
                "st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex
            )
            uv_primvar.Set(uv_vecs)

        # Set vertex colors if present - CRITICAL FIX: Convert to native Python floats
        if attrs.COLOR_0 is not None:
            colors = self.reader.get_accessor_data(attrs.COLOR_0)
            # Handle both RGB and RGBA
            if colors.shape[1] >= 3:
                color_vecs = [Gf.Vec3f(float(c[0]), float(c[1]), float(c[2])) for c in colors]
                primvar_api = UsdGeom.PrimvarsAPI(usd_mesh)
                color_primvar = primvar_api.CreatePrimvar(
                    "displayColor", Sdf.ValueTypeNames.Color3fArray, UsdGeom.Tokens.vertex
                )
                color_primvar.Set(color_vecs)

        # Apply material
        if primitive.material is not None:
            material_path = self._get_or_create_material(primitive.material)
            if material_path:
                UsdShade.MaterialBindingAPI(usd_mesh).Bind(
                    UsdShade.Material(self.stage.GetPrimAtPath(material_path))
                )

        # Set subdivision scheme to none (we want the mesh as-is)
        usd_mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)

    def _get_or_create_material(self, material_index: int) -> Optional[str]:
        """Get or create a USD material for a glTF material."""
        if material_index in self.material_cache:
            return self.material_cache[material_index]

        material = self.gltf.materials[material_index]
        mat_name = sanitize_name(material.name or f"material_{material_index}", "material")
        mat_path = f"/Root/Materials/{mat_name}"

        # Ensure unique path
        counter = 0
        base_path = mat_path
        while self.stage.GetPrimAtPath(mat_path):
            counter += 1
            mat_path = f"{base_path}_{counter}"

        # Create material
        usd_mat = UsdShade.Material.Define(self.stage, mat_path)

        # Create UsdPreviewSurface shader
        shader_path = f"{mat_path}/PreviewSurface"
        shader = UsdShade.Shader.Define(self.stage, shader_path)
        shader.CreateIdAttr("UsdPreviewSurface")

        # Get PBR metallic-roughness properties
        pbr = material.pbrMetallicRoughness
        if pbr:
            # Base color - CRITICAL FIX: Convert to native Python floats
            if pbr.baseColorFactor:
                color = [float(c) for c in pbr.baseColorFactor[:3]]
                shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(
                    Gf.Vec3f(*color)
                )
                if len(pbr.baseColorFactor) > 3:
                    shader.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(
                        float(pbr.baseColorFactor[3])
                    )

            # Base color texture
            if pbr.baseColorTexture:
                tex_path = self._get_or_create_texture(
                    pbr.baseColorTexture.index, mat_path, "diffuseTexture"
                )
                if tex_path:
                    tex_shader = UsdShade.Shader(self.stage.GetPrimAtPath(tex_path))
                    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(
                        tex_shader.ConnectableAPI(), "rgb"
                    )

            # Metallic - CRITICAL FIX: Convert to native Python float
            metallic = float(pbr.metallicFactor) if pbr.metallicFactor is not None else 1.0
            shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(metallic)

            # Roughness - CRITICAL FIX: Convert to native Python float
            roughness = float(pbr.roughnessFactor) if pbr.roughnessFactor is not None else 1.0
            shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(roughness)

            # GAP-MATERIAL-001 FIX: Handle metallic-roughness texture (packed texture)
            if hasattr(pbr, 'metallicRoughnessTexture') and pbr.metallicRoughnessTexture:
                tex_path = self._get_or_create_texture(
                    pbr.metallicRoughnessTexture.index, mat_path, "metallicRoughnessTexture"
                )
                if tex_path:
                    # Note: This is a packed texture (B=metallic, G=roughness)
                    # UsdPreviewSurface doesn't directly support packed textures,
                    # so we use it as roughness input and note the limitation
                    tex_shader = UsdShade.Shader(self.stage.GetPrimAtPath(tex_path))
                    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).ConnectToSource(
                        tex_shader.ConnectableAPI(), "g"
                    )

        # GAP-MATERIAL-002 FIX: Handle normal map texture
        if hasattr(material, 'normalTexture') and material.normalTexture:
            tex_path = self._get_or_create_texture(
                material.normalTexture.index, mat_path, "normalTexture"
            )
            if tex_path:
                tex_shader = UsdShade.Shader(self.stage.GetPrimAtPath(tex_path))
                shader.CreateInput("normal", Sdf.ValueTypeNames.Normal3f).ConnectToSource(
                    tex_shader.ConnectableAPI(), "rgb"
                )

        # GAP-MATERIAL-003 FIX: Handle occlusion texture (AO)
        if hasattr(material, 'occlusionTexture') and material.occlusionTexture:
            tex_path = self._get_or_create_texture(
                material.occlusionTexture.index, mat_path, "occlusionTexture"
            )
            if tex_path:
                tex_shader = UsdShade.Shader(self.stage.GetPrimAtPath(tex_path))
                shader.CreateInput("occlusion", Sdf.ValueTypeNames.Float).ConnectToSource(
                    tex_shader.ConnectableAPI(), "r"
                )

        # Emissive texture
        if hasattr(material, 'emissiveTexture') and material.emissiveTexture:
            tex_path = self._get_or_create_texture(
                material.emissiveTexture.index, mat_path, "emissiveTexture"
            )
            if tex_path:
                tex_shader = UsdShade.Shader(self.stage.GetPrimAtPath(tex_path))
                shader.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(
                    tex_shader.ConnectableAPI(), "rgb"
                )

        # Emissive factor - CRITICAL FIX: Convert to native Python floats
        if hasattr(material, 'emissiveFactor') and material.emissiveFactor:
            emissive = [float(e) for e in material.emissiveFactor]
            shader.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).Set(
                Gf.Vec3f(*emissive)
            )

        # Connect shader to material surface output
        usd_mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")

        self.material_cache[material_index] = mat_path
        return mat_path

    def _get_or_create_texture(
        self, texture_index: int, material_path: str, texture_name: str
    ) -> Optional[str]:
        """Create a texture reader shader and extract the image file."""
        if texture_index in self.texture_cache:
            # Return existing texture
            return self.texture_cache[texture_index]

        texture = self.gltf.textures[texture_index]
        if texture.source is None:
            return None

        # Extract image to disk
        try:
            image_data, ext = self.reader.get_image_data(texture.source)
        except Exception as e:
            print(f"[WARN] Failed to extract texture {texture_index}: {e}")
            return None

        image_filename = f"texture_{texture_index}{ext}"
        image_path = self.texture_dir / image_filename
        image_path.write_bytes(image_data)

        # Create texture reader shader
        tex_shader_path = f"{material_path}/{texture_name}"
        tex_shader = UsdShade.Shader.Define(self.stage, tex_shader_path)
        tex_shader.CreateIdAttr("UsdUVTexture")

        # Set file path (relative for USDZ packaging)
        tex_shader.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(f"./{image_filename}")
        tex_shader.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("repeat")
        tex_shader.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("repeat")

        # Create outputs
        tex_shader.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
        tex_shader.CreateOutput("a", Sdf.ValueTypeNames.Float)

        # Create UV reader for texture coordinates
        uv_reader_path = f"{material_path}/{texture_name}_uvReader"
        uv_reader = UsdShade.Shader.Define(self.stage, uv_reader_path)
        uv_reader.CreateIdAttr("UsdPrimvarReader_float2")
        uv_reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
        uv_reader.CreateOutput("result", Sdf.ValueTypeNames.Float2)

        # Connect UV reader to texture
        tex_shader.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(
            uv_reader.ConnectableAPI(), "result"
        )

        self.texture_cache[texture_index] = tex_shader_path
        return tex_shader_path


# -----------------------------------------------------------------------------
# Main Conversion Function
# -----------------------------------------------------------------------------


def convert_glb_to_usd(
    input_path: Path,
    output_path: Path,
    create_usdz: bool = True,
) -> bool:
    """
    Convert a GLB file to USD/USDZ.

    Args:
        input_path: Path to the input .glb file
        output_path: Path for the output .usd/.usda/.usdz file
        create_usdz: If True and output is .usdz, package as USDZ

    Returns:
        True if successful, False otherwise
    """
    print(f"[glb_to_usd] Converting {input_path} -> {output_path}")

    # Load GLB
    try:
        gltf = GLTF2().load(str(input_path))
    except Exception as e:
        print(f"[ERROR] Failed to load GLB: {e}")
        return False

    reader = GLBReader(gltf, input_path)

    # Determine output format
    output_suffix = output_path.suffix.lower()
    is_usdz = output_suffix == ".usdz"

    # Create a temporary directory for intermediate files
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        texture_dir = temp_path / "textures"
        texture_dir.mkdir()

        # Create intermediate USDA
        if is_usdz:
            usda_path = temp_path / "model.usda"
        else:
            usda_path = output_path

        # Create the stage
        stage = Usd.Stage.CreateNew(str(usda_path))

        # Build USD content
        builder = USDBuilder(stage, reader, texture_dir if is_usdz else output_path.parent)
        try:
            builder.build()
        except Exception as e:
            print(f"[ERROR] Failed to build USD stage: {e}")
            import traceback
            traceback.print_exc()
            return False

        # Save the stage
        stage.GetRootLayer().Save()
        print(f"[glb_to_usd] Created USD stage: {usda_path}")

        # Package as USDZ if requested
        if is_usdz:
            try:
                # Ensure output directory exists
                output_path.parent.mkdir(parents=True, exist_ok=True)

                # Copy textures next to the usda for packaging
                for tex_file in texture_dir.iterdir():
                    shutil.copy(tex_file, temp_path / tex_file.name)

                # Use UsdUtils to create USDZ package
                success = UsdUtils.CreateNewUsdzPackage(
                    Sdf.AssetPath(str(usda_path)),
                    str(output_path),
                )

                if not success:
                    # Fallback: try creating a simple zip-based USDZ
                    print("[glb_to_usd] UsdUtils.CreateNewUsdzPackage returned False, trying fallback...")
                    success = _create_usdz_fallback(usda_path, output_path, texture_dir)

                if success:
                    print(f"[glb_to_usd] Created USDZ package: {output_path}")
                else:
                    print(f"[ERROR] Failed to create USDZ package")
                    # Copy the USDA as fallback
                    usda_fallback = output_path.with_suffix(".usda")
                    shutil.copy(usda_path, usda_fallback)
                    print(f"[glb_to_usd] Saved fallback USDA: {usda_fallback}")
                    return False

            except Exception as e:
                print(f"[ERROR] Failed to create USDZ: {e}")
                import traceback
                traceback.print_exc()
                return False
        else:
            # For non-USDZ output, copy textures to output directory
            if is_usdz or texture_dir.exists():
                out_tex_dir = output_path.parent
                for tex_file in texture_dir.iterdir():
                    shutil.copy(tex_file, out_tex_dir / tex_file.name)

    return True


def _create_usdz_fallback(usda_path: Path, output_path: Path, texture_dir: Path) -> bool:
    """
    Fallback USDZ creation using Python's zipfile module.
    USDZ is essentially a zip file with specific structure.
    """
    import zipfile

    try:
        with zipfile.ZipFile(output_path, "w", compression=zipfile.ZIP_STORED) as zf:
            # Add the USDA file (must be first and at root)
            zf.write(usda_path, usda_path.name)

            # Add textures
            for tex_file in texture_dir.iterdir():
                zf.write(tex_file, tex_file.name)

        return True
    except Exception as e:
        print(f"[ERROR] Fallback USDZ creation failed: {e}")
        return False


# -----------------------------------------------------------------------------
# CLI Entry Point
# -----------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Convert GLB files to USD/USDZ format"
    )
    parser.add_argument("input", type=Path, help="Input GLB file")
    parser.add_argument("output", type=Path, help="Output USD/USDA/USDZ file")

    args = parser.parse_args()

    if not args.input.exists():
        print(f"[ERROR] Input file not found: {args.input}")
        return 1

    success = convert_glb_to_usd(args.input, args.output)
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())