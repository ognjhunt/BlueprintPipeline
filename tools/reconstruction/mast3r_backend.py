"""MASt3R Depth-Based Reconstruction Backend.

Uses MASt3R (Masked Attention in Stereo Transformers) for depth estimation
from multi-view images, then converts to meshes.

Reference: https://github.com/naver/mast3r

Requirements:
    pip install torch torchvision
    pip install git+https://github.com/naver/mast3r
    pip install open3d trimesh

Input: Directory with multi-view images of a scene
Output: Scene mesh + per-object segmented meshes (if segmentation available)
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .base import (
    ReconstructionBackend,
    ReconstructionBackendType,
    ReconstructionResult,
    ReconstructedObject,
)


@dataclass
class MASt3RConfig:
    """Configuration for MASt3R reconstruction."""
    model_name: str = "naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric"
    device: str = "cuda"  # cuda or cpu
    image_size: int = 512
    min_conf_thr: float = 1.5
    as_pointcloud: bool = False  # Output pointcloud instead of mesh
    mask_sky: bool = True
    clean_depth: bool = True
    subsample: int = 8  # Mesh subsampling factor


class MASt3RBackend(ReconstructionBackend):
    """MASt3R-based 3D reconstruction backend."""

    backend_type = ReconstructionBackendType.MAST3R

    def __init__(
        self,
        config: Optional[MASt3RConfig] = None,
        verbose: bool = True,
    ):
        super().__init__(verbose=verbose)
        self.config = config or MASt3RConfig()
        self._mast3r_available = None

    def is_available(self) -> bool:
        """Check if MASt3R is available."""
        if self._mast3r_available is not None:
            return self._mast3r_available

        try:
            # Check for required packages
            import torch
            import torchvision

            # Check for MASt3R
            try:
                from mast3r.model import AsymmetricMASt3R
                self._mast3r_available = True
            except ImportError:
                self.log("MASt3R not installed. Install with: pip install git+https://github.com/naver/mast3r")
                self._mast3r_available = False

            # Check for Open3D (for meshing)
            try:
                import open3d
            except ImportError:
                self.log("Open3D not installed. Install with: pip install open3d")
                self._mast3r_available = False

            return self._mast3r_available

        except ImportError as e:
            self.log(f"Missing dependency: {e}")
            self._mast3r_available = False
            return False

    def get_requirements(self) -> Dict[str, str]:
        """Get MASt3R requirements."""
        return {
            "torch": ">=2.0.0",
            "torchvision": ">=0.15.0",
            "mast3r": "git+https://github.com/naver/mast3r",
            "open3d": ">=0.17.0",
            "trimesh": ">=3.21.0",
            "description": "Multi-view stereo reconstruction using MASt3R",
        }

    def validate_input(self, input_path: Path) -> bool:
        """Validate input is a directory with images."""
        if not input_path.is_dir():
            return False

        # Check for images
        image_exts = {".jpg", ".jpeg", ".png", ".webp"}
        images = [
            f for f in input_path.iterdir()
            if f.suffix.lower() in image_exts
        ]

        return len(images) >= 2  # Need at least 2 views

    def reconstruct(
        self,
        input_path: Path,
        output_dir: Path,
        scene_id: str,
        environment_type: str = "generic",
        segmentation_masks: Optional[Path] = None,
        **kwargs
    ) -> ReconstructionResult:
        """Run MASt3R reconstruction.

        Args:
            input_path: Directory with multi-view images
            output_dir: Where to write outputs
            scene_id: Scene identifier
            environment_type: Type of environment
            segmentation_masks: Optional directory with per-image segmentation masks
            **kwargs: Additional MASt3R options

        Returns:
            ReconstructionResult with reconstructed scene
        """
        start_time = time.time()
        self.log(f"Running MASt3R reconstruction on: {input_path}")

        if not self.is_available():
            return ReconstructionResult(
                success=False,
                scene_id=scene_id,
                backend=self.backend_type,
                error="MASt3R not available. Install dependencies first.",
            )

        input_path = Path(input_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        try:
            # Import MASt3R
            import torch
            from mast3r.model import AsymmetricMASt3R
            from mast3r.fast_nn import fast_reciprocal_NNs
            from dust3r.inference import inference
            from dust3r.utils.image import load_images
            from dust3r.image_pairs import make_pairs
            from dust3r.cloud_opt import global_aligner, GlobalAlignerMode

            # Load model
            self.log(f"Loading MASt3R model: {self.config.model_name}")
            model = AsymmetricMASt3R.from_pretrained(self.config.model_name).to(self.config.device)

            # Find images
            image_exts = {".jpg", ".jpeg", ".png", ".webp"}
            image_paths = sorted([
                f for f in input_path.iterdir()
                if f.suffix.lower() in image_exts
            ])
            self.log(f"Found {len(image_paths)} images")

            # Load images
            images = load_images(
                [str(p) for p in image_paths],
                size=self.config.image_size
            )

            # Create image pairs
            pairs = make_pairs(
                images,
                scene_graph="complete",  # All pairs
                prefilter=None,
                symmetrize=True
            )

            # Run inference
            self.log("Running MASt3R inference...")
            output = inference(pairs, model, self.config.device, batch_size=1)

            # Global alignment
            self.log("Running global alignment...")
            scene = global_aligner(
                output,
                device=self.config.device,
                mode=GlobalAlignerMode.PointCloudOptimizer
            )

            # Optimize
            loss = scene.compute_global_alignment(
                init="mst",
                niter=300,
                schedule="cosine",
                lr=0.01
            )
            self.log(f"Alignment loss: {loss}")

            # Get point cloud
            pts3d = scene.get_pts3d()
            confidence = scene.get_conf()

            # Filter by confidence
            mask = confidence > self.config.min_conf_thr

            # Export mesh
            assets_dir = output_dir / "assets"
            assets_dir.mkdir(exist_ok=True)

            mesh_path = assets_dir / "scene_mesh.glb"
            self._export_mesh(pts3d, mask, images, mesh_path)

            # Create single scene object
            # (For per-object segmentation, we'd need SAM or similar)
            objects = [
                ReconstructedObject(
                    id="scene_0",
                    mesh_path=mesh_path,
                    category="scene",
                    position=(0, 0, 0),
                    sim_role="static",
                    source_backend=self.backend_type.value,
                )
            ]

            # If segmentation masks provided, segment objects
            if segmentation_masks and Path(segmentation_masks).is_dir():
                self.log("Segmenting objects from masks...")
                objects = self._segment_objects(
                    pts3d, mask, images,
                    Path(segmentation_masks),
                    assets_dir
                )

            processing_time = time.time() - start_time

            result = ReconstructionResult(
                success=True,
                scene_id=scene_id,
                backend=self.backend_type,
                objects=objects,
                background_mesh=mesh_path if not segmentation_masks else None,
                environment_type=environment_type,
                processing_time_seconds=processing_time,
            )

            # Save manifest
            manifest_output = output_dir / "scene_manifest.json"
            manifest_output.write_text(json.dumps(result.to_manifest(), indent=2))

            self.log(f"Reconstruction complete in {processing_time:.1f}s")

            return result

        except Exception as e:
            import traceback
            self.log(f"Reconstruction failed: {e}")
            traceback.print_exc()

            return ReconstructionResult(
                success=False,
                scene_id=scene_id,
                backend=self.backend_type,
                error=str(e),
            )

    def _export_mesh(
        self,
        pts3d,
        mask,
        images,
        output_path: Path,
    ) -> None:
        """Export point cloud to mesh."""
        import numpy as np
        import open3d as o3d
        import trimesh

        # Gather points and colors
        all_points = []
        all_colors = []

        for i, (pts, conf_mask, img_data) in enumerate(zip(pts3d, mask, images)):
            pts_np = pts.detach().cpu().numpy()
            mask_np = conf_mask.detach().cpu().numpy()

            # Get image colors
            img = img_data["img"]
            if hasattr(img, "numpy"):
                img = img.numpy()
            img = np.transpose(img, (1, 2, 0))  # CHW -> HWC
            img = (img * 255).astype(np.uint8)

            # Apply mask
            h, w = mask_np.shape
            pts_flat = pts_np.reshape(-1, 3)
            mask_flat = mask_np.reshape(-1)
            img_flat = img.reshape(-1, 3)

            valid_pts = pts_flat[mask_flat]
            valid_colors = img_flat[mask_flat]

            all_points.append(valid_pts)
            all_colors.append(valid_colors)

        points = np.concatenate(all_points, axis=0)
        colors = np.concatenate(all_colors, axis=0) / 255.0

        # Subsample if too many points
        if len(points) > 1_000_000:
            indices = np.random.choice(len(points), 1_000_000, replace=False)
            points = points[indices]
            colors = colors[indices]

        # Create Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)

        # Estimate normals
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
        )

        # Create mesh using Poisson reconstruction
        self.log("Running Poisson surface reconstruction...")
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd, depth=9
        )

        # Remove low-density vertices
        densities = np.asarray(densities)
        density_threshold = np.quantile(densities, 0.1)
        vertices_to_remove = densities < density_threshold
        mesh.remove_vertices_by_mask(vertices_to_remove)

        # Convert to trimesh for GLB export
        vertices = np.asarray(mesh.vertices)
        faces = np.asarray(mesh.triangles)
        vertex_colors = np.asarray(mesh.vertex_colors) if mesh.has_vertex_colors() else None

        trimesh_mesh = trimesh.Trimesh(
            vertices=vertices,
            faces=faces,
            vertex_colors=vertex_colors,
        )

        # Export to GLB
        trimesh_mesh.export(str(output_path), file_type="glb")
        self.log(f"Mesh exported to: {output_path}")

    def _segment_objects(
        self,
        pts3d,
        mask,
        images,
        mask_dir: Path,
        output_dir: Path,
    ) -> List[ReconstructedObject]:
        """Segment objects using provided masks."""
        import numpy as np
        from PIL import Image
        import trimesh

        objects = []

        # Load segmentation masks
        mask_files = sorted(mask_dir.glob("*.png"))
        if not mask_files:
            self.log("No segmentation masks found")
            return objects

        # Get unique object IDs from masks
        first_mask = np.array(Image.open(mask_files[0]))
        unique_ids = np.unique(first_mask)
        unique_ids = unique_ids[unique_ids > 0]  # Skip background

        for obj_id in unique_ids:
            obj_name = f"obj_{obj_id}"
            obj_dir = output_dir / obj_name
            obj_dir.mkdir(exist_ok=True)

            # Collect points for this object
            obj_points = []
            obj_colors = []

            for i, (pts, conf_mask, img_data, mask_file) in enumerate(
                zip(pts3d, mask, images, mask_files)
            ):
                seg_mask = np.array(Image.open(mask_file))
                obj_mask = (seg_mask == obj_id) & conf_mask.detach().cpu().numpy()

                pts_np = pts.detach().cpu().numpy()
                img = img_data["img"]
                if hasattr(img, "numpy"):
                    img = img.numpy()
                img = np.transpose(img, (1, 2, 0))

                h, w = obj_mask.shape
                pts_flat = pts_np.reshape(-1, 3)
                mask_flat = obj_mask.reshape(-1)
                img_flat = (img * 255).reshape(-1, 3).astype(np.uint8)

                valid_pts = pts_flat[mask_flat]
                valid_colors = img_flat[mask_flat]

                obj_points.append(valid_pts)
                obj_colors.append(valid_colors)

            if not obj_points:
                continue

            points = np.concatenate(obj_points, axis=0)
            colors = np.concatenate(obj_colors, axis=0) / 255.0

            if len(points) < 100:
                continue

            # Create mesh for this object
            mesh_path = obj_dir / "mesh.glb"

            try:
                # Simple point cloud to mesh
                import open3d as o3d

                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(points)
                pcd.colors = o3d.utility.Vector3dVector(colors)
                pcd.estimate_normals()

                # Ball pivoting or alpha shapes
                mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=7)

                vertices = np.asarray(mesh.vertices)
                faces = np.asarray(mesh.triangles)

                trimesh_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
                trimesh_mesh.export(str(mesh_path), file_type="glb")

                # Compute bounds
                bounds = trimesh_mesh.bounds
                center = trimesh_mesh.centroid

                objects.append(ReconstructedObject(
                    id=obj_name,
                    mesh_path=mesh_path,
                    category="object",
                    position=tuple(center),
                    bounds_min=tuple(bounds[0]),
                    bounds_max=tuple(bounds[1]),
                    source_backend=self.backend_type.value,
                ))

                self.log(f"  Segmented: {obj_name} ({len(points)} points)")

            except Exception as e:
                self.log(f"  Failed to mesh {obj_name}: {e}")

        return objects


def run_mast3r_cli(
    input_dir: str,
    output_dir: str,
    scene_id: str = "scene",
    **kwargs
) -> ReconstructionResult:
    """CLI wrapper for MASt3R reconstruction.

    Usage:
        python -m tools.reconstruction.mast3r_backend \\
            --input-dir ./my_images \\
            --output-dir ./output \\
            --scene-id my_scene
    """
    backend = MASt3RBackend()
    return backend.reconstruct(
        input_path=Path(input_dir),
        output_dir=Path(output_dir),
        scene_id=scene_id,
        **kwargs
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="MASt3R 3D Reconstruction")
    parser.add_argument("--input-dir", required=True, help="Directory with input images")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--scene-id", default="scene", help="Scene identifier")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])

    args = parser.parse_args()

    config = MASt3RConfig(device=args.device)
    backend = MASt3RBackend(config=config)

    result = backend.reconstruct(
        input_path=Path(args.input_dir),
        output_dir=Path(args.output_dir),
        scene_id=args.scene_id,
    )

    if result.success:
        print(f"Success! Reconstructed {len(result.objects)} objects")
    else:
        print(f"Failed: {result.error}")
