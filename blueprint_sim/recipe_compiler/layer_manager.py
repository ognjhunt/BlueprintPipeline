"""
Layer Manager - Manages USD layer creation and composition.

This module handles the creation and management of USD layers for
layered composition. It is ported from BlueprintRecipe to enable shared
USD assembly inside BlueprintPipeline.
"""

from pathlib import Path
from typing import Any, Optional

from .usd_builder import StubLayer


class LayerManager:
    """
    Manages USD layers for scene composition.

    BlueprintRecipe uses a layered USD approach:
    - scene.usda: Root stage with sublayer references
    - layers/room_shell.usda: Room geometry
    - layers/layout.usda: Object placements
    - layers/semantics.usda: Semantic annotations
    - layers/physics_overrides.usda: Physics properties

    This allows non-destructive editing and separation of concerns.
    """

    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.layers: dict[str, Any] = {}
        self._has_usd = self._check_usd_available()

        if self._has_usd:
            from pxr import Usd, Sdf

            self.Usd = Usd
            self.Sdf = Sdf

    def _check_usd_available(self) -> bool:
        """Check if OpenUSD is available."""
        try:
            from pxr import Usd  # noqa: F401

            return True
        except ImportError:
            return False

    def create_layer(self, name: str, path: Path) -> Any:
        """
        Create a new USD layer.

        Args:
            name: Logical name for the layer
            path: File path for the layer

        Returns:
            Layer object (pxr.Sdf.Layer or StubLayer)
        """
        if self._has_usd:
            layer = self.Sdf.Layer.CreateNew(str(path))
            stage = self.Usd.Stage.Open(layer)
            layer_obj = USDLayerWrapper(name, layer, stage)
        else:
            layer_obj = StubLayer(name, str(path))

        self.layers[name] = layer_obj
        return layer_obj

    def get_layer(self, name: str) -> Optional[Any]:
        """Get a layer by name."""
        return self.layers.get(name)

    def save_all(self) -> None:
        """Save all managed layers."""
        for layer in self.layers.values():
            layer.save()

    def get_relative_paths(self, from_dir: Path) -> dict[str, str]:
        """Get relative paths to all layers from a given directory."""
        paths = {}
        for name, layer in self.layers.items():
            layer_path = Path(layer.path)
            try:
                rel_path = layer_path.relative_to(from_dir)
                paths[name] = f"./{rel_path}"
            except ValueError:
                paths[name] = str(layer_path)
        return paths


class USDLayerWrapper:
    """Wrapper around USD layer/stage for consistent API."""

    def __init__(self, name: str, layer: Any, stage: Any):
        self.name = name
        self.layer = layer
        self.stage = stage
        self.path = layer.identifier

    def save(self) -> None:
        """Save the layer."""
        self.stage.Save()
        self.layer.Save()


__all__ = ["LayerManager", "USDLayerWrapper"]
