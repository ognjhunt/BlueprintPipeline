"""Recipe compiler utilities ported from BlueprintRecipe.

These modules provide a layered USD compilation pipeline that can be
reused by jobs inside BlueprintPipeline. The implementation is adapted
so it can run even when optional services (Gemini, remote sim clients)
are not available in the current environment.
"""

from .compiler import CompilationResult, CompilerConfig, RecipeCompiler
from .layer_manager import LayerManager
from .usd_builder import USDSceneBuilder

__all__ = [
    "CompilationResult",
    "CompilerConfig",
    "RecipeCompiler",
    "LayerManager",
    "USDSceneBuilder",
]
