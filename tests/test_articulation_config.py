import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from blueprint_sim.recipe_compiler.compiler import CompilerConfig, RecipeCompiler


def test_articulation_defaults_include_new_fields(tmp_path) -> None:
    compiler = RecipeCompiler(
        CompilerConfig(output_dir=str(tmp_path), asset_root=str(tmp_path))
    )

    obj = {
        "id": "cabinet_001",
        "is_articulated": True,
        "articulation_type": "door",
        "articulation_axes": ["y"],
    }

    config = compiler._get_articulation_config(obj, matched={})

    assert config["stiffness"] == 0.0
    assert config["friction"] == 0.0
    assert config["velocity_limit"] == 1.0
    assert config["effort_limit"] is None
