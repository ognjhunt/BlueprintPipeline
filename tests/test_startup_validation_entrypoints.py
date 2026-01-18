from __future__ import annotations

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _has_validate_and_fail_fast_call(node: ast.AST) -> bool:
    for child in ast.walk(node):
        if not isinstance(child, ast.Call):
            continue
        func = child.func
        if isinstance(func, ast.Name) and func.id == "validate_and_fail_fast":
            return True
        if isinstance(func, ast.Attribute) and func.attr == "validate_and_fail_fast":
            return True
    return False


def _is_main_guard(node: ast.If) -> bool:
    test = node.test
    if not isinstance(test, ast.Compare):
        return False
    if len(test.ops) != 1 or not isinstance(test.ops[0], ast.Eq):
        return False
    if len(test.comparators) != 1:
        return False
    left = test.left
    right = test.comparators[0]
    return (
        isinstance(left, ast.Name)
        and left.id == "__name__"
        and isinstance(right, ast.Constant)
        and right.value == "__main__"
    )


def test_scene_generation_entrypoint_invokes_startup_validation() -> None:
    entrypoint_path = REPO_ROOT / "scene-generation-job" / "generate_scene_images.py"
    tree = ast.parse(entrypoint_path.read_text())

    main_guards = [node for node in tree.body if isinstance(node, ast.If) and _is_main_guard(node)]
    assert main_guards, "Expected a __main__ guard in generate_scene_images.py"

    assert any(_has_validate_and_fail_fast_call(guard) for guard in main_guards)
