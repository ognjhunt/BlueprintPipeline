from __future__ import annotations

from tools.geniesim_adapter.deployment.patches import patch_object_pose_handler as patch_mod


def test_object_pose_helper_uses_resolver_v4_marker() -> None:
    assert patch_mod.RESOLVER_V4_MARKER == "object_pose_resolver_v4"
    assert "# object_pose_resolver_v4" in patch_mod.OBJECT_POSE_HELPER


def test_object_pose_helper_canonicalizes_nested_scene_paths() -> None:
    helper = patch_mod.OBJECT_POSE_HELPER
    assert "Canonicalize nested scene object paths" in helper
    assert r"^(/World/Scene/obj_[^/]+)/.+$" in helper
    assert "Prefer object root over exact nested match" in helper


def test_physics_pose_helper_skips_nested_scene_children() -> None:
    helper = patch_mod.PHYSICS_POSE_HELPER
    assert "_lp.startswith(\"/world/scene/obj_\")" in helper
    assert "_lp.count(\"/\") > 3" in helper
