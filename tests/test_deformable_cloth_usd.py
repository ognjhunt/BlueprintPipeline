from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


@pytest.mark.unit
def test_usd_assembly_authors_deformable_cloth(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify usd-assembly-job can author PhysX cloth schemas without pxr.PhysxSchema bindings.

    This test uses a tiny USD asset + simready wrapper and asserts the assembled
    scene.usda contains:
    - /World/PhysicsScene
    - /World/PhysxParticleSystem
    - PhysxDeformableSurfaceAPI + simulationOwner relationship on the referenced mesh prim.
    """
    monkeypatch.setenv("USD_ASSET_PREFETCH_CATALOG", "0")

    repo_root = Path(__file__).resolve().parents[1]
    usd_job_root = repo_root / "usd-assembly-job"
    if str(usd_job_root) not in sys.path:
        sys.path.append(str(usd_job_root))

    # Import after sys.path is updated.
    import build_scene_usd  # type: ignore

    from pxr import Gf, Sdf, Usd, UsdGeom

    root = tmp_path / "root"
    assets_prefix = "assets"
    usd_prefix = "usd"
    layout_dir = root / "layout"
    assets_dir = root / assets_prefix
    usd_dir = root / usd_prefix
    layout_dir.mkdir(parents=True, exist_ok=True)
    assets_dir.mkdir(parents=True, exist_ok=True)
    usd_dir.mkdir(parents=True, exist_ok=True)

    # Minimal visual USD: /Root with a single quad mesh.
    obj_dir = assets_dir / "obj_1"
    obj_dir.mkdir(parents=True, exist_ok=True)
    model_path = obj_dir / "model.usd"
    model_stage = Usd.Stage.CreateNew(str(model_path))
    root_xf = UsdGeom.Xform.Define(model_stage, "/Root")
    model_stage.SetDefaultPrim(root_xf.GetPrim())
    mesh = UsdGeom.Mesh.Define(model_stage, "/Root/ClothMesh")
    mesh.GetPointsAttr().Set(
        [
            Gf.Vec3f(-0.25, 0.0, -0.25),
            Gf.Vec3f(0.25, 0.0, -0.25),
            Gf.Vec3f(0.25, 0.0, 0.25),
            Gf.Vec3f(-0.25, 0.0, 0.25),
        ]
    )
    mesh.GetFaceVertexCountsAttr().Set([4])
    mesh.GetFaceVertexIndicesAttr().Set([0, 1, 2, 3])
    mesh.GetSubdivisionSchemeAttr().Set("none")
    model_stage.GetRootLayer().Save()

    # Minimal simready wrapper: /Asset with /Asset/Visual referencing model.usd.
    simready_path = obj_dir / "simready.usda"
    simready_path.write_text(
        "\n".join(
            [
                "#usda 1.0",
                "(",
                '    defaultPrim = "Asset"',
                "    metersPerUnit = 1",
                "    kilogramsPerUnit = 1",
                ")",
                "",
                'def Xform "Asset"',
                "{",
                '    def Xform "Visual" (',
                "        prepend references = @./model.usd@",
                "    )",
                "    {",
                "    }",
                "}",
                "",
            ]
        ),
        encoding="utf-8",
    )

    # Canonical manifest with deformable physics fields.
    manifest = {
        "version": "1.0.0",
        "scene_id": "test_deformable_cloth",
        "scene": {"coordinate_frame": "y_up", "meters_per_unit": 1.0},
        "objects": [
            {
                "id": "1",
                "category": "towel",
                "sim_role": "deformable_object",
                "asset": {"path": f"{assets_prefix}/obj_1/model.usd"},
                "transform": {
                    "position": {"x": 0.0, "y": 0.25, "z": 0.0},
                    "scale": {"x": 1.0, "y": 1.0, "z": 1.0},
                },
                "physics": {
                    "is_deformable": True,
                    "soft_body_type": "cloth",
                    "mass_kg": 0.3,
                    "static_friction": 0.7,
                    "dynamic_friction": 0.55,
                    "restitution": 0.05,
                    "deformable_schema": {
                        "type": "PhysxDeformableSurfaceAPI",
                        "solverPositionIterationCount": 7,
                        "selfCollision": True,
                        "collisionRestOffset": 0.001,
                        "collisionContactOffset": 0.002,
                        "deformableRestOffset": 0.001,
                        "selfCollisionFilterDistance": 0.003,
                        "bendingStiffnessScale": 0.3,
                        "stretchStiffnessScale": 0.5,
                    },
                },
            }
        ],
    }
    (assets_dir / "scene_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

    layout = {"objects": [{"id": "1", "center3d": [0.0, 0.25, 0.0]}]}
    layout_path = layout_dir / "scene_layout_scaled.json"
    layout_path.write_text(json.dumps(layout, indent=2), encoding="utf-8")

    output_path = usd_dir / "scene.usda"
    stage, _objects = build_scene_usd.build_scene(
        layout_path=layout_path,
        assets_path=assets_dir / "scene_manifest.json",
        output_path=output_path,
        root=root,
        assets_prefix=assets_prefix,
        usd_prefix=usd_prefix,
    )

    assert stage.GetPrimAtPath("/World/PhysicsScene").IsValid()
    psys = stage.GetPrimAtPath("/World/PhysxParticleSystem")
    assert psys.IsValid()
    assert psys.GetTypeName() == "PhysxParticleSystem"
    assert psys.GetAttribute("physxParticleSystem:solverPositionIterationCount").IsValid()

    # Find mesh under the assembled object prim.
    obj_prim = stage.GetPrimAtPath("/World/Objects/obj_1")
    assert obj_prim.IsValid()
    mesh_prim = None
    for p in Usd.PrimRange(obj_prim):
        if p.IsA(UsdGeom.Mesh):
            mesh_prim = p
            break
    assert mesh_prim is not None, "Expected a Mesh prim under /World/Objects/obj_1"

    api_schemas = mesh_prim.GetMetadata("apiSchemas")
    applied = []
    if isinstance(api_schemas, Sdf.TokenListOp):
        applied = list(api_schemas.GetAppliedItems())
    elif isinstance(api_schemas, (list, tuple)):
        applied = list(api_schemas)
    assert "PhysxDeformableSurfaceAPI" in applied

    # Relationship should target the particle system.
    rel = mesh_prim.GetRelationship("physxDeformable:simulationOwner")
    assert rel.IsValid()
    assert Sdf.Path("/World/PhysxParticleSystem") in rel.GetTargets()

    # A representative attribute should be authored.
    assert mesh_prim.GetAttribute("physxDeformable:solverPositionIterationCount").IsValid()
    assert int(mesh_prim.GetAttribute("physxDeformable:solverPositionIterationCount").Get()) == 7

