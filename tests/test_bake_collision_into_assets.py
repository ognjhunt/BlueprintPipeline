from __future__ import annotations

import types
from pathlib import Path

from tools.geniesim_adapter.deployment.patches import bake_collision_into_assets as bake_mod


class _FakeAttr:
    def __init__(self, value=None):
        self._value = value

    def Get(self):
        return self._value

    def Set(self, value):
        self._value = value
        return self


class _FakePrim:
    def __init__(self, name: str, *, is_mesh: bool = False, apis: set[type] | None = None):
        self._name = name
        self._is_mesh = is_mesh
        self._apis = set(apis or set())
        self._attrs: dict[str, _FakeAttr] = {}

    def IsA(self, cls):
        return bool(self._is_mesh and cls.__name__ == "Mesh")

    def HasAPI(self, api_cls):
        return api_cls in self._apis

    def GetAttribute(self, key: str):
        return self._attrs.get(key)

    def CreateAttribute(self, key: str, _type=None):
        attr = self._attrs.get(key)
        if attr is None:
            attr = _FakeAttr()
            self._attrs[key] = attr
        return attr

    def GetName(self):
        return self._name


class _FakeRoot:
    def __init__(self, stage):
        self._stage = stage


class _FakeStage:
    def __init__(self, prims: list[_FakePrim]):
        self._prims = prims

    def GetPrimAtPath(self, path: str):
        if path == "/":
            return _FakeRoot(self)
        return None

    def GetRootLayer(self):
        return self

    def Save(self):
        return None


def _install_fake_pxr(monkeypatch, stage_by_path: dict[str, _FakeStage]):
    class Mesh:
        pass

    class CollisionAPI:
        @staticmethod
        def Apply(prim: _FakePrim):
            prim._apis.add(CollisionAPI)
            return CollisionAPI(prim)

        def __init__(self, prim: _FakePrim):
            self._prim = prim

        def CreateCollisionEnabledAttr(self):
            return self._prim.CreateAttribute("physics:collisionEnabled")

    class RigidBodyAPI:
        pass

    class _StageNS:
        @staticmethod
        def Open(path: str):
            return stage_by_path.get(path)

    def _prim_range(root: _FakeRoot):
        return list(root._stage._prims)

    fake_pxr = types.ModuleType("pxr")
    fake_pxr.Usd = types.SimpleNamespace(Stage=_StageNS, PrimRange=_prim_range)
    fake_pxr.UsdGeom = types.SimpleNamespace(Mesh=Mesh)
    fake_pxr.UsdPhysics = types.SimpleNamespace(CollisionAPI=CollisionAPI, RigidBodyAPI=RigidBodyAPI)
    fake_pxr.Sdf = types.SimpleNamespace(ValueTypeNames=types.SimpleNamespace(Token="Token"))
    monkeypatch.setitem(__import__("sys").modules, "pxr", fake_pxr)
    return CollisionAPI, RigidBodyAPI


def test_bake_collision_defaults_leave_kinematic_untouched(tmp_path: Path, monkeypatch, capsys) -> None:
    asset_file = tmp_path / "obj_A" / "A.usd"
    asset_file.parent.mkdir(parents=True, exist_ok=True)
    asset_file.write_text("placeholder")

    stage_map: dict[str, _FakeStage] = {}
    collision_api, rigid_api = _install_fake_pxr(monkeypatch, stage_map)
    mesh = _FakePrim("MeshA", is_mesh=True)
    rigid = _FakePrim("RigidA", apis={rigid_api})
    rigid.CreateAttribute("physics:kinematicEnabled").Set(False)
    stage_map[str(asset_file)] = _FakeStage([mesh, rigid])

    ok = bake_mod.bake_collision(str(tmp_path), kinematic_names=None)
    out = capsys.readouterr().out

    assert ok is True
    assert mesh.HasAPI(collision_api) is True
    assert rigid.GetAttribute("physics:kinematicEnabled").Get() is False
    assert "collision_added_count: 1" in out
    assert "kinematic_fixed_true_count: 0" in out
    assert "kinematic_untouched_count: 1" in out


def test_bake_collision_honors_explicit_kinematic_names(tmp_path: Path, monkeypatch, capsys) -> None:
    asset_file = tmp_path / "obj_A" / "A.usd"
    asset_file.parent.mkdir(parents=True, exist_ok=True)
    asset_file.write_text("placeholder")

    stage_map: dict[str, _FakeStage] = {}
    _, rigid_api = _install_fake_pxr(monkeypatch, stage_map)
    rigid = _FakePrim("RigidA", apis={rigid_api})
    rigid.CreateAttribute("physics:kinematicEnabled").Set(False)
    stage_map[str(asset_file)] = _FakeStage([rigid])

    ok = bake_mod.bake_collision(str(tmp_path), kinematic_names=["A"])
    out = capsys.readouterr().out

    assert ok is True
    assert rigid.GetAttribute("physics:kinematicEnabled").Get() is True
    assert "kinematic_fixed_true_count: 1" in out
    assert "kinematic_untouched_count: 0" in out
