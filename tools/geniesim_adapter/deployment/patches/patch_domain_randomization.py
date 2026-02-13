#!/usr/bin/env python3
"""Patch: Domain randomization (P1 essential tier) via set_task_metric dispatch.

This adds a lightweight, reproducible domain randomization hook that can be
triggered per episode from the client using:

  set_task_metric("bp::domain_randomization::<json>")

The payload is queued from grpc_server.py onto the CommandController and applied
on the sim thread (inside on_command_step) to avoid USD threading deadlocks.

Implemented randomizations (best-effort, dependency-light):
- Visual: bind a UsdPreviewSurface material with randomized color/roughness to
  dynamic scene objects (root prim binding, inherited by descendants).
- Physics: bind a PhysX material with randomized friction/restitution to mesh
  descendants (materialPurpose="physics") and scale MassAPI.mass.

Notes:
- This is designed to be additive and safe; failures are logged but do not
  crash the server.
- The patch is idempotent.
"""

from __future__ import annotations

import os
import re
import sys

GENIESIM_ROOT = os.environ.get("GENIESIM_ROOT", "/opt/geniesim")
CC_PATH = os.environ.get(
    "GENIESIM_CC_PATH",
    os.path.join(GENIESIM_ROOT, "source", "data_collection", "server", "command_controller.py"),
)
GRPC_PATH = os.environ.get(
    "GENIESIM_GRPC_PATH",
    os.path.join(GENIESIM_ROOT, "source", "data_collection", "server", "grpc_server.py"),
)

MARKER = "# BPv_domain_randomization"
DISPATCH_MARKER = f"{MARKER} -- set_task_metric dispatch"


def _inject_into_set_task_metric(src: str, injected: str, marker: str) -> tuple[str, bool]:
    """Inject code at the top of set_task_metric() unless marker already exists."""
    if marker in src:
        return src, False
    match = re.search(
        r"(def set_task_metric\\(self[^)]*\\)[^:]*:)\\s*\\n",
        src,
    )
    if not match:
        return src, False
    insert_at = match.end()
    return src[:insert_at] + injected + src[insert_at:], True


def apply() -> bool:
    # ---------------------------------------------------------------------
    # Patch command_controller.py: apply queued domain randomization on sim thread
    # ---------------------------------------------------------------------
    try:
        with open(CC_PATH, "r") as f:
            cc_src = f.read()
    except OSError as exc:
        print(f"[PATCH] domain_randomization: command_controller.py read failed: {exc}")
        return False

    if MARKER in cc_src:
        print("[PATCH] domain_randomization: already applied to command_controller.py")
    else:
        inject_point = "def on_command_step(self):"
        if inject_point not in cc_src:
            print("[PATCH] domain_randomization: FAILED (on_command_step not found)")
            return False

        # NOTE: Indentation matters. This is injected into a method body inside
        # command_controller.py (class scope), so it must start at 8 spaces.
        handler = f"""
        {MARKER}
        # Process queued domain randomization requests (runs on sim thread).
        if hasattr(self, "_bp_domain_randomization_queue") and self._bp_domain_randomization_queue:
            _bp_payloads = list(self._bp_domain_randomization_queue)
            self._bp_domain_randomization_queue.clear()
            for _bp_payload in _bp_payloads:
                try:
                    import json as _bp_json
                    import random as _bp_random
                    import omni.usd
                    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade, PhysxSchema

                    stage = None
                    try:
                        if hasattr(self, "sim_app") and self.sim_app:
                            _ctx = self.sim_app.get_active_context()
                            if _ctx:
                                stage = _ctx.get_stage()
                    except Exception:
                        stage = None
                    if stage is None and hasattr(self, "_stage"):
                        stage = getattr(self, "_stage", None)
                    if stage is None:
                        stage = omni.usd.get_context().get_stage()
                    if stage is None:
                        print("[DR] No USD stage; skipping domain randomization")
                        continue

                    if isinstance(_bp_payload, str):
                        try:
                            _bp_payload = _bp_json.loads(_bp_payload)
                        except Exception:
                            _bp_payload = {{}}
                    if not isinstance(_bp_payload, dict):
                        _bp_payload = {{}}

                    _seed = int(_bp_payload.get("seed", 0) or 0)
                    _bp_random.seed(_seed)

                    _physics_cfg = _bp_payload.get("physics") if isinstance(_bp_payload.get("physics"), dict) else {{}}
                    _visual_cfg = _bp_payload.get("visual") if isinstance(_bp_payload.get("visual"), dict) else {{}}

                    # Resolve dynamic object roots: prefer patch_register_scene_objects set.
                    dyn_roots = list(getattr(self, "_bp_dynamic_scene_prims", set()) or [])
                    if not dyn_roots:
                        # Fallback: discover dynamic rigid bodies under /World/Scene
                        scene_root = stage.GetPrimAtPath("/World/Scene")
                        if scene_root and scene_root.IsValid():
                            for child in scene_root.GetChildren():
                                try:
                                    if child.HasAPI(UsdPhysics.RigidBodyAPI):
                                        ka = child.GetAttribute("physics:kinematicEnabled")
                                        if ka and ka.Get() is False:
                                            dyn_roots.append(str(child.GetPath()))
                                except Exception:
                                    pass
                    dyn_roots = sorted(set([str(p) for p in dyn_roots if p]))

                    max_objects = int(_bp_payload.get("max_objects") or _visual_cfg.get("max_objects") or 0)
                    if max_objects > 0 and len(dyn_roots) > max_objects:
                        _bp_random.shuffle(dyn_roots)
                        dyn_roots = dyn_roots[:max_objects]

                    # Cache base masses to avoid compounding mass_scale over many episodes.
                    if not hasattr(self, "_bp_dr_base_masses"):
                        self._bp_dr_base_masses = {{}}
                    _base_masses = self._bp_dr_base_masses

                    # -----------------------
                    # Physics randomization
                    # -----------------------
                    static_fric = _physics_cfg.get("static_friction")
                    dyn_fric = _physics_cfg.get("dynamic_friction")
                    restitution = _physics_cfg.get("restitution")
                    mass_scale = _physics_cfg.get("mass_scale")

                    # Create a physics material if requested.
                    phys_mat = None
                    if static_fric is not None or dyn_fric is not None or restitution is not None:
                        try:
                            static_fric_v = float(static_fric) if static_fric is not None else 0.6
                            dyn_fric_v = float(dyn_fric) if dyn_fric is not None else min(static_fric_v, 0.6)
                            rest_v = float(restitution) if restitution is not None else 0.05
                            phys_root = "/World/BlueprintPipeline/PhysicsMaterials"
                            phys_path = f"{{phys_root}}/dr_phys_{{_seed}}"
                            phys_material = UsdShade.Material.Define(stage, phys_path)
                            PhysxSchema.PhysxMaterialAPI.Apply(phys_material.GetPrim())
                            phys_api = PhysxSchema.PhysxMaterialAPI(phys_material.GetPrim())
                            phys_api.CreateStaticFrictionAttr().Set(static_fric_v)
                            phys_api.CreateDynamicFrictionAttr().Set(dyn_fric_v)
                            phys_api.CreateRestitutionAttr().Set(rest_v)
                            phys_mat = phys_material
                        except Exception as _phys_err:
                            print(f"[DR] Physics material creation failed: {{_phys_err}}")
                            phys_mat = None

                    # Apply physics: bind physics material to meshes and set mass (relative to cached base mass).
                    if phys_mat is not None or mass_scale is not None:
                        mass_scale_v = None
                        if mass_scale is not None:
                            try:
                                mass_scale_v = float(mass_scale)
                            except Exception:
                                mass_scale_v = None

                        for root_path in dyn_roots:
                            prim = stage.GetPrimAtPath(root_path)
                            if not prim or not prim.IsValid():
                                continue

                            # Mass scaling at rigid body root (non-compounding)
                            if mass_scale_v is not None and mass_scale_v > 0:
                                try:
                                    mass_api = UsdPhysics.MassAPI.Apply(prim)
                                    mass_attr = mass_api.GetMassAttr()
                                    if not mass_attr:
                                        mass_attr = mass_api.CreateMassAttr()
                                    cur = mass_attr.Get()
                                    if cur is not None:
                                        cur_f = float(cur)
                                        base = _base_masses.get(root_path)
                                        if base is None and cur_f > 0:
                                            base = cur_f
                                            _base_masses[root_path] = base
                                        if base is not None and float(base) > 0:
                                            mass_attr.Set(float(base) * mass_scale_v)
                                except Exception:
                                    pass

                            # Bind physics material to mesh descendants (materialPurpose="physics")
                            if phys_mat is not None:
                                try:
                                    for desc in Usd.PrimRange(prim):
                                        if not desc.IsA(UsdGeom.Mesh):
                                            continue
                                        try:
                                            UsdShade.MaterialBindingAPI(desc).Bind(
                                                phys_mat,
                                                materialPurpose="physics",
                                            )
                                        except Exception:
                                            # Older USD may not accept materialPurpose kwarg; fallback to default bind.
                                            UsdShade.MaterialBindingAPI(desc).Bind(phys_mat)
                                except Exception:
                                    pass

                    # -----------------------
                    # Visual randomization
                    # -----------------------
                    enable_visual = bool(_visual_cfg.get("enabled", True))
                    if enable_visual:
                        mat_root = "/World/BlueprintPipeline/Materials/DomainRandomization"
                        for idx_obj, root_path in enumerate(dyn_roots):
                            prim = stage.GetPrimAtPath(root_path)
                            if not prim or not prim.IsValid():
                                continue
                            try:
                                # Create a per-object material (inherited by descendants)
                                color = (
                                    float(_visual_cfg.get("r", _bp_random.random())),
                                    float(_visual_cfg.get("g", _bp_random.random())),
                                    float(_visual_cfg.get("b", _bp_random.random())),
                                )
                                rough = float(_visual_cfg.get("roughness", _bp_random.uniform(0.2, 0.9)))
                                metal = float(_visual_cfg.get("metallic", _bp_random.uniform(0.0, 0.2)))
                                mat_path = f"{{mat_root}}/dr_mat_{{_seed}}_{{idx_obj:03d}}"
                                material = UsdShade.Material.Define(stage, mat_path)
                                shader = UsdShade.Shader.Define(stage, mat_path + "/Shader")
                                shader.CreateIdAttr("UsdPreviewSurface")
                                shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(
                                    Gf.Vec3f(color[0], color[1], color[2])
                                )
                                shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(rough)
                                shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(metal)
                                material.CreateSurfaceOutput().ConnectToSource(shader, "surface")
                                UsdShade.MaterialBindingAPI(prim).Bind(material)
                            except Exception:
                                pass

                    print(
                        f"[DR] Applied domain randomization seed={{_seed}} "
                        f"(dyn_objects={{len(dyn_roots)}}, physics={{phys_mat is not None}}, "
                        f"mass_scale={{mass_scale}}, visual={{enable_visual}})"
                    )
                except Exception as _bp_err:
                    print(f"[DR] Domain randomization payload failed: {{_bp_err}}")
"""

        cc_src = cc_src.replace(inject_point, inject_point + handler, 1)
        try:
            with open(CC_PATH, "w") as f:
                f.write(cc_src)
            print("[PATCH] domain_randomization: APPLIED to command_controller.py")
        except OSError as exc:
            print(f"[PATCH] domain_randomization: command_controller.py write failed: {exc}")
            return False

    # ---------------------------------------------------------------------
    # Patch grpc_server.py: dispatch bp::domain_randomization::... metrics
    # ---------------------------------------------------------------------
    try:
        with open(GRPC_PATH, "r") as f:
            grpc_src = f.read()
    except OSError as exc:
        print(f"[PATCH] domain_randomization: grpc_server.py read failed: {exc}")
        return False

    injected = f"""
        {DISPATCH_MARKER}
        _metric_str = str(getattr(req, "metric", "") or "")
        if _metric_str.startswith("bp::domain_randomization::"):
            try:
                import json as _bp_json

                _bp_payload_str = _metric_str.split("::", 2)[2]
                _bp_payload = _bp_json.loads(_bp_payload_str)

                _cc = getattr(self, "command_controller", None)
                if _cc is None:
                    _cc = getattr(self, "_command_controller", None)
                if _cc is None:
                    _sf = getattr(self, "server_function", None)
                    if _sf:
                        _cc = getattr(_sf, "command_controller", None)
                        if _cc is None:
                            _cc = getattr(_sf, "_command_controller", None)
                if _cc is None:
                    return SetTaskMetricRsp(msg="bp::domain_randomization::failed::no_command_controller")
                if not hasattr(_cc, "_bp_domain_randomization_queue"):
                    _cc._bp_domain_randomization_queue = []
                _cc._bp_domain_randomization_queue.append(_bp_payload)
                return SetTaskMetricRsp(msg="bp::domain_randomization::ok")
            except Exception as _bp_err:
                return SetTaskMetricRsp(msg=f"bp::domain_randomization::error::{{_bp_err}}")
"""

    grpc_src, changed = _inject_into_set_task_metric(grpc_src, injected, DISPATCH_MARKER)
    if changed:
        try:
            with open(GRPC_PATH, "w") as f:
                f.write(grpc_src)
            print("[PATCH] domain_randomization: APPLIED to grpc_server.py")
        except OSError as exc:
            print(f"[PATCH] domain_randomization: grpc_server.py write failed: {exc}")
            return False
    else:
        print("[PATCH] domain_randomization: already applied to grpc_server.py")

    return True


if __name__ == "__main__":
    ok = apply()
    sys.exit(0 if ok else 1)
