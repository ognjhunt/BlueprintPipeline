# Issue Spec: Desk Drawer Articulation Not Detected

**Priority:** Medium (non-blocking, but limits manipulation task diversity)
**Scene:** Bedroom (`ChIJBc5E5wTjrIkRKrrWR_meHbc`)
**Object:** `obj_4` (`wooden_desk`) — console/entry table with 2 visible drawers

## Problem

The wooden desk in the bedroom scene has visible drawers (see reference photo `basicbedroom.jpg`), but it was labeled as `wooden_desk` in Stage 1 — a flat category with no articulation hint. The interactive/articulation step (Particulate) was never run, so the desk is a static solid mesh in sim with no openable drawers.

## Root Causes (3 layers)

### 1. Stage 1 — Gemini labeling misses articulation
Gemini auto-labeled the object `wooden_desk` instead of something like `desk_with_drawers` or `console_table_with_drawers`. The labeling prompt in `tools/regen3d_runner/config_template.yaml` doesn't explicitly ask Gemini to flag articulated parts (drawers, doors, hinges).

### 2. No interactive step in local pipeline
`tools/run_local_pipeline.py` has no `PipelineStep.INTERACTIVE` at all. The interactive step only exists in the cloud orchestrator:
- `workflows/interactive-pipeline.yaml` → triggers `interactive-job/run_interactive_assets.py`
- Uses the **Particulate** service (Cloud Run) to detect articulated parts on GLB meshes
- Falls back to **heuristic articulation detection** if Particulate is unavailable
- Outputs URDFs + segmented meshes to `assets/interactive/`

The local pipeline skips from `simready → usd → replicator` with no articulation detection in between.

### 3. 3D mesh is a single solid blob
Hunyuan3D-2 generates a unified mesh from a single reference image. Even if Particulate ran on it, the drawers aren't separate geometry in the GLB, so heuristic articulation detection won't find them.

## Relevant Files

| File | Role |
|------|------|
| `interactive-job/run_interactive_assets.py` | Cloud interactive step (Particulate + heuristic fallback) |
| `workflows/interactive-pipeline.yaml` | Cloud Run workflow definition |
| `tools/regen3d_runner/config_template.yaml` | Gemini labeling prompts (lines ~20-80) |
| `tools/run_local_pipeline.py` | Local pipeline — **no INTERACTIVE step exists** |
| `scenes/ChIJBc5E5wTjrIkRKrrWR_meHbc/assets/` | Current scene assets (solid desk GLB) |

## Fix Options

### Option A — Improve labeling (minimal effort, ~30 min)
Update the Gemini labeling prompt in Stage 1 (`config_template.yaml`) to explicitly ask:

> "If the object has articulated parts (drawers, doors, hinges, knobs, lids), include this in the label (e.g., `desk_with_drawers`, `cabinet_with_doors`). Also output an `articulation_hints` field listing the articulated parts and their joint types (prismatic for drawers, revolute for doors/lids)."

Store articulation hints in the object metadata JSON. This doesn't fix the mesh geometry, but makes downstream steps aware that articulation processing is needed.

### Option B — Add local INTERACTIVE step (medium effort, ~2-4 hrs)
Port `interactive-job/run_interactive_assets.py`'s **heuristic articulation detection** into the local pipeline as a new `PipelineStep.INTERACTIVE`. This would:

1. Run between `simready` and `usd` steps
2. Use the heuristic fallback (no Particulate service needed) to analyze GLB geometry for potential drawer/door regions
3. Generate URDF joints for detected articulated parts
4. Output to `assets/interactive/`

**Limitation:** Only works if the mesh has separate geometry for drawers. With current Hunyuan3D single-image reconstruction, drawers are baked into the surface.

### Option C — Multi-view reconstruction for articulated objects (high effort, ~1-2 days)
For objects flagged as articulated in labeling:

1. Generate multiple reference views showing drawers open/closed (via Gemini image gen or manual photos)
2. Use a mesh segmentation step to separate drawer parts from the desk body
3. Generate proper URDF with prismatic joints for each drawer
4. Re-run SimReady physics on segmented parts

This is the most complete solution but requires significant new pipeline work.

## Recommended Approach

1. **Start with Option A** — it's ~10 lines of prompt changes and immediately improves metadata quality for all future scenes.
2. **Then Option B** — port heuristic detection as a local pipeline step. Even if it can't segment solid meshes, it prepares the pipeline for better 3D backends.
3. **Skip Option C** unless drawer manipulation is a critical training task.

## Current Impact

- The desk behaves as a static solid in simulation
- Cannot train drawer-open/close manipulation tasks on this object
- Other bedroom objects (bed frame, books, nightstand, potted cactus) are unaffected — they don't have articulated parts
- Kitchen scenes would be more severely impacted (cabinet doors, fridge, dishwasher, oven)

## Test Criteria

- [ ] Gemini labels objects with drawers/doors as `{type}_with_{articulation}`
- [ ] Object metadata includes `articulation_hints` field
- [ ] Local pipeline has `PipelineStep.INTERACTIVE` that runs heuristic detection
- [ ] URDF generated for desk with 2 prismatic drawer joints
- [ ] Drawer joints have correct axis (typically Y or Z for pull-out) and travel limits
