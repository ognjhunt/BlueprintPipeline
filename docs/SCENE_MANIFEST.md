# Scene Manifest Schema

The canonical `scene_manifest.json` captures everything downstream jobs need to
assemble and simulate a scene: object roles, mesh references, transforms,
physics/semantic hints, and background mesh metadata. The schema is defined in
[`tools/scene_manifest/manifest_schema.json`](../tools/scene_manifest/manifest_schema.json).

Key fields:

- `version`: Required schema version tag (missing version fails QA validation).
- `scene_id`: Required identifier for the scene.
- `scene`: Required block with `coordinate_frame` and `meters_per_unit`.
- `background`: Background mesh reference and semantics.
- `objects[]`: One entry per asset with:
  - `id`: Stable object identifier (string or int).
  - `sim_role`: Simulator role (`static`, `manipulable_object`, `articulated_furniture`, `articulated_appliance`, `scene_shell`, `background`, `unknown`).
  - `asset`: Mesh reference (`path`, optional `format`, optional source IDs).
  - `transform`: Translation/rotation/scale when known.
  - `articulation`: Joint hints (e.g., PhysX endpoint, joint axes).
  - `physics`: Physical hints (mass, gravity, collision strategy, materials).
  - `semantics`: Category/description/tags.
  - `placement`: 2D polygons or approximate locations for layout synthesis.
  - `asset_generation`: How the asset was produced (pipeline + inputs/outputs).
  - `source`: Raw source snippets (scene_assets, inventory entries, BlueprintRecipe rows).

Validation and conversion utilities live in
[`tools/scene_manifest/validate_manifest.py`](../tools/scene_manifest/validate_manifest.py).

## Examples

### Gemini pipeline (scene_assets.json + inventory.json)

```bash
python tools/scene_manifest/validate_manifest.py from_scene_assets \
  scenes/<sceneId>/assets/scene_assets.json \
  --inventory scenes/<sceneId>/seg/inventory.json \
  --output scenes/<sceneId>/assets/scene_manifest.json
```

Resulting manifest excerpt:

```json
{
  "version": "1.0.0",
  "scene_id": "kitchen_123",
  "scene": {
    "coordinate_frame": "y_up",
    "meters_per_unit": 1.0
  },
  "background": {
    "mesh": {"path": "scenes/kitchen_123/assets/obj_scene_background/model.glb"},
    "semantics": {"category": "scene_background"}
  },
  "objects": [
    {
      "id": "12",
      "sim_role": "manipulable_object",
      "asset": {
        "path": "scenes/kitchen_123/assets/obj_12/asset.glb",
        "format": "glb",
        "scene_asset_id": "12",
        "inventory_id": "12"
      },
      "placement": {
        "polygon": [[0.12, 0.33], [0.24, 0.33], [0.24, 0.49], [0.12, 0.49]],
        "approx_location": "near the left countertop"
      },
      "semantics": {
        "category": "microwave",
        "short_description": "stainless steel microwave"
      },
      "asset_generation": {
        "pipeline": "ultrashape",
        "inputs": {
          "multiview_dir": "scenes/kitchen_123/multiview/obj_12",
          "crop_path": "scenes/kitchen_123/multiview/obj_12/view_0.png"
        },
        "output": "scenes/kitchen_123/assets/obj_12/asset.glb"
      }
    }
  ]
}
```

### BlueprintRecipe pipeline

```bash
python tools/scene_manifest/validate_manifest.py from_blueprint_recipe \
  scenes/<sceneId>/plan/scene_plan.json \
  scenes/<sceneId>/plan/matched_assets.json \
  --output scenes/<sceneId>/assets/scene_manifest.json
```

Example entry:

```json
{
  "version": "1.0.0",
  "scene_id": "loft_42",
  "scene": {
    "coordinate_frame": "y_up",
    "meters_per_unit": 1.0
  },
  "objects": [
    {
      "id": "coffee_table",
      "sim_role": "static",
      "asset": {
        "path": "scenes/loft_42/assets/coffee_table.usdz",
        "format": "usdz",
        "source": "blueprint_recipe"
      },
      "transform": {
        "translation": [1.2, 0.0, -0.3],
        "rotation": [0, 0, 0, 1],
        "scale": [1, 1, 1]
      },
      "semantics": {"category": "coffee table", "tags": ["living_room"]},
      "placement": {"approx_location": "center of living room"},
      "physics": {"dynamic": false},
      "source": {
        "blueprint_recipe": {"id": "coffee_table", "sim_role": "static"},
        "matched_asset": {"asset_path": "scenes/loft_42/assets/coffee_table.usdz"}
      }
    }
  ]
}
```

Use `validate` to confirm schema compliance:

```bash
python tools/scene_manifest/validate_manifest.py validate scenes/<sceneId>/assets/scene_manifest.json
```
