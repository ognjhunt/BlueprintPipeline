# interactive-job

## Purpose / scope
Runs interactive asset processing for the pipeline.

This job produces **simulation-ready articulated assets** (URDF + referenced meshes)
under `assets/interactive/obj_{id}/` for downstream `simready-job` and `usd-assembly-job`.

The implementation supports a **multi-backend** articulation flow:
- `infinigen`: category-conditioned articulated asset generation (when supported).
- `physx_anything`: image-conditioned articulated asset generation.
- `particulate`: mesh-conditioned articulation inference.

An automatic **critic** validates the produced URDF in simulation (PyBullet) and can
retry with another backend on failure (self-collision, jitter, limits).

## Primary entrypoints
- `run_interactive_assets.py` job entrypoint.
- `heuristic_articulation.py` heuristics support module.
- `Dockerfile` container image definition.

## Required inputs / outputs
- **Inputs:** asset layouts or scene descriptions used by `run_interactive_assets.py`.
- **Outputs:** processed interactive assets and associated metadata.

## Key environment variables
- `PIPELINE_ENV`: set to `production` to enforce placeholder disallow and production guardrails.
- `DISALLOW_PLACEHOLDER_URDF`: set to `true` to fail when placeholder URDFs would be generated (enabled automatically in production).
- Variables defining input/output locations and pipeline configuration.

### Backend selection
- `ARTICULATION_BACKEND`: `auto` (default) or `infinigen|physx_anything|particulate`.
- `ARTICULATION_RETRY_ENABLED`: retry other backends when the critic fails (default: true).
- `ARTICULATION_CRITIC_ENABLED`: enable PyBullet joint sweep + self-collision checks (default: true when installed).

### Infinigen backend (optional)
- `INFINIGEN_ENABLED`: enable `infinigen` in `auto` selection.
- `INFINIGEN_ENDPOINT`: optional HTTP endpoint for a generator service (POST JSON).
- `INFINIGEN_ROOT`: optional local checkout (must contain `scripts/spawn_asset.py`).

### PhysX-Anything backend (optional)
- `PHYSX_ANYTHING_ENABLED`: enable `physx_anything` in `auto` selection.
- `PHYSX_ANYTHING_ENDPOINT`: optional HTTP endpoint for a generator service (POST JSON).
- `PHYSX_ANYTHING_ROOT`: optional local checkout (must contain `1_vlm_demo.py` etc.).

### Particulate backend
- `PARTICULATE_ENDPOINT`: Particulate service URL.
- `PARTICULATE_MODE`: `remote` (default) | `local` | `mock` | `skip`.

## How to run locally
- Build the container (from repo root): `docker build -f interactive-job/Dockerfile -t blueprint-interactive:latest .`
- Run the job (in-repo): `python interactive-job/run_interactive_assets.py`
