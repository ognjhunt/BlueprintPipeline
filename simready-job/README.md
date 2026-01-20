# simready-job

## Metadata catalog integration

Simready now participates in a central asset-metadata catalog (Firestore
collection by default) so bounds, physics hints, and materials can be reused
across jobs.

### Lookup order

1. **Catalog**: query by `asset_id` (object id) or `asset_path`.
2. **Declared metadata path**: `metadata_path` on the object.
3. **Next to the asset**: `metadata.json` beside the mesh.
4. **Static fallback**: `assets/static/obj_<id>/metadata.json`.

The catalog step is best-effort; if credentials or connectivity are missing the
script falls back to local files for offline runs.

### Publishing fresh metadata

When simready computes bounds or Gemini-backed physics estimates, it writes
them into `simready.usda` **and** publishes the same payload back to the
catalog. Future runs of simready and `usd-assembly-job` will then pick up the
cached values without re-querying Gemini.

The published document includes:

- `mesh_bounds.export.size` / `center` / `volume_m3` / `source`
- `physics` (dynamic flag, friction, restitution, mass, material, robotics
  hints)
- `asset_path` to correlate the record with GCS objects

### Environment variables

| Variable | Purpose | Default |
| --- | --- | --- |
| `ASSET_CATALOG_PROJECT` | Firestore project id | None (required for managed catalog) |
| `ASSET_CATALOG_COLLECTION` | Collection name for metadata | `asset_metadata` |
| `ASSET_CATALOG_CREDENTIALS` | Path to service account JSON | None |
| `ASSET_CATALOG_EMULATOR_HOST` | Host for emulator/local testing | None |
| `SIMREADY_PRODUCTION_MODE` | Enforce production physics rules (requires Gemini or deterministic mode) | `false` |
| `SIMREADY_PHYSICS_MODE` | Physics estimation mode (`auto`/`gemini`/`deterministic`) | `auto` |
| `SIMREADY_ALLOW_DETERMINISTIC_PHYSICS` | Allow deterministic (LLM-free) physics estimation when Gemini is unavailable | `false` |
| `SIMREADY_ALLOW_HEURISTIC_FALLBACK` | Allow heuristic-only physics estimation in CI/testing when Gemini and deterministic modes are unavailable | `false` |
| `SIMREADY_FALLBACK_MIN_COVERAGE` | Minimum coverage ratio for deterministic fallback physics (0-1) | `0.6` |
| `SIMREADY_NON_LLM_MIN_QUALITY` | Minimum quality ratio for non-LLM physics checks (0-1) | `0.85` |
| `PIPELINE_ENV` | Marks environment (`production`/`prod` enables production mode) | None |

If these variables are unset, catalog calls are skipped and the job continues
with local metadata.

### Physics estimation fallback policy

Simready prefers Gemini-backed physics estimation. When Gemini is unavailable,
you can enable **deterministic physics** (LLM-free, metadata/material-driven)
by setting `SIMREADY_PHYSICS_MODE=deterministic` or `SIMREADY_ALLOW_DETERMINISTIC_PHYSICS=1`.
In **production mode** (`SIMREADY_PRODUCTION_MODE=1` or `PIPELINE_ENV=production`),
Gemini is required unless deterministic mode is explicitly enabled; heuristic-only
physics is rejected and the job exits with an error. In production, Gemini
credentials must come from Secret Manager; env var fallbacks are rejected.

When deterministic fallback is used, simready enforces a minimum physics coverage
ratio (`SIMREADY_FALLBACK_MIN_COVERAGE`, default 0.6). Coverage counts objects
that were estimated with metadata/material priors; if too many assets fall back
to generic defaults the job fails, prompting you to improve metadata coverage.

Deterministic (non-LLM) physics also enforces a minimum quality ratio
(`SIMREADY_NON_LLM_MIN_QUALITY`, default 0.85). Quality checks validate that
mass, density, friction, restitution, and collision shape remain within
simulation-safe bounds. If the quality ratio is too low, the job fails to
prompt richer metadata or tuned material priors.

### Physics validation notes

When simready validates physics properties, it now prefers USD-derived mass
properties or mesh-based inertia calculations (via `trimesh`, when available).
If mesh computation is unavailable or fails, the validator falls back to a box
approximation and records the provenance in the validation output via
`inertia_source` (`usd`, `mesh`, or `box_approx`) so operators can spot
approximations in reports.

### Production modes (free vs. paid)

**Free production (deterministic, no Gemini)**:
- Required flags: `SIMREADY_PRODUCTION_MODE=1` (or `PIPELINE_ENV=production`) and
  `SIMREADY_PHYSICS_MODE=deterministic`.
- Enforces `SIMREADY_FALLBACK_MIN_COVERAGE` and `SIMREADY_NON_LLM_MIN_QUALITY`.

**Paid production (Gemini-backed)**:
- Required flags: `SIMREADY_PRODUCTION_MODE=1` (or `PIPELINE_ENV=production`) and
  either `SIMREADY_PHYSICS_MODE=gemini` or `SIMREADY_PHYSICS_MODE=auto` with Gemini
  credentials available.
- Configure `gemini-api-key` in Secret Manager (production rejects env var fallbacks).
