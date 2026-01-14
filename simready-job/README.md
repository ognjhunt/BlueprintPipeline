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
| `SIMREADY_PRODUCTION_MODE` | Enforce Gemini-backed physics estimates (no heuristic-only physics) | `false` |
| `SIMREADY_ALLOW_HEURISTIC_FALLBACK` | Allow heuristic-only physics estimation in CI/testing when Gemini is unavailable | `false` |
| `PIPELINE_ENV` | Marks environment (`production`/`prod` enables production mode) | None |

If these variables are unset, catalog calls are skipped and the job continues
with local metadata.

### Physics estimation fallback policy

Simready prefers Gemini-backed physics estimation. When Gemini is unavailable,
it will fall back to heuristic estimates. In **production mode**
(`SIMREADY_PRODUCTION_MODE=1` or `PIPELINE_ENV=production`), heuristic-only
physics is rejected and the job exits with an error. In production, Gemini
credentials must come from Secret Manager; env var fallbacks are rejected. For
CI/testing runs where Gemini is intentionally unavailable, set
`SIMREADY_ALLOW_HEURISTIC_FALLBACK=1` to acknowledge and allow the heuristic
fallback explicitly.
