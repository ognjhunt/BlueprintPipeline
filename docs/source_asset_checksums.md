# Source asset checksums

## Entry points that write source assets

The pipeline writes source assets (and records checksums) at these entry points:

- `scene-generation-job/generate_scene_images.py` writes `scenes/<scene_id>/source_image.png` and a matching `source_checksums.json` for the generated image.
- `regen3d-job/regen3d_adapter_job.py` copies reconstructed assets into `scenes/<scene_id>/assets/` and writes `source_checksums.json` alongside the assets.

## Checksum file format

Each `source_checksums.json` contains SHA-256 digests and file sizes for the files that were first staged into the scene folder or assets directory:

```json
{
  "schema_version": "1.0",
  "generated_at": "2025-01-01T00:00:00Z",
  "root": "/mnt/gcs/scenes/example/assets",
  "files": {
    "obj_1/asset.glb": {
      "sha256": "...",
      "size_bytes": 123456
    }
  }
}
```

## Downstream verification expectations

Downstream jobs verify these checksums before processing:

- `episode-generation-job/generate_episodes.py` validates `scenes/<scene_id>/assets/source_checksums.json` before loading the scene manifest.
- `genie-sim-export-job/export_to_geniesim.py` validates the same checksum file before exporting to Genie Sim and includes the metadata in `export_manifest.json`.

## Remediation steps when checksums fail

1. **Re-stage assets**: Re-run the upstream job that produced the assets (`scene-generation-job` for source images or `regen3d-job` for reconstructed assets) to regenerate the files and `source_checksums.json`.
2. **Confirm storage integrity**: If the checksum mismatches persist, verify the storage layer (GCS bucket, local staging mount) for partial or corrupted uploads before retrying.
3. **Re-run downstream job**: After the checksum file is restored and validated, re-run the downstream job that failed (`episode-generation-job` or `genie-sim-export-job`).

## Manifest integration

- Genie Sim exports include the `source_assets` section in `geniesim/export_manifest.json`, pointing to the checksum metadata for traceability.
- Genie Sim imports propagate any available `source_assets` metadata into `import_manifest.json` for auditing.
