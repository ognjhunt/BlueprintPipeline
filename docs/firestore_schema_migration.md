# Firestore Schema Migration Notes

This document tracks how to migrate Firestore documents between schema versions.

## Current Versions

| Collection | Version |
|-----------|---------|
| `customers` | 1 |
| `scenes` | 1 |
| `feature_flags` | 1 |
| `usage_tracking` | 1 |
| `assets` (asset catalog) | 1 |

## Migration Process

1. **Identify target collections**: List collections that require updates.
2. **Back up data**: Export the collections before applying changes.
3. **Register migration transforms**: Add version-to-version transforms in
   `tools/firestore/migrations.py`.
4. **Run migration script**: Apply version-specific transformations with
   `scripts/firestore_migrate.py`.
5. **Verify**: Sample documents to confirm `schema_version` and new fields.
6. **Deploy**: Roll out application changes that depend on the new schema.

## Adding a New Migration Transform

Edit `tools/firestore/migrations.py` and register a transform for a collection
and version hop. The registry keys are `(from_version, to_version)` tuples.

```python
from tools.firestore.migrations import MIGRATIONS


def migrate_scene_1_to_2(data: dict) -> dict:
    data["new_field"] = data.pop("old_field", None)
    return data


MIGRATIONS["scenes"][(1, 2)] = migrate_scene_1_to_2
```

## Running a Migration

```bash
python scripts/firestore_migrate.py scenes 1 2
```

The script logs the registered steps and updates documents in batch commits.
