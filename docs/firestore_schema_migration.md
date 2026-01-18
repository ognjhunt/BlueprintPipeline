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

## Migration Process (Placeholder)

1. **Identify target collections**: List collections that require updates.
2. **Back up data**: Export the collections before applying changes.
3. **Run migration script**: Apply version-specific transformations.
4. **Verify**: Sample documents to confirm `schema_version` and new fields.
5. **Deploy**: Roll out application changes that depend on the new schema.

## Placeholder Script

```python
"""Placeholder Firestore schema migration script.

Fill in the transformation logic when a new schema version is introduced.
"""

from __future__ import annotations

from typing import Iterable

from google.cloud import firestore


def migrate_collection(
    client: firestore.Client,
    collection_name: str,
    from_version: int,
    to_version: int,
) -> None:
    """Migrate documents in a collection from one schema version to another."""
    collection = client.collection(collection_name)
    batch = client.batch()
    updated = 0

    for doc in collection.stream():
        data = doc.to_dict() or {}
        if data.get("schema_version") != from_version:
            continue

        # TODO: apply transformation logic when bumping schema versions.
        data["schema_version"] = to_version

        batch.set(doc.reference, data, merge=True)
        updated += 1

        if updated % 400 == 0:
            batch.commit()
            batch = client.batch()

    if updated % 400 != 0:
        batch.commit()


def main(collections: Iterable[str], from_version: int, to_version: int) -> None:
    client = firestore.Client()
    for collection in collections:
        migrate_collection(client, collection, from_version, to_version)


if __name__ == "__main__":
    main(
        collections=["customers", "scenes", "feature_flags", "usage_tracking", "assets"],
        from_version=1,
        to_version=2,
    )
```
