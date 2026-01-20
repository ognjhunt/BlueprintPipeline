from __future__ import annotations

from collections import defaultdict
from typing import Any, Callable, Dict, List, MutableMapping, Tuple

Transform = Callable[[MutableMapping[str, Any]], MutableMapping[str, Any]]

MIGRATIONS: Dict[str, Dict[Tuple[int, int], Transform]] = defaultdict(dict)


class MigrationError(ValueError):
    """Raised when requested migrations cannot be satisfied."""


def _migration_steps(
    collection: str,
    from_version: int,
    to_version: int,
) -> List[Transform]:
    if from_version > to_version:
        raise MigrationError("from_version cannot be greater than to_version")
    if from_version == to_version:
        return []

    steps: List[Transform] = []
    migrations = MIGRATIONS.get(collection, {})
    for version in range(from_version, to_version):
        key = (version, version + 1)
        if key not in migrations:
            raise MigrationError(
                f"Missing migration for {collection} from {version} to {version + 1}"
            )
        steps.append(migrations[key])
    return steps


def migrate_collection(
    client: Any,
    collection_name: str,
    from_version: int,
    to_version: int,
    batch_size: int = 400,
) -> int:
    """Apply Firestore schema migrations for a collection.

    Returns the number of documents updated.
    """
    steps = _migration_steps(collection_name, from_version, to_version)
    if not steps:
        return 0

    collection = client.collection(collection_name)
    batch = client.batch()
    updated = 0

    for doc in collection.stream():
        data = doc.to_dict() or {}
        if data.get("schema_version") != from_version:
            continue

        updated_data: MutableMapping[str, Any] = dict(data)
        for transform in steps:
            updated_data = transform(updated_data)
        updated_data["schema_version"] = to_version

        batch.set(doc.reference, dict(updated_data), merge=True)
        updated += 1

        if updated % batch_size == 0:
            batch.commit()
            batch = client.batch()

    if updated % batch_size != 0:
        batch.commit()

    return updated
