#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging

from google.cloud import firestore

from tools.firestore.migrations import MIGRATIONS, migrate_collection


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Firestore schema migrations.")
    parser.add_argument("collection", help="Firestore collection name to migrate")
    parser.add_argument("from_version", type=int, help="Current schema version")
    parser.add_argument("to_version", type=int, help="Target schema version")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=400,
        help="Number of writes per commit batch",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = _parse_args()

    migrations = MIGRATIONS.get(args.collection, {})
    logging.info(
        "Preparing migration for %s from v%d to v%d with steps: %s",
        args.collection,
        args.from_version,
        args.to_version,
        sorted(migrations),
    )

    client = firestore.Client()
    updated = migrate_collection(
        client,
        args.collection,
        args.from_version,
        args.to_version,
        batch_size=args.batch_size,
    )
    logging.info("Updated %d documents in %s", updated, args.collection)


if __name__ == "__main__":
    main()
