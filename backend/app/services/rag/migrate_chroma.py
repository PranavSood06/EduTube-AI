"""Copy local persistent Chroma collections to the configured Chroma Cloud database.

Run once from ``backend`` with:
    uv run python -m app.services.rag.migrate_chroma

Use ``--dry-run`` to see the affected collections without writing to Chroma Cloud.
The operation is safe to repeat: records are upserted by their existing IDs.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import chromadb

from .store import VectorStore

logger = logging.getLogger(__name__)
DEFAULT_BATCH_SIZE = 100
DEFAULT_LOCAL_PATH = Path(__file__).resolve().parents[2] / "data" / "chroma"


def migrate_local_collections(
    local_path: Path = DEFAULT_LOCAL_PATH,
    *,
    batch_size: int = DEFAULT_BATCH_SIZE,
    dry_run: bool = False,
) -> dict[str, int]:
    """Upsert every local collection and its embeddings into Chroma Cloud."""
    if batch_size <= 0:
        raise ValueError("batch_size must be greater than 0")
    if not local_path.exists():
        raise FileNotFoundError(f"Local Chroma directory does not exist: {local_path}")

    local_client = chromadb.PersistentClient(path=str(local_path))
    local_collections = local_client.list_collections()
    result = {collection.name: collection.count() for collection in local_collections}

    if dry_run:
        return result

    cloud_client = VectorStore().client
    for local_collection in local_collections:
        name = local_collection.name
        total = result[name]
        cloud_collection = cloud_client.get_or_create_collection(name=name)
        logger.info("Migrating %d records from '%s'", total, name)

        for offset in range(0, total, batch_size):
            batch = local_collection.get(
                limit=batch_size,
                offset=offset,
                include=["documents", "embeddings", "metadatas"],
            )
            cloud_collection.upsert(
                ids=batch["ids"],
                documents=batch["documents"],
                embeddings=batch["embeddings"],
                metadatas=batch["metadatas"],
            )

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-path", type=Path, default=DEFAULT_LOCAL_PATH)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    migrated = migrate_local_collections(
        args.local_path,
        batch_size=args.batch_size,
        dry_run=args.dry_run,
    )
    action = "Found" if args.dry_run else "Migrated"
    for name, count in migrated.items():
        print(f"{action} {count} records: {name}")


if __name__ == "__main__":
    main()
