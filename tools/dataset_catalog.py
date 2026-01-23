"""
Helpers for reading/writing dataset metadata from a central catalog.

The dataset catalog stores import/export summaries for LeRobot datasets
so downstream systems can discover datasets without scanning raw storage.
All interactions are best-effort: if Firestore is unavailable or config
is missing, publishes are skipped.
"""

from __future__ import annotations

import datetime
import importlib
import os
import sys
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Iterable, Mapping, Optional

DATASET_CATALOG_SCHEMA_VERSION = 1

firestore_spec = importlib.util.find_spec("google.cloud.firestore")
firestore = None

service_account_spec = importlib.util.find_spec("google.oauth2.service_account")
service_account = None


@dataclass
class DatasetDocument:
    """Canonical document stored in the dataset catalog collection."""

    dataset_id: str
    scene_id: str
    job_id: str
    schema_version: int = DATASET_CATALOG_SCHEMA_VERSION
    dataset_version: Optional[str] = None
    export_schema_version: Optional[str] = None
    export_format: Optional[str] = None
    robot_types: list[str] = field(default_factory=list)
    total_episodes: Optional[int] = None
    quality_summary: Dict[str, Any] = field(default_factory=dict)
    timestamps: Dict[str, Any] = field(default_factory=dict)
    storage_locations: Dict[str, Any] = field(default_factory=dict)
    extra_metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_firestore(cls, payload: Dict[str, Any]) -> "DatasetDocument":
        return cls(
            dataset_id=str(payload.get("dataset_id") or payload.get("id")),
            scene_id=str(payload.get("scene_id") or ""),
            job_id=str(payload.get("job_id") or ""),
            schema_version=int(payload.get("schema_version", DATASET_CATALOG_SCHEMA_VERSION)),
            dataset_version=payload.get("dataset_version"),
            export_schema_version=payload.get("export_schema_version"),
            export_format=payload.get("export_format"),
            robot_types=list(payload.get("robot_types", []) or []),
            total_episodes=payload.get("total_episodes"),
            quality_summary=payload.get("quality_summary", {}) or {},
            timestamps=payload.get("timestamps", {}) or {},
            storage_locations=payload.get("storage_locations", {}) or {},
            extra_metadata={
                k: v
                for k, v in payload.items()
                if k
                not in {
                    "dataset_id",
                    "id",
                    "scene_id",
                    "job_id",
                    "schema_version",
                    "dataset_version",
                    "export_schema_version",
                    "export_format",
                    "robot_types",
                    "total_episodes",
                    "quality_summary",
                    "timestamps",
                    "storage_locations",
                }
            },
        )

    def to_firestore(self) -> Dict[str, Any]:
        payload = {
            "dataset_id": self.dataset_id,
            "scene_id": self.scene_id,
            "job_id": self.job_id,
            "schema_version": self.schema_version,
            "dataset_version": self.dataset_version,
            "export_schema_version": self.export_schema_version,
            "export_format": self.export_format,
            "robot_types": self.robot_types,
            "total_episodes": self.total_episodes,
            "quality_summary": self.quality_summary,
            "timestamps": self.timestamps,
            "storage_locations": self.storage_locations,
        }
        payload.update(self.extra_metadata)
        return {k: v for k, v in payload.items() if v is not None}


def _iso_now() -> str:
    return datetime.datetime.utcnow().isoformat() + "Z"


def build_dataset_document(
    *,
    scene_id: str,
    job_id: str,
    import_manifest: Mapping[str, Any],
    dataset_info: Optional[Mapping[str, Any]] = None,
    firebase_summary: Optional[Mapping[str, Any]] = None,
    gcs_output_path: Optional[str] = None,
    robot_types: Optional[Iterable[str]] = None,
    document_id: Optional[str] = None,
) -> DatasetDocument:
    dataset_info = dataset_info or {}
    robot_types_list = list(robot_types or [])
    dataset_version = (
        dataset_info.get("schema_version")
        or dataset_info.get("format_version")
        or import_manifest.get("schema_version")
    )
    export_schema_version = dataset_info.get("export_schema_version")
    export_format = dataset_info.get("dataset_type") or "lerobot"
    episodes_payload = import_manifest.get("episodes", {}) or {}
    quality_payload = import_manifest.get("quality", {}) or {}
    total_episodes = dataset_info.get("total_episodes")
    if total_episodes is None:
        total_episodes = episodes_payload.get("passed_validation")

    quality_summary = {
        "average_score": dataset_info.get("average_quality_score", quality_payload.get("average_score")),
        "min_score": dataset_info.get("min_quality_score", quality_payload.get("min_score")),
        "max_score": dataset_info.get("max_quality_score", quality_payload.get("max_score")),
        "threshold": quality_payload.get("threshold"),
        "component_failed_episodes": quality_payload.get("component_failed_episodes"),
        "component_failure_counts": quality_payload.get("component_failure_counts"),
    }

    timestamps = {
        "generated_at": import_manifest.get("generated_at"),
        "converted_at": dataset_info.get("converted_at"),
        "upload_started_at": import_manifest.get("upload_started_at"),
        "upload_completed_at": import_manifest.get("upload_completed_at"),
        "status_updated_at": import_manifest.get("status_updated_at"),
        "catalog_updated_at": _iso_now(),
    }

    storage_locations = {
        "gcs_output_path": gcs_output_path or import_manifest.get("gcs_output_path"),
        "package_path": (import_manifest.get("package") or {}).get("path"),
        "import_manifest_path": import_manifest.get("import_manifest_path"),
        "firebase_prefix": (firebase_summary or {}).get("remote_prefix"),
        "firebase_bucket": os.getenv("FIREBASE_STORAGE_BUCKET"),
    }

    dataset_id = document_id or job_id
    return DatasetDocument(
        dataset_id=dataset_id,
        scene_id=scene_id,
        job_id=job_id,
        dataset_version=str(dataset_version) if dataset_version is not None else None,
        export_schema_version=export_schema_version,
        export_format=export_format,
        robot_types=robot_types_list,
        total_episodes=int(total_episodes) if total_episodes is not None else None,
        quality_summary={k: v for k, v in quality_summary.items() if v is not None},
        timestamps={k: v for k, v in timestamps.items() if v is not None},
        storage_locations={k: v for k, v in storage_locations.items() if v is not None},
        extra_metadata={
            "dataset_info": dict(dataset_info) if dataset_info else None,
            "import_status": import_manifest.get("import_status"),
            "success": import_manifest.get("success"),
        },
    )


@dataclass
class DatasetCatalogConfig:
    """Lightweight config holder for the dataset catalog client."""

    project_id: Optional[str] = None
    collection: str = "datasets"
    credentials_path: Optional[str] = None
    emulator_host: Optional[str] = None

    @classmethod
    def from_env(cls) -> "DatasetCatalogConfig":
        return cls(
            project_id=os.getenv("DATASET_CATALOG_PROJECT"),
            collection=os.getenv("DATASET_CATALOG_COLLECTION", "datasets"),
            credentials_path=os.getenv("DATASET_CATALOG_CREDENTIALS"),
            emulator_host=os.getenv("DATASET_CATALOG_EMULATOR_HOST"),
        )


class DatasetCatalogClient:
    """Small helper for dataset catalog publishes."""

    def __init__(self, config: Optional[DatasetCatalogConfig] = None) -> None:
        self.config = config or DatasetCatalogConfig.from_env()
        self._client = None

    def _ensure_client(self):
        if self._client is not None:
            return self._client

        global firestore
        global service_account

        if firestore is None and firestore_spec:
            firestore = importlib.import_module("google.cloud.firestore")

        if (
            service_account is None
            and service_account_spec
            and self.config.credentials_path
        ):
            service_account = importlib.import_module("google.oauth2.service_account")

        if firestore is None:
            return None

        kwargs: Dict[str, Any] = {}
        if self.config.emulator_host:
            os.environ.setdefault("FIRESTORE_EMULATOR_HOST", self.config.emulator_host)

        if self.config.credentials_path and service_account is not None:
            try:
                creds_cls = getattr(service_account, "Credentials", None)
                if creds_cls:
                    kwargs["credentials"] = creds_cls.from_service_account_file(
                        self.config.credentials_path
                    )
            except Exception as exc:  # pragma: no cover - credential parsing errors
                print(f"[DATASET_CATALOG] WARNING: failed to load credentials: {exc}", file=sys.stderr)

        try:
            self._client = firestore.Client(project=self.config.project_id, **kwargs)
        except Exception as exc:  # pragma: no cover - client creation errors
            print(
                f"[DATASET_CATALOG] WARNING: failed to initialize Firestore client: {exc}",
                file=sys.stderr,
            )
            self._client = None

        return self._client

    def _collection(self):
        client = self._ensure_client()
        if client is None:
            return None
        return client.collection(self.config.collection)

    def upsert_dataset_document(self, dataset: DatasetDocument) -> None:
        coll = self._collection()
        if coll is None:
            return

        try:
            coll.document(dataset.dataset_id).set(dataset.to_firestore(), merge=True)
        except Exception as exc:  # pragma: no cover - network/firestore errors
            print(
                f"[DATASET_CATALOG] WARNING: failed to upsert dataset {dataset.dataset_id}: {exc}",
                file=sys.stderr,
            )
