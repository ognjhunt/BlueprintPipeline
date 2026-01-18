"""Firestore-backed approval store implementation."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from .quality_gate import ApprovalRequest, ApprovalStatus

firestore_spec = importlib.util.find_spec("google.cloud.firestore")
if firestore_spec is None:  # pragma: no cover - depends on optional dependency
    firestore = None
    HAVE_FIRESTORE = False
else:
    from google.cloud import firestore
    HAVE_FIRESTORE = True


class FirestoreApprovalStore:
    """Store approval requests in Firestore."""

    def __init__(
        self,
        scene_id: str,
        collection: str,
        client: Optional["firestore.Client"] = None,
    ) -> None:
        if not HAVE_FIRESTORE:
            raise RuntimeError(
                "google-cloud-firestore is required for FirestoreApprovalStore. "
                "Install google-cloud-firestore or switch to filesystem backend."
            )
        self.scene_id = scene_id
        self.collection_name = collection
        self.client = client or firestore.Client()
        self.collection = self.client.collection(collection)
        self.audit_collection = self.client.collection(f"{collection}_audit")

    def save_request(self, request: ApprovalRequest) -> None:
        data = request.to_dict()
        data.setdefault("scene_id", self.scene_id)
        self.collection.document(request.request_id).set(data)

    def load_request(self, request_id: str) -> Optional[ApprovalRequest]:
        doc = self.collection.document(request_id).get()
        if not doc.exists:
            return None
        data = doc.to_dict() or {}
        if "status" not in data:
            return None
        return ApprovalRequest.from_dict(data)

    def list_requests(self, status: Optional[ApprovalStatus] = None) -> List[ApprovalRequest]:
        query = self.collection.where("scene_id", "==", self.scene_id)
        if status:
            query = query.where("status", "==", status.value)
        results: List[ApprovalRequest] = []
        for doc in query.stream():
            data = doc.to_dict()
            if not data or "status" not in data:
                continue
            results.append(ApprovalRequest.from_dict(data))
        return results

    def write_audit_entry(self, audit_entry: Dict[str, Any]) -> None:
        payload = dict(audit_entry)
        payload.setdefault("scene_id", self.scene_id)
        self.audit_collection.add(payload)

    def migrate_from_filesystem(self, base_dir: Path) -> int:
        source_dir = base_dir / self.scene_id
        if not source_dir.exists():
            return 0
        migrated = 0
        for path in source_dir.glob("*.json"):
            with open(path) as f:
                data = json.load(f)
            self.save_request(ApprovalRequest.from_dict(data))
            migrated += 1
        audit_log = source_dir / "audit" / "audit.log.jsonl"
        if audit_log.exists():
            with open(audit_log) as f:
                for line in f:
                    if not line.strip():
                        continue
                    self.write_audit_entry(json.loads(line))
        return migrated
