from types import SimpleNamespace

import tools.dataset_catalog as dataset_catalog


def test_build_dataset_document_payload():
    import_manifest = {
        "schema_version": "manifest-v1",
        "generated_at": "2025-01-10T10:00:00Z",
        "gcs_output_path": "gs://bucket/scenes/scene-1",
        "import_status": "completed",
        "success": True,
        "upload_started_at": "2025-01-10T10:05:00Z",
        "upload_completed_at": "2025-01-10T10:10:00Z",
        "episodes": {"passed_validation": 12},
        "quality": {
            "average_score": 0.95,
            "min_score": 0.9,
            "max_score": 0.99,
            "threshold": 0.9,
            "component_failed_episodes": 0,
            "component_failure_counts": {"vision": 0},
        },
        "package": {"path": "lerobot_bundle_job_123.tar.gz"},
    }
    dataset_info = {
        "schema_version": "1.0.0",
        "dataset_type": "lerobot",
        "export_schema_version": "2024-05-01",
        "total_episodes": 12,
        "average_quality_score": 0.96,
        "converted_at": "2025-01-10T09:55:00Z",
    }
    firebase_summary = {"remote_prefix": "datasets/scene-1"}

    document = dataset_catalog.build_dataset_document(
        scene_id="scene-1",
        job_id="job-123",
        import_manifest=import_manifest,
        dataset_info=dataset_info,
        firebase_summary=firebase_summary,
        gcs_output_path="gs://bucket/scenes/scene-1",
        robot_types=["franka"],
        document_id="job-123-franka",
    )

    payload = document.to_firestore()
    assert payload["dataset_id"] == "job-123-franka"
    assert payload["scene_id"] == "scene-1"
    assert payload["job_id"] == "job-123"
    assert payload["dataset_version"] == "1.0.0"
    assert payload["export_schema_version"] == "2024-05-01"
    assert payload["export_format"] == "lerobot"
    assert payload["robot_types"] == ["franka"]
    assert payload["total_episodes"] == 12
    assert payload["quality_summary"]["average_score"] == 0.96
    assert payload["storage_locations"]["firebase_prefix"] == "datasets/scene-1"
    assert payload["storage_locations"]["gcs_output_path"] == "gs://bucket/scenes/scene-1"


def test_dataset_catalog_client_upsert(monkeypatch):
    captured = {}

    class FakeDoc:
        def __init__(self, doc_id):
            self.doc_id = doc_id

        def set(self, payload, merge=False):
            captured["doc_id"] = self.doc_id
            captured["payload"] = payload
            captured["merge"] = merge

    class FakeCollection:
        def document(self, doc_id):
            return FakeDoc(doc_id)

    class FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        def collection(self, name):
            captured["collection"] = name
            return FakeCollection()

    monkeypatch.setattr(dataset_catalog, "firestore", SimpleNamespace(Client=FakeClient))

    client = dataset_catalog.DatasetCatalogClient(
        dataset_catalog.DatasetCatalogConfig(collection="datasets-test")
    )
    dataset = dataset_catalog.DatasetDocument(
        dataset_id="job-1",
        scene_id="scene-1",
        job_id="job-1",
        export_format="lerobot",
    )
    client.upsert_dataset_document(dataset)

    assert captured["collection"] == "datasets-test"
    assert captured["doc_id"] == "job-1"
    assert captured["payload"]["export_format"] == "lerobot"
    assert captured["merge"] is True
