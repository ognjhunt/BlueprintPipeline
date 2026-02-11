def test_migrate_legacy_dataset_info_payload(add_repo_to_path) -> None:
    from tools import schema_migrations

    legacy_payload = {
        "format": "lerobot",
        "export_format": "parquet",
        "version": "0.1.0",
        "episodes": 2,
        "skipped": 1,
        "total_frames": 120,
        "exported_at": "2024-01-01T00:00:00Z",
        "lerobot_info": "meta/info.json",
    }

    result = schema_migrations.migrate_dataset_info_payload(legacy_payload)

    assert result.payload["schema_version"] == schema_migrations.DATASET_INFO_SCHEMA_VERSION
    assert result.payload["total_episodes"] == 2
    assert result.payload["skipped_episodes"] == 1
    assert result.payload["episodes"] == []
    assert "legacy_payload" in result.payload
    assert "migrate-legacy-dataset-info-version-field" in result.applied_steps


def test_migrate_legacy_import_manifest_payload(add_repo_to_path) -> None:
    from tools import schema_migrations

    legacy_manifest = {
        "schema_version": "0.1.0",
        "generated_at": "2024-01-01T00:00:00Z",
        "checksums_path": "checksums.json",
        "episodes": {"downloaded": 2},
    }

    result = schema_migrations.migrate_import_manifest_payload(legacy_manifest)

    assert result.payload["schema_version"] == schema_migrations.MANIFEST_SCHEMA_VERSION
    assert "migrate-import-manifest-0.1.0-to-1.3" in result.applied_steps
    assert result.payload["scene_id"] == "unknown"
    assert result.payload["run_id"] == result.payload.get("job_id")
    assert result.payload["status"] == "unknown"
    assert result.payload["import_status"] == "unknown"
    assert result.payload["robot_types"] == []


def test_migrate_import_manifest_1_2_to_1_3(add_repo_to_path) -> None:
    from tools import schema_migrations

    payload = {
        "schema_version": "1.2",
        "job_id": "job-123",
        "provenance": {"scene_id": "scene-abc"},
        "robots": [{"robot_type": "franka"}, {"robot_type": "ur5e"}],
        "file_inventory": [{"path": "recordings/task_0_ep0000.json"}],
    }

    result = schema_migrations.migrate_import_manifest_payload(payload)

    assert result.payload["schema_version"] == schema_migrations.MANIFEST_SCHEMA_VERSION
    assert "migrate-import-manifest-1.2-to-1.3" in result.applied_steps
    assert result.payload["scene_id"] == "scene-abc"
    assert result.payload["run_id"] == "job-123"
    assert result.payload["status"] == "unknown"
    assert result.payload["import_status"] == "unknown"
    assert result.payload["robot_types"] == ["franka", "ur5e"]
    assert result.payload["recordings_format"] == "json"
    assert result.payload["artifact_contract_version"] == "1.0"
