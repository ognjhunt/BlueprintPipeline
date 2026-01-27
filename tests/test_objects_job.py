from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest


def test_load_da3_geom_valid_and_missing(load_job_module, tmp_path: Path):
    module = load_job_module("objects", "run_objects_from_layout.py")

    valid_path = tmp_path / "da3_geom.npz"
    np.savez(
        valid_path,
        depth=np.zeros((1, 2, 2), dtype=np.float32),
        conf=np.ones((1, 2, 2), dtype=np.float32),
        extrinsics=np.zeros((1, 3, 4), dtype=np.float32),
        intrinsics=np.eye(3, dtype=np.float32)[None, ...],
        image_paths=np.array(["image.png"]),
    )
    depth, conf, extrinsics, intrinsics, image_paths = module.load_da3_geom(valid_path)
    assert depth.shape == (1, 2, 2)
    assert conf.shape == (1, 2, 2)
    assert extrinsics.shape == (1, 3, 4)
    assert intrinsics.shape == (1, 3, 3)
    assert image_paths.shape == (1,)

    invalid_path = tmp_path / "da3_geom_missing.npz"
    np.savez(
        invalid_path,
        depth=np.zeros((1, 2, 2), dtype=np.float32),
        conf=np.ones((1, 2, 2), dtype=np.float32),
    )
    with pytest.raises(ValueError):
        module.load_da3_geom(invalid_path)


def test_parse_yolo_labels_and_class_names(load_job_module, tmp_path: Path):
    module = load_job_module("objects", "run_objects_from_layout.py")

    data_yaml = tmp_path / "data.yaml"
    data_yaml.write_text("names:\n  - mug\n  - table\n")
    class_names = module.load_class_names(data_yaml)
    assert class_names == ["mug", "table"]

    label_path = tmp_path / "labels.txt"
    label_path.write_text("1 0 0 1 0 1 1 0 1\n")
    objects = module.parse_yolo_labels(label_path, class_names=class_names)

    assert len(objects) == 1
    assert objects[0]["class_name"] == "table"
    assert objects[0]["bbox2d"][2] > 0


def test_backproject_region_to_world_minimal(load_job_module):
    module = load_job_module("objects", "run_objects_from_layout.py")

    depth = np.ones((2, 2), dtype=np.float32)
    conf = np.ones((2, 2), dtype=np.float32)
    K = np.eye(3, dtype=np.float32)
    w2c = np.hstack([np.eye(3, dtype=np.float32), np.zeros((3, 1), dtype=np.float32)])

    points = module.backproject_region_to_world(
        depth,
        conf,
        K,
        w2c,
        0,
        0,
        1,
        1,
        conf_thresh=0.5,
    )

    assert points.shape[1] == 3
    assert points.shape[0] == 1
    assert np.all(points[:, 2] == 1.0)
