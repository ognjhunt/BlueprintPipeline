from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence, Union

import google_crc32c

from tools.rlds.proto import example_pb2, feature_pb2


def _masked_crc32c(data: bytes) -> int:
    """TFRecord masked CRC32C (Castagnoli) checksum."""
    crc = int(google_crc32c.value(data)) & 0xFFFFFFFF
    masked = ((crc >> 15) | (crc << 17)) & 0xFFFFFFFF
    masked = (masked + 0xA282EAD8) & 0xFFFFFFFF
    return masked


class TFRecordWriter:
    """Minimal TFRecord writer (no compression)."""

    def __init__(self, path: Union[str, Path]) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = open(self.path, "wb")

    def write(self, record: bytes) -> None:
        length_bytes = struct.pack("<Q", len(record))
        self._handle.write(length_bytes)
        self._handle.write(struct.pack("<I", _masked_crc32c(length_bytes)))
        self._handle.write(record)
        self._handle.write(struct.pack("<I", _masked_crc32c(record)))

    def close(self) -> None:
        if self._handle:
            self._handle.close()
            self._handle = None

    def __enter__(self) -> "TFRecordWriter":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        self.close()
        return False


def bytes_feature(value: bytes) -> feature_pb2.Feature:
    return feature_pb2.Feature(bytes_list=feature_pb2.BytesList(value=[value]))


def int64_feature(value: int) -> feature_pb2.Feature:
    return feature_pb2.Feature(int64_list=feature_pb2.Int64List(value=[int(value)]))


def float_feature(values: Union[float, Sequence[float]]) -> feature_pb2.Feature:
    if isinstance(values, (list, tuple)):
        payload = [float(v) for v in values]
    else:
        payload = [float(values)]
    return feature_pb2.Feature(float_list=feature_pb2.FloatList(value=payload))


def make_tf_example(features: Mapping[str, feature_pb2.Feature]) -> bytes:
    example = example_pb2.Example(
        features=feature_pb2.Features(
            feature=dict(features),
        )
    )
    return example.SerializeToString()


@dataclass(frozen=True)
class TFRecordWriteResult:
    path: Path
    records: int
    bytes_written: int


def write_tfrecord(
    path: Union[str, Path],
    records: Iterable[bytes],
) -> TFRecordWriteResult:
    output_path = Path(path)
    count = 0
    bytes_written = 0
    with TFRecordWriter(output_path) as writer:
        for record in records:
            writer.write(record)
            count += 1
            bytes_written += len(record)
    return TFRecordWriteResult(path=output_path, records=count, bytes_written=bytes_written)

