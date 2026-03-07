from __future__ import annotations

from pathlib import Path

import pytest

import tools.source_pipeline.adapter as adapter_mod


def test_validated_download_url_rejects_unsafe_schemes_and_hosts() -> None:
    with pytest.raises(ValueError, match="unsupported_download_scheme"):
        adapter_mod._validated_download_url("file:///etc/passwd")

    with pytest.raises(ValueError, match="disallowed_download_host"):
        adapter_mod._validated_download_url("http://127.0.0.1/asset.glb")

    with pytest.raises(ValueError, match="disallowed_download_host"):
        adapter_mod._validated_download_url("http://localhost/asset.glb")


def test_validated_download_url_rejects_private_dns_resolution(monkeypatch) -> None:
    def _fake_getaddrinfo(*_args, **_kwargs):
        return [(None, None, None, None, ("10.0.0.5", 443))]

    monkeypatch.setattr(adapter_mod.socket, "getaddrinfo", _fake_getaddrinfo)

    with pytest.raises(ValueError, match="disallowed_download_host"):
        adapter_mod._validated_download_url("https://evil.example/model.glb")


def test_download_file_validates_url_before_retrieval(tmp_path: Path, monkeypatch) -> None:
    captured: dict[str, str] = {}

    def _fake_urlretrieve(url: str, out_path: Path):
        captured["url"] = url
        Path(out_path).write_bytes(b"ok")

    monkeypatch.setattr(adapter_mod.urllib.request, "urlretrieve", _fake_urlretrieve)

    with pytest.raises(ValueError, match="unsupported_download_scheme"):
        adapter_mod._download_file("file:///tmp/secret", tmp_path / "out.glb")

    assert "url" not in captured
