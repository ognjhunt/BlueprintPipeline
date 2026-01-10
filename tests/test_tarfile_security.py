#!/usr/bin/env python3
"""
Unit tests for tarfile path traversal security fix.

Tests the safe_extract_tar function to ensure it properly prevents
path traversal attacks (CWE-22).
"""

import os
import sys
import tarfile
import tempfile
from pathlib import Path

import pytest

# Add parent to path
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Import the function to test
sys.path.insert(0, str(REPO_ROOT / "genie-sim-export-job"))
from geniesim_client import safe_extract_tar


class TestSafeExtractTar:
    """Tests for safe_extract_tar function."""

    def test_normal_extraction(self, tmp_path):
        """Test that normal archives extract successfully."""
        # Create a normal archive
        archive_dir = tmp_path / "archive_content"
        archive_dir.mkdir()

        test_file = archive_dir / "test.txt"
        test_file.write_text("Hello, World!")

        nested_dir = archive_dir / "nested"
        nested_dir.mkdir()
        nested_file = nested_dir / "nested.txt"
        nested_file.write_text("Nested content")

        archive_path = tmp_path / "normal.tar.gz"
        with tarfile.open(archive_path, "w:gz") as tar:
            tar.add(test_file, arcname="test.txt")
            tar.add(nested_file, arcname="nested/nested.txt")

        # Extract
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        safe_extract_tar(archive_path, output_dir)

        # Verify extraction
        assert (output_dir / "test.txt").exists()
        assert (output_dir / "test.txt").read_text() == "Hello, World!"
        assert (output_dir / "nested" / "nested.txt").exists()
        assert (output_dir / "nested" / "nested.txt").read_text() == "Nested content"

    def test_blocks_absolute_path(self, tmp_path):
        """Test that absolute paths are blocked."""
        # Create malicious archive with absolute path
        archive_path = tmp_path / "absolute.tar.gz"
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        with tarfile.open(archive_path, "w:gz") as tar:
            # Create a file with absolute path
            content = b"Malicious content"
            tarinfo = tarfile.TarInfo(name="/etc/passwd")
            tarinfo.size = len(content)

            import io
            tar.addfile(tarinfo, io.BytesIO(content))

        # Should raise ValueError
        with pytest.raises(ValueError, match="Suspicious path"):
            safe_extract_tar(archive_path, output_dir)

    def test_blocks_parent_directory_traversal(self, tmp_path):
        """Test that parent directory references are blocked."""
        # Create malicious archive with ../
        archive_path = tmp_path / "traversal.tar.gz"
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        with tarfile.open(archive_path, "w:gz") as tar:
            # Create file that tries to escape
            content = b"Malicious content"
            tarinfo = tarfile.TarInfo(name="../../../etc/passwd")
            tarinfo.size = len(content)

            import io
            tar.addfile(tarinfo, io.BytesIO(content))

        # Should raise ValueError
        with pytest.raises(ValueError, match="Suspicious path|Path traversal"):
            safe_extract_tar(archive_path, output_dir)

    def test_blocks_complex_traversal(self, tmp_path):
        """Test that complex traversal attempts are blocked."""
        # Create archive with subtle traversal
        archive_path = tmp_path / "complex.tar.gz"
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        with tarfile.open(archive_path, "w:gz") as tar:
            # Try to escape using legitimate-looking path
            content = b"Malicious content"
            tarinfo = tarfile.TarInfo(name="legitimate/../../../etc/shadow")
            tarinfo.size = len(content)

            import io
            tar.addfile(tarinfo, io.BytesIO(content))

        # Should raise ValueError (path normalization will catch this)
        with pytest.raises(ValueError, match="Path traversal"):
            safe_extract_tar(archive_path, output_dir)

    def test_allows_safe_relative_paths(self, tmp_path):
        """Test that safe relative paths within output dir are allowed."""
        # Create archive with various safe relative paths
        archive_path = tmp_path / "safe_relative.tar.gz"

        files_to_create = [
            "file.txt",
            "dir/file.txt",
            "dir/subdir/file.txt",
            "./file2.txt",
            "dir/./file3.txt",
        ]

        with tarfile.open(archive_path, "w:gz") as tar:
            for filename in files_to_create:
                content = f"Content of {filename}".encode()
                tarinfo = tarfile.TarInfo(name=filename)
                tarinfo.size = len(content)

                import io
                tar.addfile(tarinfo, io.BytesIO(content))

        # Extract
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        safe_extract_tar(archive_path, output_dir)

        # Verify all files extracted
        assert (output_dir / "file.txt").exists()
        assert (output_dir / "dir" / "file.txt").exists()
        assert (output_dir / "dir" / "subdir" / "file.txt").exists()
        assert (output_dir / "file2.txt").exists()
        assert (output_dir / "dir" / "file3.txt").exists()

    def test_empty_archive(self, tmp_path):
        """Test handling of empty archives."""
        # Create empty archive
        archive_path = tmp_path / "empty.tar.gz"
        with tarfile.open(archive_path, "w:gz") as tar:
            pass

        output_dir = tmp_path / "output"
        output_dir.mkdir()

        # Should not raise error
        safe_extract_tar(archive_path, output_dir)

        # Output dir should still be empty (no files added)
        assert list(output_dir.iterdir()) == []

    def test_symlink_attack(self, tmp_path):
        """Test that symlink attacks are handled safely."""
        # Create archive with symlink pointing outside
        archive_path = tmp_path / "symlink.tar.gz"
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        with tarfile.open(archive_path, "w:gz") as tar:
            # Add a symlink that points outside output_dir
            tarinfo = tarfile.TarInfo(name="evil_link")
            tarinfo.type = tarfile.SYMTYPE
            tarinfo.linkname = "../../../etc/passwd"
            tar.addfile(tarinfo)

        # This should either:
        # 1. Extract safely (symlink creation fails/is contained)
        # 2. Raise an error
        # Either way, we verify the symlink doesn't escape
        try:
            safe_extract_tar(archive_path, output_dir)

            # If extraction succeeded, verify symlink doesn't escape
            if (output_dir / "evil_link").exists():
                link_target = os.readlink(output_dir / "evil_link")
                resolved = (output_dir / link_target).resolve()

                # Verify resolved path is still inside output_dir
                try:
                    resolved.relative_to(output_dir.resolve())
                except ValueError:
                    pytest.fail("Symlink escaped output directory!")
        except (ValueError, tarfile.TarError):
            # Expected: extraction blocked the symlink
            pass

    def test_large_member_names(self, tmp_path):
        """Test handling of very long member names."""
        # Create archive with very long filename
        archive_path = tmp_path / "long_name.tar.gz"

        long_name = "a" * 200 + "/" + "b" * 100 + ".txt"

        with tarfile.open(archive_path, "w:gz") as tar:
            content = b"Content"
            tarinfo = tarfile.TarInfo(name=long_name)
            tarinfo.size = len(content)

            import io
            tar.addfile(tarinfo, io.BytesIO(content))

        output_dir = tmp_path / "output"
        output_dir.mkdir()

        # Should extract successfully
        safe_extract_tar(archive_path, output_dir)

        # Verify file exists (path may be truncated by filesystem)
        extracted_files = list(output_dir.rglob("*.txt"))
        assert len(extracted_files) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
