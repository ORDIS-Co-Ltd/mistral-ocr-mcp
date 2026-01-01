"""Tests for path_sandbox module."""

import os
import sys
from pathlib import Path
import pytest

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mistral_ocr_mcp.path_sandbox import (
    SUPPORTED_EXTENSIONS,
    PathValidationError,
    validate_file_path,
    validate_output_dir,
)


class TestSupportedExtensions:
    """Tests for supported extensions constant."""

    def test_all_expected_extensions_present(self):
        """Test that all expected extensions are in SUPPORTED_EXTENSIONS."""
        expected = {".pdf", ".png", ".jpg", ".jpeg", ".webp", ".gif"}
        assert SUPPORTED_EXTENSIONS == expected


class TestValidateFilePath:
    """Tests for validate_file_path function."""

    def test_non_absolute_path(self, tmp_path):
        """Test that relative paths are rejected."""
        with pytest.raises(PathValidationError) as exc_info:
            validate_file_path("relative/path.txt")

        assert "validate file_path" in str(exc_info.value)
        assert "absolute path" in str(exc_info.value)
        assert "relative/path.txt" in str(exc_info.value)

    def test_nonexistent_file(self):
        """Test that nonexistent files are rejected."""
        with pytest.raises(PathValidationError) as exc_info:
            validate_file_path("/nonexistent/file.pdf")

        assert "validate file_path" in str(exc_info.value)
        assert "resolve failed" in str(exc_info.value)
        assert "does not exist" in str(exc_info.value)
        assert "/nonexistent/file.pdf" in str(exc_info.value)

    def test_unsupported_extension(self, tmp_path):
        """Test that files with unsupported extensions are rejected."""
        # Create a file with .txt extension
        test_file = tmp_path / "test.txt"
        test_file.write_text("test")

        with pytest.raises(PathValidationError) as exc_info:
            validate_file_path(str(test_file))

        assert "validate file_path" in str(exc_info.value)
        assert "unsupported file type" in str(exc_info.value)
        assert ".txt" in str(exc_info.value)
        assert ".pdf" in str(exc_info.value)
        assert ".png" in str(exc_info.value)
        assert str(test_file) in str(exc_info.value)

    def test_supported_pdf_file(self, tmp_path):
        """Test that .pdf files are accepted."""
        test_file = tmp_path / "test.pdf"
        test_file.write_text("test")

        result = validate_file_path(str(test_file))
        assert result == test_file.resolve()

    def test_supported_png_file(self, tmp_path):
        """Test that .png files are accepted."""
        test_file = tmp_path / "test.png"
        test_file.write_text("test")

        result = validate_file_path(str(test_file))
        assert result == test_file.resolve()

    def test_supported_jpg_file(self, tmp_path):
        """Test that .jpg files are accepted."""
        test_file = tmp_path / "test.jpg"
        test_file.write_text("test")

        result = validate_file_path(str(test_file))
        assert result == test_file.resolve()

    def test_supported_jpeg_file(self, tmp_path):
        """Test that .jpeg files are accepted."""
        test_file = tmp_path / "test.jpeg"
        test_file.write_text("test")

        result = validate_file_path(str(test_file))
        assert result == test_file.resolve()

    def test_supported_webp_file(self, tmp_path):
        """Test that .webp files are accepted."""
        test_file = tmp_path / "test.webp"
        test_file.write_text("test")

        result = validate_file_path(str(test_file))
        assert result == test_file.resolve()

    def test_supported_gif_file(self, tmp_path):
        """Test that .gif files are accepted."""
        test_file = tmp_path / "test.gif"
        test_file.write_text("test")

        result = validate_file_path(str(test_file))
        assert result == test_file.resolve()

    def test_case_insensitive_extension(self, tmp_path):
        """Test that extension validation is case-insensitive."""
        test_file = tmp_path / "test.PDF"
        test_file.write_text("test")

        result = validate_file_path(str(test_file))
        assert result == test_file.resolve()

    def test_canonicalization(self, tmp_path):
        """Test that paths are canonicalized."""
        test_file = tmp_path / "test.pdf"
        test_file.write_text("test")

        # Use path with double slashes
        path_with_dots = str(tmp_path) + "//test.pdf"
        result = validate_file_path(path_with_dots)

        assert result == test_file.resolve()


class TestValidateOutputDir:
    """Tests for validate_output_dir function."""

    def test_non_absolute_path(self):
        """Test that relative paths are rejected."""
        with pytest.raises(PathValidationError) as exc_info:
            validate_output_dir("relative/path", Path("/tmp"), "/tmp")

        assert "validate output_dir" in str(exc_info.value)
        assert "absolute path" in str(exc_info.value)
        assert "relative/path" in str(exc_info.value)

    def test_nonexistent_directory(self):
        """Test that nonexistent directories are rejected."""
        with pytest.raises(PathValidationError) as exc_info:
            validate_output_dir("/nonexistent/directory", Path("/tmp"), "/tmp")

        assert "validate output_dir" in str(exc_info.value)
        assert "resolve failed" in str(exc_info.value)
        assert "does not exist" in str(exc_info.value)
        assert "/nonexistent/directory" in str(exc_info.value)

    def test_file_instead_of_directory(self, tmp_path):
        """Test that files are rejected when a directory is expected."""
        test_file = tmp_path / "testfile"
        test_file.write_text("test")

        with pytest.raises(PathValidationError) as exc_info:
            validate_output_dir(str(test_file), Path("/tmp"), "/tmp")

        assert "validate output_dir" in str(exc_info.value)
        assert "is not a directory" in str(exc_info.value)
        assert str(test_file) in str(exc_info.value)

    def test_not_writable(self, tmp_path):
        """Test that non-writable directories are rejected."""
        # On Unix, we can make a directory read-only
        test_dir = tmp_path / "readonly"
        test_dir.mkdir()

        # Make directory read-only (skip on Windows)
        if os.name != "nt":
            original_mode = test_dir.stat().st_mode
            try:
                test_dir.chmod(0o444)  # Read-only

                with pytest.raises(PathValidationError) as exc_info:
                    validate_output_dir(str(test_dir), Path("/tmp"), "/tmp")

                assert "validate output_dir" in str(exc_info.value)
                assert "writability check failed" in str(exc_info.value)
                assert "not writable" in str(exc_info.value)
                assert str(test_dir) in str(exc_info.value)
            finally:
                # Restore permissions
                test_dir.chmod(original_mode)

    def test_outside_allowed_dir(self, tmp_path):
        """Test that directories outside allowed dir are rejected."""
        allowed_dir = tmp_path / "allowed"
        allowed_dir.mkdir()

        output_dir = tmp_path / "outside"
        output_dir.mkdir()

        with pytest.raises(PathValidationError) as exc_info:
            validate_output_dir(str(output_dir), allowed_dir, "/allowed/original")

        assert (
            str(exc_info.value)
            == "output_dir must be within the allowed directory: /allowed/original"
        )

    def test_parent_traversal_attack(self, tmp_path):
        """Test that .. traversal attacks are blocked."""
        allowed_dir = tmp_path / "allowed"
        allowed_dir.mkdir()

        # Try to use a path that resolves to parent directory
        output_dir = tmp_path  # Parent of allowed_dir
        output_dir_str = str(allowed_dir / "..")

        with pytest.raises(PathValidationError) as exc_info:
            validate_output_dir(output_dir_str, allowed_dir.resolve(), str(allowed_dir))

        # The error should mention the original allowed dir string
        assert (
            str(exc_info.value)
            == f"output_dir must be within the allowed directory: {allowed_dir}"
        )

    def test_success_exact_match(self, tmp_path):
        """Test that exact match to allowed dir is accepted."""
        result = validate_output_dir(str(tmp_path), tmp_path, str(tmp_path))

        assert result == tmp_path.resolve()

    def test_success_subdirectory(self, tmp_path):
        """Test that subdirectory of allowed dir is accepted."""
        subdir = tmp_path / "subdir"
        subdir.mkdir()

        result = validate_output_dir(str(subdir), tmp_path, "/tmp/original")

        assert result == subdir.resolve()

    def test_success_nested_subdirectory(self, tmp_path):
        """Test that deeply nested subdirectory is accepted."""
        nested = tmp_path / "a" / "b" / "c"
        nested.mkdir(parents=True)

        result = validate_output_dir(str(nested), tmp_path, "/tmp/original")

        assert result == nested.resolve()

    def test_symlink_escape(self, tmp_path):
        """Test that symlinks pointing outside allowed dir are rejected."""
        # Skip on Windows where symlink support may be limited
        if os.name == "nt":
            pytest.skip("symlinks not supported on Windows")

        # Create allowed_dir (real directory)
        allowed_dir = tmp_path / "allowed"
        allowed_dir.mkdir()

        # Create outside_dir (sibling)
        outside_dir = tmp_path / "outside"
        outside_dir.mkdir()

        # Create symlink from allowed_dir/link to outside_dir
        link_path = allowed_dir / "link"
        try:
            link_path.symlink_to(outside_dir)
        except (OSError, NotImplementedError) as e:
            pytest.skip(f"symlink creation not supported: {e}")

        # Passing output_dir=allowed_dir/link should be rejected
        # because it resolves outside allowed_dir
        with pytest.raises(PathValidationError) as exc_info:
            validate_output_dir(
                str(link_path), allowed_dir.resolve(), "/allowed/original"
            )

        assert (
            str(exc_info.value)
            == "output_dir must be within the allowed directory: /allowed/original"
        )
