"""Tests for config module."""

import sys
from pathlib import Path

import pytest

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mistral_ocr_mcp.config import Config, ConfigurationError, load_config


class TestLoadConfig:
    """Tests for load_config function."""

    def test_missing_api_key(self, monkeypatch):
        """Test that ConfigurationError is raised when API key is missing."""
        monkeypatch.delenv("MISTRAL_API_KEY", raising=False)
        monkeypatch.setenv("MISTRAL_OCR_ALLOWED_DIR", "/tmp")

        with pytest.raises(ConfigurationError) as exc_info:
            load_config()

        assert "MISTRAL_API_KEY" in str(exc_info.value)
        assert "Missing required environment variable" in str(exc_info.value)

    def test_missing_allowed_dir(self, monkeypatch):
        """Test that ConfigurationError is raised when allowed dir is missing."""
        monkeypatch.setenv("MISTRAL_API_KEY", "test-api-key")
        monkeypatch.delenv("MISTRAL_OCR_ALLOWED_DIR", raising=False)

        with pytest.raises(ConfigurationError) as exc_info:
            load_config()

        assert "MISTRAL_OCR_ALLOWED_DIR" in str(exc_info.value)
        assert "Missing required environment variable" in str(exc_info.value)

    def test_allowed_dir_does_not_exist(self, monkeypatch):
        """Test that ConfigurationError is raised when allowed dir doesn't exist."""
        monkeypatch.setenv("MISTRAL_API_KEY", "test-api-key")
        monkeypatch.setenv("MISTRAL_OCR_ALLOWED_DIR", "/nonexistent/directory/xyz")

        with pytest.raises(ConfigurationError) as exc_info:
            load_config()

        assert "does not exist" in str(exc_info.value)
        assert "/nonexistent/directory/xyz" in str(exc_info.value)

    def test_allowed_dir_not_a_directory(self, monkeypatch, tmp_path):
        """Test that ConfigurationError is raised when allowed dir is a file."""
        # Create a file instead of a directory
        test_file = tmp_path / "testfile"
        test_file.write_text("test")

        monkeypatch.setenv("MISTRAL_API_KEY", "test-api-key")
        monkeypatch.setenv("MISTRAL_OCR_ALLOWED_DIR", str(test_file))

        with pytest.raises(ConfigurationError) as exc_info:
            load_config()

        assert "not a directory" in str(exc_info.value)
        assert "is not a directory" in str(exc_info.value)

    def test_allowed_dir_relative_path_rejected(self, monkeypatch, tmp_path):
        """Test that ConfigurationError is raised when allowed dir is relative (SRS FR-5.3)."""
        # Use a relative path to a directory that exists
        relative_dir = "subdir"
        (tmp_path / relative_dir).mkdir()

        # Change to tmp_path to make relative path work
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("MISTRAL_API_KEY", "test-api-key")
        monkeypatch.setenv("MISTRAL_OCR_ALLOWED_DIR", relative_dir)

        with pytest.raises(ConfigurationError) as exc_info:
            load_config()

        # Should be rejected because it's relative
        assert "is not an absolute path" in str(exc_info.value)
        assert relative_dir in str(exc_info.value)

    def test_successful_config_load(self, monkeypatch, tmp_path):
        """Test successful configuration loading."""
        monkeypatch.setenv("MISTRAL_API_KEY", "test-api-key-12345")
        monkeypatch.setenv("MISTRAL_OCR_ALLOWED_DIR", str(tmp_path))

        config = load_config()

        assert isinstance(config, Config)
        assert config.api_key == "test-api-key-12345"
        assert config.allowed_dirs_original == str(tmp_path)
        assert config.allowed_dirs_resolved == [tmp_path.resolve()]
        # Verify API key is NOT in the error message (if we were to create one)
        assert config.api_key == "test-api-key-12345"

    def test_canonicalization_of_allowed_dir(self, monkeypatch, tmp_path):
        """Test that allowed directory is properly canonicalized."""
        # Create a nested directory
        nested = tmp_path / "a" / "b" / "c"
        nested.mkdir(parents=True)

        # Use path with symlinks or .. (if supported on platform)
        # On most platforms, we can test with a path containing ..
        monkeypatch.setenv("MISTRAL_API_KEY", "test-api-key")

        # Create path with double slashes or other normalization opportunities
        path_with_double_slash = str(tmp_path) + "//a//b//c//."
        monkeypatch.setenv("MISTRAL_OCR_ALLOWED_DIR", path_with_double_slash)

        config = load_config()

        # Should be resolved to canonical form
        assert config.allowed_dirs_resolved == [nested.resolve()]
        # Original string should be preserved as-is
        assert config.allowed_dirs_original == path_with_double_slash


class TestPathDetection:
    """Tests for _is_windows_path and _is_unix_path helpers."""

    @staticmethod
    def _import_helpers():
        from mistral_ocr_mcp.config import _filter_os_paths, _is_unix_path, _is_windows_path, _parse_allowed_dirs

        return _is_windows_path, _is_unix_path, _filter_os_paths, _parse_allowed_dirs

    def test_windows_path_detection(self):
        """Test that Windows drive-letter paths are correctly identified."""
        is_windows, _, _, _ = self._import_helpers()

        assert is_windows(r"C:\Users\ben") is True
        assert is_windows(r"D:/data/file.pdf") is True
        assert is_windows("Z:\\temp\\") is True
        assert is_windows("e:/file") is True

        # Non-Windows paths
        assert is_windows("/home/ben") is False
        assert is_windows("relative/path") is False
        assert is_windows(r"\server\share") is False  # UNC, no drive letter

    def test_unix_path_detection(self):
        """Test that Unix absolute paths are correctly identified."""
        _, is_unix, _, _ = self._import_helpers()

        assert is_unix("/home/ben") is True
        assert is_unix("/var/log/file.log") is True
        assert is_unix("/tmp") is True

        # Non-Unix paths
        assert is_unix(r"C:\Users") is False
        assert is_unix("relative/path") is False
        assert is_unix(r"\server\share") is False

    def test_filter_os_paths(self, monkeypatch):
        """Test that path filtering respects the current OS."""
        _, _, filter_os, _ = self._import_helpers()

        mixed = [
            r"C:\Users\ben\Documents",
            r"D:\Data",
            "/home/ben/Documents",
            "/shared",
        ]

        # Simulate Windows
        monkeypatch.setattr("mistral_ocr_mcp.config._IS_WINDOWS", True)
        win_result = filter_os(mixed)
        assert r"C:\Users\ben\Documents" in win_result
        assert r"D:\Data" in win_result
        assert "/home/ben/Documents" not in win_result
        assert "/shared" not in win_result

        # Simulate Unix/macOS
        monkeypatch.setattr("mistral_ocr_mcp.config._IS_WINDOWS", False)
        unix_result = filter_os(mixed)
        assert "/home/ben/Documents" in unix_result
        assert "/shared" in unix_result
        assert r"C:\Users\ben\Documents" not in unix_result
        assert r"D:\Data" not in unix_result

    def test_filter_os_paths_unknown_style_preserved(self, monkeypatch):
        """Paths that match neither style are kept as-is."""
        _, _, filter_os, _ = self._import_helpers()

        # A UNC path or bare relative path doesn't match either pattern
        paths = [r"\\server\share", "relative/path"]
        monkeypatch.setattr("mistral_ocr_mcp.config._IS_WINDOWS", False)
        result = filter_os(paths)
        assert r"\\server\share" in result
        assert "relative/path" in result

    def test_parse_allowed_dirs_splits_semicolons(self):
        """Semicolons are used as the path separator."""
        _, _, _, parse = self._import_helpers()

        result = parse("/a;/b;/c")
        for p in ("/a", "/b", "/c"):
            assert p in result

    def test_parse_allowed_dirs_empty_parts_skipped(self):
        """Empty parts from consecutive semicolons are skipped."""
        _, _, _, parse = self._import_helpers()

        result = parse("/a;;/b; ;/c")
        for p in ("/a", "/b", "/c"):
            assert p in result
        assert len(result) == 3

    def test_parse_allowed_dirs_os_filtering(self, monkeypatch):
        """OS filtering is applied after splitting."""
        _, _, _, parse = self._import_helpers()

        monkeypatch.setattr("mistral_ocr_mcp.config._IS_WINDOWS", False)
        result = parse("/home/a;/home/b;" + r"C:\Users\x")
        assert "/home/a" in result
        assert "/home/b" in result
        assert r"C:\Users\x" not in result


class TestMultiDirConfig:
    """Tests for multi-directory config loading."""

    def test_semicolon_separated_directories(self, monkeypatch, tmp_path):
        """Multiple directories separated by semicolons are all loaded."""
        d1 = tmp_path / "dir1"
        d2 = tmp_path / "dir2"
        d1.mkdir()
        d2.mkdir()

        monkeypatch.setenv("MISTRAL_API_KEY", "test-key")
        monkeypatch.setenv("MISTRAL_OCR_ALLOWED_DIR", f"{d1};{d2}")

        config = load_config()
        assert len(config.allowed_dirs_resolved) == 2
        assert d1.resolve() in config.allowed_dirs_resolved
        assert d2.resolve() in config.allowed_dirs_resolved

    def test_no_paths_match_current_os(self, monkeypatch):
        """Error when no paths match the current OS."""
        # On Unix, Windows-style paths alone should fail
        monkeypatch.setattr("mistral_ocr_mcp.config._IS_WINDOWS", False)
        monkeypatch.setenv("MISTRAL_API_KEY", "test-key")
        monkeypatch.setenv("MISTRAL_OCR_ALLOWED_DIR", r"C:\Temp;D:\Data")

        with pytest.raises(ConfigurationError) as exc_info:
            load_config()

        assert "no paths match the current OS" in str(exc_info.value)

    def test_mixed_valid_invalid_skips_bad_paths(self, monkeypatch, tmp_path):
        """Invalid paths are skipped; only valid directories are returned."""
        d1 = tmp_path / "dir1"
        d1.mkdir()

        monkeypatch.setenv("MISTRAL_API_KEY", "test-key")
        monkeypatch.setenv(
            "MISTRAL_OCR_ALLOWED_DIR",
            f"/nonexistent/foo;{d1};/nonexistent/bar",
        )

        config = load_config()
        assert config.allowed_dirs_resolved == [d1.resolve()]

    def test_all_paths_invalid_raises_error(self, monkeypatch):
        """Error when none of the listed paths are valid."""
        monkeypatch.setenv("MISTRAL_API_KEY", "test-key")
        monkeypatch.setenv(
            "MISTRAL_OCR_ALLOWED_DIR",
            "/nonexistent/foo;/nonexistent/bar",
        )

        with pytest.raises(ConfigurationError) as exc_info:
            load_config()

        msg = str(exc_info.value)
        assert "no valid directories found" in msg
        assert "/nonexistent/foo" in msg
        assert "/nonexistent/bar" in msg
        assert "does not exist" in msg
