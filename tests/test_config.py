"""Tests for config module."""

import os
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
        assert "must be an absolute path" in str(exc_info.value)
        assert relative_dir in str(exc_info.value)

    def test_successful_config_load(self, monkeypatch, tmp_path):
        """Test successful configuration loading."""
        monkeypatch.setenv("MISTRAL_API_KEY", "test-api-key-12345")
        monkeypatch.setenv("MISTRAL_OCR_ALLOWED_DIR", str(tmp_path))

        config = load_config()

        assert isinstance(config, Config)
        assert config.api_key == "test-api-key-12345"
        assert config.allowed_dir_original == str(tmp_path)
        assert config.allowed_dir_resolved == tmp_path.resolve()
        # Verify API key is NOT in the error message (if we were to create one)
        assert "test-api-key-12345" == config.api_key

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
        assert config.allowed_dir_resolved == nested.resolve()
        # Original string should be preserved as-is
        assert config.allowed_dir_original == path_with_double_slash
