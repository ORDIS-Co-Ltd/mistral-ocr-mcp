"""Configuration module for Mistral OCR MCP server.

This module loads and validates environment variables required for the server.
"""

import os
import re
import sys
from pathlib import Path
from typing import NamedTuple


class Config(NamedTuple):
    """Configuration for the Mistral OCR MCP server.

    Attributes:
        api_key: Mistral API key (never logged)
        allowed_dirs_original: Semicolon-separated original string from environment
        allowed_dirs_resolved: List of resolved canonical paths valid for this OS
    """

    api_key: str
    allowed_dirs_original: str
    allowed_dirs_resolved: list[Path]


class ConfigurationError(Exception):
    """Exception raised for configuration errors."""

    pass


_IS_WINDOWS = sys.platform == "win32"


def _is_windows_path(p: str) -> bool:
    """Return True if *p* looks like a Windows drive-letter path (e.g. ``C:\\...``)."""
    return bool(re.match(r"^[A-Za-z]:[/\\]", p))


def _is_unix_path(p: str) -> bool:
    """Return True if *p* looks like a Unix absolute path (starts with ``/``)."""
    return p.startswith("/")


def _filter_os_paths(paths: list[str]) -> list[str]:
    """Keep only paths that match the current operating system's style.

    - On Windows: keep Windows-style paths (drive letter, e.g. ``C:\\...``).
    - On Unix/macOS: keep Unix-style paths (start with ``/``).
    - Paths that match neither style are kept as-is (they may be valid absolute
      paths on the current system despite an unusual format).
    """
    if _IS_WINDOWS:
        return [p for p in paths if not _is_unix_path(p)]
    return [p for p in paths if not _is_windows_path(p)]


def _parse_allowed_dirs(raw: str) -> list[str]:
    """Split *raw* on semicolons and return OS-appropriate paths."""
    parts = [p.strip() for p in raw.split(";") if p.strip()]
    return _filter_os_paths(parts)


def load_config() -> Config:
    """Load and validate configuration from environment variables.

    Reads:
        - MISTRAL_API_KEY: Required API key for Mistral OCR service
        - MISTRAL_OCR_ALLOWED_DIR: Semicolon-separated list of absolute paths
          to allowed directories.  Paths are filtered by OS: on Windows only
          Windows-style paths (drive letter, e.g. ``C:\\...``) are used; on
          Unix/macOS only Unix-style paths (starting with ``/``) are used.

    Returns:
        Config object with validated settings

    Raises:
        ConfigurationError: If any required environment variable is missing
                            or if no valid allowed directory matches the
                            current OS.
    """
    # Load API key
    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        raise ConfigurationError(
            "Missing required environment variable: MISTRAL_API_KEY"
        )

    # Load allowed directories
    allowed_dirs_raw = os.getenv("MISTRAL_OCR_ALLOWED_DIR")
    if not allowed_dirs_raw:
        raise ConfigurationError(
            "Missing required environment variable: MISTRAL_OCR_ALLOWED_DIR"
        )

    candidate_paths = _parse_allowed_dirs(allowed_dirs_raw)
    if not candidate_paths:
        raise ConfigurationError(
            f"MISTRAL_OCR_ALLOWED_DIR: no paths match the current OS "
            f"({'Windows' if _IS_WINDOWS else 'Unix/macOS'}): "
            f"{allowed_dirs_raw}"
        )

    resolved: list[Path] = []
    errors: list[str] = []
    for raw_path in candidate_paths:
        try:
            p = Path(raw_path)
            if not p.is_absolute():
                errors.append(f"{raw_path} is not an absolute path")
                continue

            resolved_path = p.resolve(strict=True)
            if not resolved_path.is_dir():
                errors.append(f"{raw_path} is not a directory")
                continue

            resolved.append(resolved_path)
        except FileNotFoundError:
            errors.append(f"{raw_path} does not exist")
        except RuntimeError as e:
            errors.append(f"{raw_path} - {e}")
        except Exception as e:
            errors.append(f"{raw_path} - {e}")

    if not resolved:
        raise ConfigurationError(
            f"MISTRAL_OCR_ALLOWED_DIR: no valid directories found for the "
            f"current OS. Errors:\n  " + "\n  ".join(errors)
        )

    return Config(
        api_key=api_key,
        allowed_dirs_original=allowed_dirs_raw,
        allowed_dirs_resolved=resolved,
    )
