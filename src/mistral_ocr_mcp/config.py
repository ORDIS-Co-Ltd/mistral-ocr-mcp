"""Configuration module for Mistral OCR MCP server.

This module loads and validates environment variables required for the server.
"""

import os
from pathlib import Path
from typing import NamedTuple


class Config(NamedTuple):
    """Configuration for the Mistral OCR MCP server.

    Attributes:
        api_key: Mistral API key (never logged)
        allowed_dir_original: Original allowed directory string from environment
        allowed_dir_resolved: Resolved canonical path to allowed directory
    """

    api_key: str
    allowed_dir_original: str
    allowed_dir_resolved: Path


class ConfigurationError(Exception):
    """Exception raised for configuration errors."""

    pass


def load_config() -> Config:
    """Load and validate configuration from environment variables.

    Reads:
        - MISTRAL_API_KEY: Required API key for Mistral OCR service
        - MISTRAL_OCR_ALLOWED_DIR: Required absolute path to allowed directory

    Returns:
        Config object with validated settings

    Raises:
        ConfigurationError: If any required environment variable is missing
                            or if the allowed directory is invalid
    """
    # Load API key
    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        raise ConfigurationError(
            "Missing required environment variable: MISTRAL_API_KEY"
        )

    # Load allowed directory
    allowed_dir_str = os.getenv("MISTRAL_OCR_ALLOWED_DIR")
    if not allowed_dir_str:
        raise ConfigurationError(
            "Missing required environment variable: MISTRAL_OCR_ALLOWED_DIR"
        )

    # Verify it's an absolute path BEFORE canonicalization (SRS FR-5.3)
    if not Path(allowed_dir_str).is_absolute():
        raise ConfigurationError(
            f"MISTRAL_OCR_ALLOWED_DIR must be an absolute path: {allowed_dir_str}"
        )

    # Validate and canonicalize allowed directory
    try:
        allowed_dir = Path(allowed_dir_str).resolve(strict=True)
    except FileNotFoundError:
        raise ConfigurationError(
            f"MISTRAL_OCR_ALLOWED_DIR does not exist: {allowed_dir_str}"
        )
    except RuntimeError as e:
        # Can happen if path contains infinite symlink loops
        raise ConfigurationError(
            f"Invalid MISTRAL_OCR_ALLOWED_DIR: {allowed_dir_str} - {e}"
        )

    # Verify it's a directory
    if not allowed_dir.is_dir():
        raise ConfigurationError(
            f"MISTRAL_OCR_ALLOWED_DIR is not a directory: {allowed_dir_str}"
        )

    return Config(
        api_key=api_key,
        allowed_dir_original=allowed_dir_str,
        allowed_dir_resolved=allowed_dir,
    )
