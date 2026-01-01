"""Path sandbox validation for Mistral OCR MCP server.

This module provides validation for file paths and output directories,
ensuring they are within the allowed directory sandbox.
"""

import os
from pathlib import Path
from typing import Set

# Supported file extensions for OCR processing
SUPPORTED_EXTENSIONS: Set[str] = {
    ".pdf",
    ".png",
    ".jpg",
    ".jpeg",
    ".webp",
    ".gif",
}


class PathValidationError(Exception):
    """Exception raised for path validation errors."""

    pass


def validate_file_path(file_path: str) -> Path:
    """Validate and canonicalize an input file path.

    Args:
        file_path: Absolute path to the input file

    Returns:
        Resolved canonical Path to the file

    Raises:
        PathValidationError: If path is not absolute, doesn't exist,
                            has unsupported extension, or other filesystem error
    """
    path = Path(file_path)

    # Check if absolute
    if not path.is_absolute():
        raise PathValidationError(f"file_path must be an absolute path: {file_path}")

    # Canonicalize and check existence
    try:
        resolved_path = path.resolve(strict=True)
    except FileNotFoundError:
        raise PathValidationError(f"file_path does not exist: {file_path}")
    except RuntimeError as e:
        # Can happen with infinite symlink loops
        raise PathValidationError(f"Invalid file_path: {file_path} - {e}")

    # Check extension
    if resolved_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise PathValidationError(
            "Unsupported file type. Supported types: .pdf, .png, .jpg, .jpeg, .webp, .gif"
        )

    return resolved_path


def validate_output_dir(
    output_dir: str,
    allowed_dir_resolved: Path,
    allowed_dir_original: str,
) -> Path:
    """Validate and canonicalize an output directory path.

    Args:
        output_dir: Absolute path to the output directory
        allowed_dir_resolved: Canonical path to the allowed directory
        allowed_dir_original: Original string from environment (for error messages)

    Returns:
        Resolved canonical Path to the output directory

    Raises:
        PathValidationError: If path is not absolute, doesn't exist,
                            is not a directory, not writable, or outside allowed dir
    """
    path = Path(output_dir)

    # Check if absolute
    if not path.is_absolute():
        raise PathValidationError(f"output_dir must be an absolute path: {output_dir}")

    # Canonicalize and check existence
    try:
        resolved_path = path.resolve(strict=True)
    except FileNotFoundError:
        raise PathValidationError(f"output_dir does not exist: {output_dir}")
    except RuntimeError as e:
        raise PathValidationError(f"Invalid output_dir: {output_dir} - {e}")

    # Verify it's a directory
    if not resolved_path.is_dir():
        raise PathValidationError(f"output_dir is not a directory: {output_dir}")

    # Check writability - try creating a temporary file
    try:
        # Use a unique filename to avoid collisions
        temp_file = resolved_path / f".write_test_{os.getpid()}"
        temp_file.touch()
        temp_file.unlink()
    except PermissionError:
        raise PathValidationError(f"output_dir is not writable: {output_dir}")
    except OSError as e:
        raise PathValidationError(f"Failed to check writability of {output_dir}: {e}")

    # Sandbox enforcement: output_dir must be within allowed directory
    try:
        resolved_path.relative_to(allowed_dir_resolved)
    except ValueError:
        # output_dir is not a descendant of allowed_dir
        raise PathValidationError(
            f"output_dir must be within the allowed directory: {allowed_dir_original}"
        )

    return resolved_path
