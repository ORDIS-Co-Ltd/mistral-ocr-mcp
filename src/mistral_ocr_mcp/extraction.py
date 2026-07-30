"""Extraction orchestration for Mistral OCR MCP server.

This module provides the main extraction functions that orchestrate OCR calls,
image saving, and markdown rewriting.
"""

import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, TypedDict


class ExtractMarkdownWithImagesResult(TypedDict):
    """Result of extracting markdown with embedded images."""

    output_directory: str
    """Absolute path to the output subdirectory."""
    markdown_file: str
    """Absolute path to the content.md file."""
    images: list[str]
    """List of saved image filenames (not full paths)."""

from .config import load_config
from .images import save_images
from .markdown_rewrite import rewrite_markdown
from .mistral_client import (
    check_api_status,
    process_local_file,
    process_local_file_advanced,
    process_url,
)
from .path_sandbox import validate_file_path, validate_output_dir


def extract_markdown(file_path: str) -> str:
    """Extract markdown text from a file without images.

    Args:
        file_path: Absolute path to the input file (PDF or image)

    Returns:
        Concatenated markdown content from all pages

    Raises:
        PathValidationError: If file_path is invalid
        MistralOCRAPIError: If the OCR API call fails
        MistralOCRFileError: If filesystem operations fail
    """
    # Validate file path
    validated_path = validate_file_path(file_path)

    # Call OCR without images
    response = process_local_file(validated_path, include_image_base64=False)

    # Join page markdowns with double newline
    page_markdowns = [page.markdown for page in response.pages]
    return "\n\n".join(page_markdowns)


def extract_markdown_with_images(file_path: str, output_dir: str) -> ExtractMarkdownWithImagesResult:
    """Extract markdown with embedded images and save them as separate files.

    This function:
    1. Validates both file_path and output_dir
    2. Enforces sandbox constraints using config
    3. Creates a unique output subdirectory
    4. Calls OCR with include_image_base64=True
    5. Saves images to the output subdirectory
    6. Rewrites markdown to replace base64 URIs with relative paths
    7. Saves the rewritten markdown as content.md
    8. Returns metadata about the extracted content

    Args:
        file_path: Absolute path to the input file (PDF or image)
        output_dir: Absolute path to the output directory (must be within allowed dir)

    Returns:
        Dictionary with keys:
            - output_directory: Absolute path to the output subdirectory
            - markdown_file: Absolute path to the content.md file
            - images: List of saved image filenames (not full paths)

    Raises:
        PathValidationError: If file_path or output_dir is invalid
        MistralOCRAPIError: If the OCR API call fails
        MistralOCRFileError: If filesystem operations fail
    """
    # Load config to get allowed directory
    config = load_config()

    # Validate file path
    validated_file_path = validate_file_path(file_path)

    # Validate output directory with sandbox enforcement
    validated_output_dir = validate_output_dir(
        output_dir,
        config.allowed_dir_resolved,
        config.allowed_dir_original,
    )

    # Create output subdirectory with collision handling
    output_subdir = _create_output_subdirectory(
        validated_output_dir, validated_file_path
    )

    # Call OCR with images
    response = process_local_file(validated_file_path, include_image_base64=True)

    # Extract images from response
    images: List[dict] = []
    for page in response.pages:
        if hasattr(page, "images") and page.images:
            images.extend(
                [
                    img.model_dump() if hasattr(img, "model_dump") else img
                    for img in page.images
                ]
            )

    # Save images
    saved_filenames = save_images(output_subdir, images)

    # Join page markdowns
    page_markdowns = [page.markdown for page in response.pages]
    markdown_content = "\n\n".join(page_markdowns)

    # Rewrite markdown to replace base64 URIs with relative paths
    rewritten_markdown = rewrite_markdown(markdown_content, images, saved_filenames)

    # Save markdown as content.md
    markdown_file_path = output_subdir / "content.md"
    markdown_file_path.write_text(rewritten_markdown, encoding="utf-8")

    return {
        "output_directory": str(output_subdir),
        "markdown_file": str(markdown_file_path),
        "images": saved_filenames,
    }


class OCRStatusResult(TypedDict):
    """Result of an API status check."""

    status: str
    """"ok" or "error"."""
    message: str
    """Human-readable status description."""


def extract_from_url(
    file_url: str,
    include_image_base64: bool = False,
) -> str:
    """Extract markdown from a publicly accessible URL.

    Processes a PDF or image directly from a URL without uploading
    a local file first.

    Args:
        file_url: Publicly accessible URL to a PDF or image.
        include_image_base64: Whether to include base64 images.

    Returns:
        Concatenated markdown content from all pages.

    Raises:
        MistralOCRAPIError: If the OCR API call fails.
    """
    response = process_url(
        file_url,
        include_image_base64=include_image_base64,
    )
    page_markdowns = [page.markdown for page in response.pages]
    return "\n\n".join(page_markdowns)


def extract_markdown_advanced(
    file_path: str,
    pages: Optional[list[int]] = None,
    table_format_: Optional[str] = None,
    model: str = "mistral-ocr-latest",
) -> str:
    """Extract markdown with advanced OCR options.

    Args:
        file_path: Absolute path to the input file (PDF or image).
        pages: Specific page numbers to process (1-indexed).
        table_format_: Output format for tables ("markdown" or "html").
        model: OCR model to use (default: "mistral-ocr-latest").

    Returns:
        Extracted markdown content as a string.

    Raises:
        PathValidationError: If file_path is invalid.
        MistralOCRAPIError: If the OCR API call fails.
        MistralOCRFileError: If filesystem operations fail.
    """
    validated_path = validate_file_path(file_path)
    response = process_local_file_advanced(
        validated_path,
        include_image_base64=False,
        pages=pages,
        table_format=table_format_,
        model=model,
    )
    page_markdowns = [page.markdown for page in response.pages]
    return "\n\n".join(page_markdowns)


def extract_tables(
    file_path: str,
    table_format: str = "markdown",
) -> list[dict[str, str]]:
    """Extract tables from a PDF or image file.

    Processes the file through OCR and returns all detected tables
    from all pages. Each table includes its id, content, and format.

    Args:
        file_path: Absolute path to the input file (PDF or image).
        table_format: Output format for tables ("markdown" or "html").

    Returns:
        List of dicts, each with:
            - page_index: Page number (0-indexed)
            - table_id: Table identifier
            - content: Table content in the requested format
            - format: Format of the content ("markdown" or "html")

    Raises:
        PathValidationError: If file_path is invalid.
        MistralOCRAPIError: If the OCR API call fails.
        MistralOCRFileError: If filesystem operations fail.
    """
    validated_path = validate_file_path(file_path)
    response = process_local_file_advanced(
        validated_path,
        include_image_base64=False,
        table_format=table_format,
    )

    tables: list[dict[str, Any]] = []
    for page in response.pages:
        if hasattr(page, "tables") and page.tables:
            for tbl in page.tables:
                tables.append({
                    "page_index": page.index,
                    "table_id": tbl.id,
                    "content": tbl.content,
                    "format": tbl.format_,
                })
    return tables


def ocr_status() -> OCRStatusResult:
    """Check Mistral API connectivity and key validity.

    Returns:
        Dict with status and message.
    """
    return check_api_status()


def _create_output_subdirectory(output_dir: Path, file_path: Path) -> Path:
    """Create a unique output subdirectory for a file's extracted content.

    The subdirectory name is based on the file stem (without extension).
    If a directory with that name already exists, appends a timestamp
    in the format _YYYYMMDD_HHMMSS.

    Args:
        output_dir: The validated output directory
        file_path: The validated input file path

    Returns:
        Path to the created output subdirectory
    """
    base_name = file_path.stem
    subdir_path = output_dir / base_name

    # If base directory doesn't exist, just use it
    if not subdir_path.exists():
        subdir_path.mkdir(parents=True, exist_ok=True)
        return subdir_path

    # Directory exists, append timestamp until we find a unique name
    while True:
        timestamp = datetime.datetime.now().strftime("_%Y%m%d_%H%M%S")
        timestamped_path = output_dir / f"{base_name}{timestamp}"

        if not timestamped_path.exists():
            timestamped_path.mkdir(parents=True, exist_ok=True)
            return timestamped_path

        # Extremely unlikely but possible: timestamp collision
        # Sleep a tiny bit and try again
        import time

        time.sleep(0.001)
