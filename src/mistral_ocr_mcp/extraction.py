"""Extraction orchestration for Mistral OCR MCP server.

This module provides the main extraction functions that orchestrate OCR calls,
image saving, and markdown rewriting.
"""

import datetime
from pathlib import Path
from typing import TypedDict
from urllib.parse import urlparse

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


class ExtractMarkdownWithImagesResult(TypedDict):
    """Result of extracting markdown from a file.

    All fields are required but nullable.  Every return path includes all
    four keys; unused fields are set to ``None``.  This avoids client-side
    validation issues with ``NotRequired`` fields.
    """

    output_directory: str | None
    """Absolute path to the output subdirectory (``None`` when inline result)."""
    markdown_file: str | None
    """Absolute path to the content.md file (``None`` when inline result)."""
    images: list[str] | None
    """Saved image filenames (``None`` when inline result)."""
    result: str | None
    """Extracted markdown content (``None`` when saved to disk)."""


def extract_markdown(
    file_path: str,
    output_dir: str | None = None,
    include_images: bool = False,
) -> ExtractMarkdownWithImagesResult:
    """Extract markdown text from a PDF or image file.

    When ``output_dir`` is provided, saves the extracted markdown to
    ``content.md`` inside a named subdirectory. When ``include_images``
    is also ``True``, saves embedded images alongside the markdown file.
    Otherwise returns the markdown text inline.

    Args:
        file_path: Absolute path to the input file (PDF or image)
        output_dir: Absolute path to an existing output directory (must be
            within allowed dir). When set, saves markdown to disk at
            ``<output_dir>/<file_stem>/content.md``.
        include_images: When True (requires output_dir), save images to
            disk and rewrite markdown with relative image links.

    Returns:
        When output_dir is not set:
            result: Extracted markdown content
        When output_dir is set (with or without images):
            output_directory: Absolute path to the output subdirectory
            markdown_file: Absolute path to the content.md file
            images: List of saved image filenames (empty if include_images
                is False)

    Raises:
        PathValidationError: If file_path or output_dir is invalid
        MistralOCRAPIError: If the OCR API call fails
        MistralOCRFileError: If filesystem operations fail
    """
    if include_images and output_dir:
        return _extract_markdown_with_images(file_path, output_dir)

    validated_path = validate_file_path(file_path)
    response = process_local_file(validated_path, include_image_base64=False)
    page_markdowns = [page.markdown for page in response.pages]
    markdown = "\n\n".join(page_markdowns)

    if output_dir:
        config = load_config()
        validated_output_dir = validate_output_dir(
            output_dir,
            config.allowed_dir_resolved,
            config.allowed_dir_original,
        )
        output_subdir = _create_output_subdirectory(
            validated_output_dir, validated_path
        )
        markdown_file_path = output_subdir / "content.md"
        markdown_file_path.write_text(markdown, encoding="utf-8")
        return {
            "output_directory": str(output_subdir),
            "markdown_file": str(markdown_file_path),
            "images": [],
            "result": None,
        }

    return {
        "output_directory": None,
        "markdown_file": None,
        "images": None,
        "result": markdown,
    }


def _extract_markdown_with_images(
    file_path: str, output_dir: str
) -> ExtractMarkdownWithImagesResult:
    """Extract markdown with embedded images and save them as separate files.

    Args:
        file_path: Absolute path to the input file (PDF or image)
        output_dir: Absolute path to the output directory (must be within allowed dir)

    Returns:
        Dictionary with:
            - output_directory: Absolute path to the output subdirectory
            - markdown_file: Absolute path to the content.md file
            - images: List of saved image filenames
    """
    config = load_config()
    validated_file_path = validate_file_path(file_path)
    validated_output_dir = validate_output_dir(
        output_dir,
        config.allowed_dir_resolved,
        config.allowed_dir_original,
    )
    output_subdir = _create_output_subdirectory(
        validated_output_dir, validated_file_path
    )
    response = process_local_file(validated_file_path, include_image_base64=True)

    images: list[dict] = []
    for page in response.pages:
        if hasattr(page, "images") and page.images:
            images.extend(
                [
                    img.model_dump() if hasattr(img, "model_dump") else img
                    for img in page.images
                ]
            )

    saved_filenames = save_images(output_subdir, images)
    page_markdowns = [page.markdown for page in response.pages]
    markdown_content = "\n\n".join(page_markdowns)
    rewritten_markdown = rewrite_markdown(markdown_content, images, saved_filenames)
    markdown_file_path = output_subdir / "content.md"
    markdown_file_path.write_text(rewritten_markdown, encoding="utf-8")

    return {
        "output_directory": str(output_subdir),
        "markdown_file": str(markdown_file_path),
        "images": saved_filenames,
        "result": None,
    }


class OCRStatusResult(TypedDict):
    """Result of an API status check."""

    status: str
    """"ok" or "error"."""
    message: str
    """Human-readable status description."""


def extract_from_url(
    file_url: str,
    output_dir: str | None = None,
    include_images: bool = False,
) -> ExtractMarkdownWithImagesResult:
    """Extract markdown from a publicly accessible URL.

    When ``output_dir`` is provided, saves the extracted markdown to
    ``content.md`` inside a named subdirectory. When ``include_images``
    is also ``True``, saves embedded images alongside the markdown file.
    Otherwise returns the markdown text inline.

    Args:
        file_url: Publicly accessible URL to a PDF or image.
        output_dir: Absolute path to an existing output directory (must be
            within allowed dir). When set, saves markdown to disk at
            ``<output_dir>/<url_stem>/content.md``.
        include_images: When True (requires output_dir), save images to
            disk and rewrite markdown with relative image links.

    Returns:
        When output_dir is not set:
            result: Extracted markdown content
        When output_dir is set (with or without images):
            output_directory: Absolute path to the output subdirectory
            markdown_file: Absolute path to the content.md file
            images: List of saved image filenames (empty if include_images
                is False)

    Raises:
        MistralOCRAPIError: If the OCR API call fails.
    """
    if include_images and output_dir:
        return _extract_from_url_with_images(file_url, output_dir)

    response = process_url(file_url, include_image_base64=False)
    page_markdowns = [page.markdown for page in response.pages]
    markdown = "\n\n".join(page_markdowns)

    if output_dir:
        config = load_config()
        validated_output_dir = validate_output_dir(
            output_dir,
            config.allowed_dir_resolved,
            config.allowed_dir_original,
        )
        parsed = urlparse(file_url)
        url_path = Path(parsed.path)
        subdir_name = url_path.stem or "document"
        output_subdir = _create_output_subdirectory(
            validated_output_dir, name=subdir_name
        )
        markdown_file_path = output_subdir / "content.md"
        markdown_file_path.write_text(markdown, encoding="utf-8")
        return {
            "output_directory": str(output_subdir),
            "markdown_file": str(markdown_file_path),
            "images": [],
            "result": None,
        }

    return {
        "output_directory": None,
        "markdown_file": None,
        "images": None,
        "result": markdown,
    }


def extract_markdown_advanced(
    file_path: str,
    pages: list[int] | None = None,
    table_format_: str | None = None,
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


def _extract_from_url_with_images(
    file_url: str,
    output_dir: str,
) -> ExtractMarkdownWithImagesResult:
    """Extract markdown from a URL and save embedded images to disk.

    Args:
        file_url: Publicly accessible URL to a PDF or image.
        output_dir: Absolute path to an existing output directory (must be within
            the allowed directory from config).

    Returns:
        Dictionary with:
            - output_directory: Absolute path to the output subdirectory
            - markdown_file: Absolute path to the content.md file
            - images: List of saved image filenames (not full paths)
    """
    config = load_config()

    validated_output_dir = validate_output_dir(
        output_dir,
        config.allowed_dir_resolved,
        config.allowed_dir_original,
    )

    # Derive a name from the URL for the output subdirectory
    parsed = urlparse(file_url)
    url_path = Path(parsed.path)
    subdir_name = url_path.stem or "document"

    output_subdir = _create_output_subdirectory(validated_output_dir, name=subdir_name)

    response = process_url(file_url, include_image_base64=True)

    # Extract images from response
    images: list[dict] = []
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
        "result": None,
    }


def ocr_status() -> OCRStatusResult:
    """Check Mistral API connectivity and key validity.

    Returns:
        Dict with status and message.
    """
    return check_api_status()


def _create_output_subdirectory(
    output_dir: Path,
    file_path: Path | None = None,
    *,
    name: str | None = None,
) -> Path:
    """Create a unique output subdirectory for a file's extracted content.

    The subdirectory name is based on the file stem (without extension), or
    the explicit ``name`` parameter when ``file_path`` is omitted.
    If a directory with that name already exists, appends a timestamp
    in the format _YYYYMMDD_HHMMSS.

    Args:
        output_dir: The validated output directory
        file_path: The validated input file path (ignored when ``name`` is set)
        name: Explicit directory name (takes precedence over file_path)

    Returns:
        Path to the created output subdirectory
    """
    base_name = name if name is not None else file_path.stem
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
