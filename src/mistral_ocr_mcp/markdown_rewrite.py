"""Markdown rewrite module for Mistral OCR MCP server.

This module rewrites markdown content to replace embedded base64 image URIs
with relative file paths.
"""

import re
from typing import List, Optional


def rewrite_markdown(
    markdown: str, images: List[dict], output_filenames: Optional[List[str]] = None
) -> str:
    """Rewrite markdown to replace embedded base64 image URIs with relative paths.

    The function uses a deterministic strategy:
    1. First, try exact-match replacement using the image_base64 strings
       returned by the API, paired with output_filenames.
    2. If no output_filenames provided or exact matches fail, fall back to
       sequential regex replacement in document order.

    Args:
        markdown: The original markdown content with embedded base64 images
        images: List of image dictionaries from OCR response with 'image_base64' keys
        output_filenames: Optional list of filenames to use for exact-match replacement.
                         Must match length of images. If None, uses sequential strategy.

    Returns:
        Rewritten markdown with base64 URIs replaced by relative paths like './img_id.ext'

    Raises:
        ValueError: If output_filenames is provided but doesn't match images length
    """
    if output_filenames is not None:
        if len(output_filenames) != len(images):
            raise ValueError(
                f"output_filenames length ({len(output_filenames)}) "
                f"must match images length ({len(images)})"
            )
        # Strategy 1: Exact-match replacement
        return _rewrite_exact_match(markdown, images, output_filenames)
    else:
        # Strategy 2: Sequential regex replacement
        return _rewrite_sequential(markdown, images)


def _rewrite_exact_match(
    markdown: str, images: List[dict], output_filenames: List[str]
) -> str:
    """Rewrite using exact-match replacement of data URIs.

    Args:
        markdown: Original markdown content
        images: List of image dicts with 'image_base64' keys
        output_filenames: List of filenames to use for replacement

    Returns:
        Rewritten markdown
    """
    result = markdown
    for img, filename in zip(images, output_filenames):
        data_uri = img.get("image_base64")
        if data_uri:
            # Replace exact data URI string with relative path
            result = result.replace(data_uri, f"./{filename}")
    return result


def _rewrite_sequential(markdown: str, images: List[dict]) -> str:
    """Rewrite using sequential regex replacement.

    Finds all data:image/...;base64,... patterns and replaces them
    sequentially with ./<id>.ext using extensions determined from the data URI.

    Args:
        markdown: Original markdown content
        images: List of image dicts with 'id' and 'image_base64' keys

    Returns:
        Rewritten markdown
    """
    # Import here to avoid circular dependency
    from .images import get_extension_from_mime, parse_data_uri

    result = markdown

    # Pattern to match data URIs in markdown
    # This matches: data:image/...;base64,... (case-insensitive for mime type)
    data_uri_pattern = re.compile(r'data:image/[^;]+;base64,[^"\'\)]+', re.IGNORECASE)

    # Find all data URIs in the markdown
    matches = list(data_uri_pattern.finditer(result))

    # Replace sequentially in order of appearance
    for i, match in enumerate(matches):
        # Get the image data for this position
        if i < len(images):
            img = images[i]
            image_id = img.get("id", f"image_{i}")
            data_uri = img.get("image_base64", match.group(0))

            try:
                # Parse the data URI to get the extension
                mime_type, _ = parse_data_uri(data_uri)
                ext = get_extension_from_mime(mime_type)
            except Exception:
                # If parsing fails, default to .png
                ext = ".png"

            # Replace this occurrence with relative path
            result = (
                result[: match.start()] + f"./{image_id}{ext}" + result[match.end() :]
            )
        else:
            # More data URIs in markdown than images - skip extras
            break

    return result
