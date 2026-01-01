"""Image handling for Mistral OCR MCP server.

This module handles parsing and saving base64-encoded images from OCR responses.
"""

import base64
import binascii
import re
from pathlib import Path
from typing import List, Tuple


class ImageError(Exception):
    """Exception raised for image processing errors."""

    pass


def parse_data_uri(data_uri: str) -> Tuple[str, str]:
    """Parse a data URI to extract MIME type and raw base64 data.

    Args:
        data_uri: Data URI string like `data:image/jpeg;base64,<...>`

    Returns:
        Tuple of (mime_type, raw_base64_string)

    Raises:
        ImageError: If the data URI is invalid or missing required parts
    """
    if not data_uri:
        raise ImageError("data_uri cannot be empty")

    # Match data URI pattern: data:<mime>;base64,<data>
    match = re.match(r"^data:([^;]*);base64,(.*)$", data_uri)
    if not match:
        raise ImageError(
            f"Invalid data URI format, expected 'data:<mime>;base64,<data>': {data_uri[:50]}..."
        )

    mime_type = match.group(1)
    raw_b64 = match.group(2)

    if not mime_type:
        raise ImageError(f"Missing MIME type in data URI: {data_uri[:50]}...")

    if not raw_b64:
        raise ImageError(f"Missing base64 data in data URI: {data_uri[:50]}...")

    return mime_type, raw_b64


def get_extension_from_mime(mime_type: str) -> str:
    """Determine file extension from MIME type.

    Args:
        mime_type: MIME type string like 'image/jpeg'

    Returns:
        File extension including the dot (e.g., '.jpeg')

    Returns:
        str: File extension with leading dot
    """
    # Mapping of common image MIME types to extensions
    mime_to_ext = {
        "image/jpeg": ".jpeg",
        "image/jpg": ".jpg",
        "image/png": ".png",
        "image/webp": ".webp",
        "image/gif": ".gif",
    }

    # Normalize MIME type to lowercase
    mime_lower = mime_type.lower()

    # Default to .png if unknown
    return mime_to_ext.get(mime_lower, ".png")


def save_base64_image(output_dir: Path, image_id: str, data_uri: str) -> str:
    """Decode a base64 data URI and save it as an image file.

    Args:
        output_dir: Directory to save the image in
        image_id: Identifier for the image (used as filename; may already include an extension)
        data_uri: Base64 data URI string

    Returns:
        Filename of the saved image (without directory path)

    Raises:
        ImageError: If parsing, decoding, or saving fails
    """
    try:
        mime_type, raw_b64 = parse_data_uri(data_uri)
        ext = get_extension_from_mime(mime_type)

        # Sanitize image_id to prevent path traversal attacks
        # First, remove any path separators and null bytes
        sanitized_id = image_id.replace("\0", "_")
        # Remove any directory separators and parent directory references
        sanitized_id = sanitized_id.replace("/", "_").replace("\\", "_")
        sanitized_id = sanitized_id.replace("..", "__")

        # Only allow alphanumeric characters, underscores, hyphens, and dots
        sanitized_id = re.sub(r"[^\w\-.]", "_", sanitized_id)

        # Ensure it doesn't start with a dot (hidden file) or dash (flag)
        if sanitized_id.startswith(".") or sanitized_id.startswith("-"):
            sanitized_id = "_" + sanitized_id.lstrip(".-")

        # Remove any remaining path traversal patterns
        sanitized_id = re.sub(r"\.\.+", "__", sanitized_id)

        # Limit length to prevent filesystem issues
        max_id_length = 200  # Leave room for extension
        if len(sanitized_id) > max_id_length:
            sanitized_id = sanitized_id[:max_id_length]

        # Don't append extension if image_id already ends with it (case-insensitive)
        # This prevents duplicate extensions like image.jpeg.jpeg
        if sanitized_id.lower().endswith(ext.lower()):
            filename = sanitized_id
        else:
            filename = f"{sanitized_id}{ext}"

        output_path = output_dir / filename

        # Decode base64
        image_data = base64.b64decode(raw_b64)

        # Ensure output directory exists
        output_dir.mkdir(parents=True, exist_ok=True)

        # Write file
        output_path.write_bytes(image_data)

        return filename

    except (binascii.Error, ValueError) as e:
        raise ImageError(
            f"Failed to decode base64 image data for image_id={image_id}: {e}"
        ) from e


def save_images(output_dir: Path, images: List[dict]) -> List[str]:
    """Save multiple base64-encoded images from OCR response.

    Args:
        output_dir: Directory to save images in
        images: List of image dictionaries with 'id' and 'image_base64' keys

    Returns:
        List of saved filenames in the same order as input images

    Raises:
        ImageError: If any image fails to parse, decode, or save
    """
    saved_filenames = []

    for img in images:
        image_id = img.get("id")
        image_base64 = img.get("image_base64")

        if image_id is None:
            raise ImageError("Image missing 'id' field")

        if image_base64 is None:
            raise ImageError(f"Image '{image_id}' missing 'image_base64' field")

        filename = save_base64_image(output_dir, image_id, image_base64)
        saved_filenames.append(filename)

    return saved_filenames
