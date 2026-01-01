"""Mistral OCR client adapter.

This module wraps the official `mistralai` SDK behind a small adapter function.
It centralizes:
- client initialization from environment-based config
- the upload -> signed URL -> OCR process flow
- consistent error normalization (FR-6)
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from mistralai import Mistral, models

from .config import load_config


class MistralOCRClientError(RuntimeError):
    """Base exception for Mistral OCR client adapter errors."""


class MistralOCRAPIError(MistralOCRClientError):
    """Raised when the Mistral API returns an error."""


class MistralOCRFileError(MistralOCRClientError):
    """Raised when local filesystem operations fail."""


def process_local_file(
    path: Path,
    *,
    include_image_base64: bool = False,
    client: Optional[Mistral] = None,
) -> models.OCRResponse:
    """Run OCR against a local file path.

    Flow:
      1) Upload file via `client.files.upload(..., purpose="ocr")`
      2) Fetch signed URL via `client.files.get_signed_url(...)`
      3) Call `client.ocr.process(...)` with a `document_url` for PDFs and an
         `image_url` for other supported image formats.

    Args:
        path: Local filesystem path to a PDF or image.
        include_image_base64: Whether to include base64 images in OCR response.
        client: Optional injected Mistral client (useful for unit tests).

    Returns:
        The SDK's OCRResponse.

    Raises:
        MistralOCRAPIError: For SDK/API errors (includes status code + message).
        MistralOCRFileError: For local filesystem errors (includes path + operation).
    """

    config = load_config()
    mistral = client or Mistral(api_key=config.api_key)

    try:
        with path.open("rb") as fh:
            uploaded = mistral.files.upload(
                file={"file_name": path.name, "content": fh},
                purpose="ocr",
            )
    except OSError as e:
        raise MistralOCRFileError(
            f"Filesystem error during open/read for upload: path={path!s}"
        ) from e
    except models.MistralError as e:
        raise MistralOCRAPIError(
            f"Mistral OCR request failed (status={e.status_code}): {e.message}"
        ) from e

    try:
        signed_url = mistral.files.get_signed_url(file_id=uploaded.id)

        is_pdf = path.suffix.lower() == ".pdf"
        if is_pdf:
            document = {"type": "document_url", "document_url": signed_url.url}
        else:
            document = {"type": "image_url", "image_url": signed_url.url}

        return mistral.ocr.process(
            model="mistral-ocr-latest",
            document=document,
            include_image_base64=bool(include_image_base64),
        )
    except models.MistralError as e:
        raise MistralOCRAPIError(
            f"Mistral OCR request failed (status={e.status_code}): {e.message}"
        ) from e
