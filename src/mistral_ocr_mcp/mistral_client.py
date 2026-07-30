"""Mistral OCR client adapter.

This module wraps the official `mistralai` SDK behind a small adapter function.
It centralizes:
- client initialization from environment-based config
- the upload -> signed URL -> OCR process flow
- consistent error normalization (FR-6)
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional

try:
    from mistralai import Mistral, OCRResponse, SDKError

    models = SimpleNamespace(
        OCRResponse=OCRResponse,
        MistralError=Exception,
        SDKError=SDKError,
    )
except ModuleNotFoundError:  # pragma: no cover
    # Allow offline unit tests to inject a fake client without requiring the SDK.
    Mistral = None  # type: ignore[assignment]
    models = SimpleNamespace(  # type: ignore[assignment]
        OCRResponse=Any,
        MistralError=Exception,
    )

from .config import load_config


def _mistral_error_types() -> tuple[type[BaseException], ...]:
    """Return the Mistral SDK exception types we normalize (FR-6.2)."""

    error_types: list[type[BaseException]] = [models.MistralError]
    sdk_error = getattr(models, "SDKError", None)
    if sdk_error is not None:
        error_types.append(sdk_error)
    return tuple(error_types)


def _format_mistral_error(e: BaseException) -> str:
    status_code = getattr(e, "status_code", None)
    message = getattr(e, "message", str(e))

    if status_code is None:
        return f"Mistral OCR request failed: {message}"

    return f"Mistral OCR request failed (status={status_code}): {message}"


class MistralOCRClientError(RuntimeError):
    """Base exception for Mistral OCR client adapter errors."""


class MistralOCRAPIError(MistralOCRClientError):
    """Raised when the Mistral API returns an error."""


class MistralOCRFileError(MistralOCRClientError):
    """Raised when local filesystem operations fail."""


def _get_client(client: Optional[Mistral] = None) -> Mistral:
    """Get a Mistral client, either from the optional injection or from config."""
    if client is not None:
        return client
    if Mistral is None:
        raise MistralOCRClientError(
            "mistralai SDK is required when no client is injected"
        )
    config = load_config()
    return Mistral(api_key=config.api_key)


def _build_document(path: Path, signed_url: str) -> dict:
    """Build the document dict for ocr.process based on file type."""
    is_pdf = path.suffix.lower() == ".pdf"
    if is_pdf:
        return {"type": "document_url", "document_url": signed_url}
    return {"type": "image_url", "image_url": signed_url}


def _build_url_document(url: str) -> dict:
    """Build the document dict for ocr.process from a URL.

    Detects PDF vs image from the URL path suffix.
    """
    from urllib.parse import urlparse

    parsed = urlparse(url)
    path = parsed.path.lower()
    if path.endswith(".pdf"):
        return {"type": "document_url", "document_url": url}
    return {"type": "image_url", "image_url": url}


def _upload_and_process(
    mistral: Mistral,
    path: Path,
    *,
    include_image_base64: bool = False,
    pages: Optional[list[int]] = None,
    table_format: Optional[str] = None,
    model: str = "mistral-ocr-latest",
) -> models.OCRResponse:
    """Upload a local file and process it with OCR."""
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
    except _mistral_error_types() as e:
        raise MistralOCRAPIError(_format_mistral_error(e)) from e

    try:
        signed_url = mistral.files.get_signed_url(file_id=uploaded.id)
        document = _build_document(path, signed_url.url)

        kwargs: dict[str, Any] = {
            "model": model,
            "document": document,
            "include_image_base64": include_image_base64,
        }
        if pages is not None:
            kwargs["pages"] = pages
        if table_format is not None:
            kwargs["table_format"] = table_format

        return mistral.ocr.process(**kwargs)
    except _mistral_error_types() as e:
        raise MistralOCRAPIError(_format_mistral_error(e)) from e


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

    def _process(mistral: Mistral) -> models.OCRResponse:
        return _upload_and_process(mistral, path, include_image_base64=include_image_base64)

    if client is not None:
        return _process(client)

    mistral = _get_client()
    with mistral:
        return _process(mistral)


def process_url(
    url: str,
    *,
    include_image_base64: bool = False,
    pages: Optional[list[int]] = None,
    table_format: Optional[str] = None,
    model: str = "mistral-ocr-latest",
    client: Optional[Mistral] = None,
) -> models.OCRResponse:
    """Run OCR against a publicly accessible URL.

    Unlike `process_local_file`, this skips the upload step and calls the
    OCR API directly with the URL as the document source.

    Args:
        url: Publicly accessible URL to a PDF or image.
        include_image_base64: Whether to include base64 images in OCR response.
        pages: Specific page numbers to process (1-indexed).
        table_format: Output format for tables ("markdown" or "html").
        model: OCR model to use (default: "mistral-ocr-latest").
        client: Optional injected Mistral client (useful for unit tests).

    Returns:
        The SDK's OCRResponse.

    Raises:
        MistralOCRAPIError: For SDK/API errors (includes status code + message).
    """

    def _process(mistral: Mistral) -> models.OCRResponse:
        document = _build_url_document(url)
        kwargs: dict[str, Any] = {
            "model": model,
            "document": document,
            "include_image_base64": include_image_base64,
        }
        if pages is not None:
            kwargs["pages"] = pages
        if table_format is not None:
            kwargs["table_format"] = table_format

        try:
            return mistral.ocr.process(**kwargs)
        except _mistral_error_types() as e:
            raise MistralOCRAPIError(_format_mistral_error(e)) from e

    if client is not None:
        return _process(client)

    mistral = _get_client()
    with mistral:
        return _process(mistral)


def process_local_file_advanced(
    path: Path,
    *,
    include_image_base64: bool = False,
    pages: Optional[list[int]] = None,
    table_format: Optional[str] = None,
    model: str = "mistral-ocr-latest",
    client: Optional[Mistral] = None,
) -> models.OCRResponse:
    """Run OCR against a local file with advanced options.

    Same upload flow as `process_local_file`, but exposes extra OCR
    parameters: page selection, table format, and model choice.

    Args:
        path: Local filesystem path to a PDF or image.
        include_image_base64: Whether to include base64 images in OCR response.
        pages: Specific page numbers to process (1-indexed).
        table_format: Output format for tables ("markdown" or "html").
        model: OCR model to use (default: "mistral-ocr-latest").
        client: Optional injected Mistral client (useful for unit tests).

    Returns:
        The SDK's OCRResponse.

    Raises:
        MistralOCRAPIError: For SDK/API errors (includes status code + message).
        MistralOCRFileError: For local filesystem errors (includes path + operation).
    """

    def _process(mistral: Mistral) -> models.OCRResponse:
        return _upload_and_process(
            mistral, path,
            include_image_base64=include_image_base64,
            pages=pages,
            table_format=table_format,
            model=model,
        )

    if client is not None:
        return _process(client)

    mistral = _get_client()
    with mistral:
        return _process(mistral)


def check_api_status(
    client: Optional[Mistral] = None,
) -> dict[str, Any]:
    """Verify API connectivity and key validity.

    Makes a lightweight API call (list models) to confirm the
    Mistral API key is configured and working.

    Args:
        client: Optional injected Mistral client.

    Returns:
        A dict with keys:
            - status: "ok" or "error"
            - message: Human-readable status description
    """
    def _check(mistral: Mistral) -> dict[str, Any]:
        try:
            mistral.models.list()
            return {
                "status": "ok",
                "message": "Mistral API key is configured and working",
            }
        except _mistral_error_types() as e:
            return {
                "status": "error",
                "message": _format_mistral_error(e),
            }

    if client is not None:
        return _check(client)

    try:
        mistral = _get_client()
        with mistral:
            return _check(mistral)
    except MistralOCRClientError as e:
        return {
            "status": "error",
            "message": str(e),
        }
