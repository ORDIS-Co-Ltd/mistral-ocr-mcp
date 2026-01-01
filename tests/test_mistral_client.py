"""Tests for mistral_client adapter (no network)."""

import sys
from pathlib import Path

import pytest

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mistral_ocr_mcp.config import Config
from mistral_ocr_mcp import mistral_client


class _Uploaded:
    def __init__(self, file_id: str):
        self.id = file_id


class _SignedURL:
    def __init__(self, url: str):
        self.url = url


class _FilesAPI:
    def __init__(self, recorder: dict):
        self._recorder = recorder

    def upload(self, *, file, purpose):
        self._recorder["upload"] = (file, purpose)
        return _Uploaded("uploaded-1")

    def get_signed_url(self, *, file_id):
        self._recorder["get_signed_url"] = file_id
        return _SignedURL("https://example.test/signed")


class _OCRAPI:
    def __init__(self, recorder: dict):
        self._recorder = recorder

    def process(self, *, model, document, include_image_base64):
        self._recorder["process"] = (model, document, include_image_base64)
        return {"ok": True}


def _make_injected_client(*, recorder: dict):
    class _InjectedMistral:
        def __init__(self):
            self.files = _FilesAPI(recorder)
            self.ocr = _OCRAPI(recorder)

    return _InjectedMistral()


def test_process_local_file_pdf_uses_document_url(tmp_path):
    recorder: dict = {}
    injected = _make_injected_client(recorder=recorder)

    input_path = tmp_path / "doc.PDF"
    input_path.write_bytes(b"%PDF-1.4\n")

    res = mistral_client.process_local_file(
        input_path,
        include_image_base64=True,
        client=injected,
    )

    assert res == {"ok": True}

    upload_file, upload_purpose = recorder["upload"]
    assert upload_purpose == "ocr"
    assert upload_file["file_name"] == "doc.PDF"
    assert hasattr(upload_file["content"], "read")
    # The adapter should close the file handle.
    assert upload_file["content"].closed is True

    assert recorder["get_signed_url"] == "uploaded-1"

    model, document, include_image_base64 = recorder["process"]
    assert model == "mistral-ocr-latest"
    assert document == {
        "type": "document_url",
        "document_url": "https://example.test/signed",
    }
    assert include_image_base64 is True


def test_process_local_file_image_uses_image_url(tmp_path):
    recorder: dict = {}
    injected = _make_injected_client(recorder=recorder)

    input_path = tmp_path / "image.png"
    input_path.write_bytes(b"\x89PNG\r\n\x1a\n")

    mistral_client.process_local_file(
        input_path,
        include_image_base64=False,
        client=injected,
    )

    _, document, include_image_base64 = recorder["process"]
    assert document == {"type": "image_url", "image_url": "https://example.test/signed"}
    assert include_image_base64 is False


def test_injected_client_does_not_require_config(monkeypatch, tmp_path):
    recorder: dict = {}
    injected = _make_injected_client(recorder=recorder)

    # If a client is injected, the adapter must not read env/config.
    def _fail_load_config():
        raise AssertionError("load_config called")

    monkeypatch.setattr(mistral_client, "load_config", _fail_load_config)

    input_path = tmp_path / "doc.pdf"
    input_path.write_bytes(b"%PDF-1.4\n")

    mistral_client.process_local_file(input_path, client=injected)


def test_mistral_error_wrapped_with_status_code(monkeypatch, tmp_path):
    recorder: dict = {}

    class _DummyMistralError(Exception):
        def __init__(self, *, status_code: int, message: str):
            super().__init__(message)
            self.status_code = status_code
            self.message = message

    monkeypatch.setattr(mistral_client.models, "MistralError", _DummyMistralError)

    class _FailingFilesAPI(_FilesAPI):
        def upload(self, *, file, purpose):
            raise _DummyMistralError(status_code=401, message="Unauthorized")

    class _FailingMistral:
        def __init__(self, *, api_key: str):
            recorder["api_key"] = api_key
            recorder["closed"] = False
            self.files = _FailingFilesAPI(recorder)
            self.ocr = _OCRAPI(recorder)

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            recorder["closed"] = True
            return False

    monkeypatch.setattr(mistral_client, "Mistral", _FailingMistral)
    monkeypatch.setattr(
        mistral_client,
        "load_config",
        lambda: Config(
            api_key="should-not-leak",
            allowed_dir_original="/allowed",
            allowed_dir_resolved=Path("/allowed"),
        ),
    )

    input_path = tmp_path / "doc.pdf"
    input_path.write_bytes(b"%PDF-1.4\n")

    with pytest.raises(mistral_client.MistralOCRAPIError) as exc_info:
        mistral_client.process_local_file(input_path)

    message = str(exc_info.value)
    assert "status=401" in message
    assert "Unauthorized" in message
    assert "should-not-leak" not in message
    assert recorder["closed"] is True


def test_sdk_error_is_wrapped(monkeypatch, tmp_path):
    recorder: dict = {}

    class _DummySDKError(Exception):
        def __init__(self, *, status_code: int, message: str):
            super().__init__(message)
            self.status_code = status_code
            self.message = message

    # Some mistralai versions expose SDKError; ensure we cover it when present.
    monkeypatch.setattr(
        mistral_client.models, "SDKError", _DummySDKError, raising=False
    )

    class _FailingOCRAPI(_OCRAPI):
        def process(self, *, model, document, include_image_base64):
            raise _DummySDKError(status_code=503, message="Service Unavailable")

    class _InjectedMistral:
        def __init__(self):
            self.files = _FilesAPI(recorder)
            self.ocr = _FailingOCRAPI(recorder)

    input_path = tmp_path / "doc.pdf"
    input_path.write_bytes(b"%PDF-1.4\n")

    with pytest.raises(mistral_client.MistralOCRAPIError) as exc_info:
        mistral_client.process_local_file(input_path, client=_InjectedMistral())

    message = str(exc_info.value)
    assert "status=503" in message
    assert "Service Unavailable" in message
