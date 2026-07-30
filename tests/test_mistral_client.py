"""Tests for mistral_client adapter (no network)."""

import sys
from pathlib import Path

import pytest

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mistral_ocr_mcp import mistral_client
from mistral_ocr_mcp.config import Config


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


def test_process_url_pdf_uses_document_url():
    """Test that process_url uses document_url for PDF URLs."""
    recorder: dict = {}

    class _OCRProcess:
        def process(self, *, model, document, include_image_base64):
            recorder["model"] = model
            recorder["document"] = document
            recorder["include_image_base64"] = include_image_base64
            return {"ok": True}

    class _InjectedMistral:
        def __init__(self):
            self.ocr = _OCRProcess()

    result = mistral_client.process_url(
        "https://example.com/doc.pdf",
        include_image_base64=True,
        client=_InjectedMistral(),
    )

    assert result == {"ok": True}
    assert recorder["model"] == "mistral-ocr-latest"
    assert recorder["document"] == {
        "type": "document_url",
        "document_url": "https://example.com/doc.pdf",
    }
    assert recorder["include_image_base64"] is True


def test_process_url_image_uses_image_url():
    """Test that process_url uses image_url for image URLs."""
    recorder: dict = {}

    class _OCRProcess:
        def process(self, *, model, document, include_image_base64):
            recorder["document"] = document
            return {"ok": True}

    class _InjectedMistral:
        def __init__(self):
            self.ocr = _OCRProcess()

    mistral_client.process_url(
        "https://example.com/image.png",
        client=_InjectedMistral(),
    )

    assert recorder["document"] == {
        "type": "image_url",
        "image_url": "https://example.com/image.png",
    }


def test_process_url_passes_pages_and_table_format():
    """Test that process_url passes pages and table_format to OCR."""
    recorder: dict = {}

    class _OCRProcess:
        def process(self, **kwargs):
            recorder.update(kwargs)
            return {"ok": True}

    class _InjectedMistral:
        def __init__(self):
            self.ocr = _OCRProcess()

    mistral_client.process_url(
        "https://example.com/doc.pdf",
        pages=[1, 3],
        table_format="html",
        model="mistral-ocr-latest",
        client=_InjectedMistral(),
    )

    assert recorder["pages"] == [1, 3]
    assert recorder["table_format"] == "html"
    assert recorder["model"] == "mistral-ocr-latest"


def test_process_url_error_wrapped():
    """Test that process_url wraps SDK errors."""

    class _DummySDKError(Exception):
        def __init__(self, *, status_code: int, message: str):
            super().__init__(message)
            self.status_code = status_code
            self.message = message

    class _OCRProcess:
        def process(self, **kwargs):
            raise _DummySDKError(status_code=400, message="Bad URL")

    class _InjectedMistral:
        def __init__(self):
            self.ocr = _OCRProcess()

    with pytest.raises(mistral_client.MistralOCRAPIError) as exc_info:
        mistral_client.process_url(
            "https://example.com/doc.pdf",
            client=_InjectedMistral(),
        )

    message = str(exc_info.value)
    assert "status=400" in message
    assert "Bad URL" in message


def test_process_local_file_advanced_passes_extra_params(tmp_path):
    """Test that process_local_file_advanced passes pages, table_format, model."""
    recorder: dict = {}

    class _Uploaded:
        def __init__(self):
            self.id = "uploaded-1"

    class _SignedURL:
        def __init__(self):
            self.url = "https://example.test/signed"

    class _FilesAPI:
        def upload(self, *, file, purpose):
            recorder["upload"] = (file, purpose)
            return _Uploaded()

        def get_signed_url(self, *, file_id):
            recorder["get_signed_url"] = file_id
            return _SignedURL()

    class _OCRProcess:
        def process(self, **kwargs):
            recorder["process"] = kwargs
            return {"ok": True}

    class _InjectedMistral:
        def __init__(self):
            self.files = _FilesAPI()
            self.ocr = _OCRProcess()

    input_path = tmp_path / "doc.pdf"
    input_path.write_bytes(b"%PDF-1.4\n")

    result = mistral_client.process_local_file_advanced(
        input_path,
        pages=[2, 4],
        table_format="html",
        model="mistral-ocr-latest",
        client=_InjectedMistral(),
    )

    assert result == {"ok": True}
    assert recorder["process"]["pages"] == [2, 4]
    assert recorder["process"]["table_format"] == "html"
    assert recorder["process"]["model"] == "mistral-ocr-latest"


def test_check_api_status_ok(monkeypatch):
    """Test that check_api_status returns ok when API works."""

    class _InjectedMistral:
        def __init__(self):
            self.called = False

        @property
        def models(self):
            class _Models:
                def list(self):
                    self.called = True

            m = _Models()
            return m

    result = mistral_client.check_api_status(client=_InjectedMistral())

    assert result["status"] == "ok"
    assert "working" in result["message"]


def test_check_api_status_error(monkeypatch):
    """Test that check_api_status returns error status on failure."""

    class _DummyError(Exception):
        def __init__(self, *, status_code: int, message: str):
            super().__init__(message)
            self.status_code = status_code
            self.message = message

    class _Models:
        def list(self):
            raise _DummyError(status_code=401, message="Invalid API key")

    class _InjectedMistral:
        def __init__(self):
            self.models = _Models()

    monkeypatch.setattr(mistral_client.models, "MistralError", _DummyError)

    result = mistral_client.check_api_status(client=_InjectedMistral())

    assert result["status"] == "error"
    assert "401" in result["message"]


def test_check_api_status_no_client_no_config(monkeypatch):
    """Test that check_api_status handles missing config gracefully."""

    def _fail_load_config():
        raise mistral_client.MistralOCRClientError("No API key configured")

    monkeypatch.setattr(mistral_client, "load_config", _fail_load_config)

    result = mistral_client.check_api_status()

    assert result["status"] == "error"
    assert "No API key configured" in result["message"]
