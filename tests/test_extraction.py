"""Tests for extraction module."""

import sys
from pathlib import Path
import pytest
from unittest.mock import Mock

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mistral_ocr_mcp.extraction import (
    _create_output_subdirectory,
    extract_from_url,
    extract_markdown,
    extract_markdown_advanced,
    extract_markdown_with_images,
    extract_tables,
    ocr_status,
)
from mistral_ocr_mcp.path_sandbox import PathValidationError


class TestCreateOutputSubdirectory:
    """Tests for _create_output_subdirectory function."""

    def test_creates_directory_from_file_stem(self, tmp_path):
        """Test that directory is created using file stem."""
        file_path = tmp_path / "document.pdf"
        file_path.write_text("test")

        output_dir = tmp_path / "output"
        output_dir.mkdir()

        subdir = _create_output_subdirectory(output_dir, file_path)

        assert subdir.name == "document"
        assert subdir.exists()
        assert subdir.is_dir()

    def test_creates_nested_subdirectory(self, tmp_path):
        """Test that subdirectory is created within output_dir."""
        file_path = tmp_path / "mydoc.pdf"
        file_path.write_text("test")

        output_dir = tmp_path / "output" / "nested"
        output_dir.mkdir(parents=True)

        subdir = _create_output_subdirectory(output_dir, file_path)

        assert subdir.parent == output_dir
        assert subdir.name == "mydoc"
        assert subdir.exists()

    def test_appends_timestamp_if_directory_exists(self, tmp_path):
        """Test that timestamp is appended if directory already exists."""
        file_path = tmp_path / "document.pdf"
        file_path.write_text("test")

        output_dir = tmp_path / "output"
        output_dir.mkdir()

        # Pre-create the directory
        existing_dir = output_dir / "document"
        existing_dir.mkdir()

        subdir = _create_output_subdirectory(output_dir, file_path)

        # Should have timestamp suffix
        assert subdir.name.startswith("document_")
        assert subdir.exists()
        # Verify it's a different directory
        assert subdir != existing_dir

    def test_timestamp_format(self, tmp_path):
        """Test that timestamp is in format _YYYYMMDD_HHMMSS."""
        file_path = tmp_path / "doc.pdf"
        file_path.write_text("test")

        output_dir = tmp_path / "output"
        output_dir.mkdir()

        # Pre-create to force timestamp
        existing_dir = output_dir / "doc"
        existing_dir.mkdir()

        subdir = _create_output_subdirectory(output_dir, file_path)

        # Check format: <stem>_YYYYMMDD_HHMMSS
        import re

        match = re.match(r"^doc_(\d{8})_(\d{6})$", subdir.name)
        assert match is not None, (
            f"Directory name '{subdir.name}' doesn't match expected format"
        )

        date_part = match.group(1)  # YYYYMMDD
        time_part = match.group(2)  # HHMMSS

        # Verify it's a valid date and time
        assert date_part.isdigit()
        assert time_part.isdigit()
        assert len(date_part) == 8
        assert len(time_part) == 6

    def test_handles_timestamp_collision(self, tmp_path):
        """Test that timestamp collision is handled by looping."""
        file_path = tmp_path / "doc.pdf"
        file_path.write_text("test")

        output_dir = tmp_path / "output"
        output_dir.mkdir()

        # Pre-create the base directory to force timestamp
        existing_doc_dir = output_dir / "doc"
        existing_doc_dir.mkdir()

        # Create subdirectory - should have timestamp suffix
        subdir = _create_output_subdirectory(output_dir, file_path)

        assert subdir.name.startswith("doc_")
        assert subdir.exists()
        # Should be different from the base directory
        assert subdir != existing_doc_dir

    def test_file_with_dots_in_name(self, tmp_path):
        """Test handling files with multiple dots in name."""
        file_path = tmp_path / "my.document.v2.pdf"
        file_path.write_text("test")

        output_dir = tmp_path / "output"
        output_dir.mkdir()

        subdir = _create_output_subdirectory(output_dir, file_path)

        # Should use stem (everything before last dot)
        assert subdir.name == "my.document.v2"
        assert subdir.exists()

    def test_file_without_extension(self, tmp_path):
        """Test handling file without extension (should work with empty stem)."""
        file_path = tmp_path / "noextension"
        file_path.write_text("test")

        output_dir = tmp_path / "output"
        output_dir.mkdir()

        subdir = _create_output_subdirectory(output_dir, file_path)

        # Should use the full filename as stem
        assert subdir.name == "noextension"
        assert subdir.exists()


class TestExtractMarkdown:
    """Tests for extract_markdown function."""

    def test_valid_pdf_file(self, tmp_path, monkeypatch):
        """Test extracting markdown from a valid PDF file."""
        # Create a test PDF file
        test_file = tmp_path / "test.pdf"
        test_file.write_bytes(b"%PDF-1.4\n%EOF")

        # Mock process_local_file
        mock_response = Mock()
        mock_page1 = Mock()
        mock_page1.markdown = "# Page 1\n\nContent of page 1"
        mock_page2 = Mock()
        mock_page2.markdown = "# Page 2\n\nContent of page 2"
        mock_response.pages = [mock_page1, mock_page2]

        def mock_process(path, **kwargs):
            return mock_response

        import mistral_ocr_mcp.extraction

        monkeypatch.setattr(
            mistral_ocr_mcp.extraction, "process_local_file", mock_process
        )

        result = extract_markdown(str(test_file))

        assert "# Page 1" in result
        assert "# Page 2" in result
        assert "\n\n" in result  # Pages are joined with double newline

    def test_calls_ocr_without_images(self, tmp_path, monkeypatch):
        """Test that OCR is called with include_image_base64=False."""
        test_file = tmp_path / "test.pdf"
        test_file.write_bytes(b"%PDF-1.4\n%EOF")

        mock_response = Mock()
        mock_page = Mock()
        mock_page.markdown = "Content"
        mock_response.pages = [mock_page]

        # Track call arguments
        calls = []

        def mock_process(path, **kwargs):
            calls.append((path, kwargs))
            return mock_response

        import mistral_ocr_mcp.extraction

        monkeypatch.setattr(
            mistral_ocr_mcp.extraction, "process_local_file", mock_process
        )

        extract_markdown(str(test_file))

        assert len(calls) == 1
        assert calls[0][1]["include_image_base64"] is False

    def test_invalid_file_path_raises_error(self, tmp_path, monkeypatch):
        """Test that invalid file path raises PathValidationError."""
        import mistral_ocr_mcp.extraction

        # We need to make sure validate_file_path is actually called
        # Since it's imported, we need to patch it in the module
        def mock_validate(path):
            raise PathValidationError("Invalid path")

        monkeypatch.setattr(
            mistral_ocr_mcp.extraction, "validate_file_path", mock_validate
        )

        with pytest.raises(PathValidationError) as exc_info:
            extract_markdown("/nonexistent/file.pdf")

        assert "Invalid path" in str(exc_info.value)

    def test_unsupported_extension_raises_error(self, tmp_path):
        """Test that unsupported file extension raises PathValidationError."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("test")

        with pytest.raises(PathValidationError) as exc_info:
            extract_markdown(str(test_file))

        assert "unsupported file type" in str(exc_info.value)


class TestExtractMarkdownWithImages:
    """Tests for extract_markdown_with_images function."""

    def test_full_extraction_workflow(self, tmp_path, monkeypatch):
        """Test complete workflow: OCR, save images, rewrite markdown, save file."""
        # Create test files
        test_file = tmp_path / "document.pdf"
        test_file.write_bytes(b"%PDF-1.4\n%EOF")

        output_dir = tmp_path / "output"
        output_dir.mkdir()

        # Mock OCR response
        mock_response = Mock()
        mock_page = Mock()
        mock_page.markdown = "# Page 1\n\nContent"
        mock_page.images = []
        mock_response.pages = [mock_page]

        def mock_process(path, **kwargs):
            return mock_response

        import mistral_ocr_mcp.extraction

        monkeypatch.setattr(
            mistral_ocr_mcp.extraction, "process_local_file", mock_process
        )

        # Mock save_images
        def mock_save_images(out_dir, images):
            return []

        monkeypatch.setattr(mistral_ocr_mcp.extraction, "save_images", mock_save_images)

        # Mock config
        mock_config = Mock()
        mock_config.allowed_dir_resolved = tmp_path
        mock_config.allowed_dir_original = str(tmp_path)
        monkeypatch.setattr(
            mistral_ocr_mcp.extraction, "load_config", lambda: mock_config
        )

        result = extract_markdown_with_images(str(test_file), str(output_dir))

        # Verify return structure
        assert "output_directory" in result
        assert "markdown_file" in result
        assert "images" in result
        assert isinstance(result["images"], list)

        # Verify files exist
        output_subdir = Path(result["output_directory"])
        assert output_subdir.exists()
        markdown_file = Path(result["markdown_file"])
        assert markdown_file.exists()
        assert markdown_file.name == "content.md"

        # Verify markdown content is preserved
        content = markdown_file.read_text()
        assert "# Page 1" in content
        assert "Content" in content

    def test_creates_unique_output_subdirectory(self, tmp_path, monkeypatch):
        """Test that unique output subdirectory is created."""
        test_file = tmp_path / "doc.pdf"
        test_file.write_bytes(b"%PDF-1.4\n%EOF")

        output_dir = tmp_path / "output"
        output_dir.mkdir()

        # Pre-create the "doc" directory
        existing_doc_dir = output_dir / "doc"
        existing_doc_dir.mkdir()

        # Mock OCR response
        mock_response = Mock()
        mock_page = Mock()
        mock_page.markdown = "Content"
        mock_page.images = []
        mock_response.pages = [mock_page]

        import mistral_ocr_mcp.extraction

        monkeypatch.setattr(
            mistral_ocr_mcp.extraction,
            "process_local_file",
            lambda path, **kwargs: mock_response,
        )

        # Mock save_images
        monkeypatch.setattr(
            mistral_ocr_mcp.extraction, "save_images", lambda out_dir, images: []
        )

        # Mock config
        mock_config = Mock()
        mock_config.allowed_dir_resolved = tmp_path
        mock_config.allowed_dir_original = str(tmp_path)
        monkeypatch.setattr(
            mistral_ocr_mcp.extraction, "load_config", lambda: mock_config
        )

        result = extract_markdown_with_images(str(test_file), str(output_dir))

        output_subdir = Path(result["output_directory"])

        # Should be a different directory (with timestamp)
        assert output_subdir.name.startswith("doc_")
        assert output_subdir != existing_doc_dir
        assert output_subdir.exists()

    def test_calls_ocr_with_images(self, tmp_path, monkeypatch):
        """Test that OCR is called with include_image_base64=True."""
        test_file = tmp_path / "test.pdf"
        test_file.write_bytes(b"%PDF-1.4\n%EOF")

        output_dir = tmp_path / "output"
        output_dir.mkdir()

        # Track call arguments
        calls = []

        mock_response = Mock()
        mock_page = Mock()
        mock_page.markdown = "Content"
        mock_page.images = []
        mock_response.pages = [mock_page]

        def mock_process(path, **kwargs):
            calls.append((path, kwargs))
            return mock_response

        import mistral_ocr_mcp.extraction

        monkeypatch.setattr(
            mistral_ocr_mcp.extraction, "process_local_file", mock_process
        )

        # Mock save_images
        monkeypatch.setattr(
            mistral_ocr_mcp.extraction, "save_images", lambda out_dir, images: []
        )

        # Mock config
        mock_config = Mock()
        mock_config.allowed_dir_resolved = tmp_path
        mock_config.allowed_dir_original = str(tmp_path)
        monkeypatch.setattr(
            mistral_ocr_mcp.extraction, "load_config", lambda: mock_config
        )

        extract_markdown_with_images(str(test_file), str(output_dir))

        assert len(calls) == 1
        assert calls[0][1]["include_image_base64"] is True

    def test_invalid_file_path_raises_error(self, tmp_path, monkeypatch):
        """Test that invalid file path raises PathValidationError."""
        import mistral_ocr_mcp.extraction

        # Mock config first (called before validate_file_path)
        mock_config = Mock()
        mock_config.allowed_dir_resolved = tmp_path
        mock_config.allowed_dir_original = str(tmp_path)
        monkeypatch.setattr(
            mistral_ocr_mcp.extraction, "load_config", lambda: mock_config
        )

        def mock_validate(path):
            raise PathValidationError("Invalid file path")

        monkeypatch.setattr(
            mistral_ocr_mcp.extraction, "validate_file_path", mock_validate
        )

        with pytest.raises(PathValidationError) as exc_info:
            extract_markdown_with_images("/nonexistent/file.pdf", str(tmp_path))

        assert "Invalid file path" in str(exc_info.value)

    def test_invalid_output_dir_raises_error(self, tmp_path, monkeypatch):
        """Test that invalid output directory raises PathValidationError."""
        test_file = tmp_path / "test.pdf"
        test_file.write_bytes(b"%PDF-1.4\n%EOF")

        import mistral_ocr_mcp.extraction

        # Mock config first (called before validate_output_dir)
        mock_config = Mock()
        mock_config.allowed_dir_resolved = tmp_path
        mock_config.allowed_dir_original = str(tmp_path)
        monkeypatch.setattr(
            mistral_ocr_mcp.extraction, "load_config", lambda: mock_config
        )

        # Mock file validation to succeed
        def mock_validate_file(path):
            return Path(path).resolve(strict=True)

        monkeypatch.setattr(
            mistral_ocr_mcp.extraction, "validate_file_path", mock_validate_file
        )

        # Mock output dir validation to fail
        def mock_validate_output(path, *args, **kwargs):
            raise PathValidationError("Invalid output dir")

        monkeypatch.setattr(
            mistral_ocr_mcp.extraction, "validate_output_dir", mock_validate_output
        )

        with pytest.raises(PathValidationError) as exc_info:
            extract_markdown_with_images(str(test_file), str(tmp_path))

        assert "Invalid output dir" in str(exc_info.value)

    def test_returns_absolute_paths(self, tmp_path, monkeypatch):
        """Test that output_directory and markdown_file are absolute paths."""
        test_file = tmp_path / "doc.pdf"
        test_file.write_bytes(b"%PDF-1.4\n%EOF")

        output_dir = tmp_path / "output"
        output_dir.mkdir()

        # Mock OCR response
        mock_response = Mock()
        mock_page = Mock()
        mock_page.markdown = "Content"
        mock_page.images = []
        mock_response.pages = [mock_page]

        import mistral_ocr_mcp.extraction

        monkeypatch.setattr(
            mistral_ocr_mcp.extraction,
            "process_local_file",
            lambda path, **kwargs: mock_response,
        )

        # Mock save_images
        monkeypatch.setattr(
            mistral_ocr_mcp.extraction, "save_images", lambda out_dir, images: []
        )

        # Mock config
        mock_config = Mock()
        mock_config.allowed_dir_resolved = tmp_path
        mock_config.allowed_dir_original = str(tmp_path)
        monkeypatch.setattr(
            mistral_ocr_mcp.extraction, "load_config", lambda: mock_config
        )

        result = extract_markdown_with_images(str(test_file), str(output_dir))

        # Verify paths are absolute
        assert Path(result["output_directory"]).is_absolute()
        assert Path(result["markdown_file"]).is_absolute()

    def test_handles_multiple_pages(self, tmp_path, monkeypatch):
        """Test handling of multi-page documents."""
        test_file = tmp_path / "doc.pdf"
        test_file.write_bytes(b"%PDF-1.4\n%EOF")

        output_dir = tmp_path / "output"
        output_dir.mkdir()

        # Mock OCR response with multiple pages
        mock_response = Mock()
        mock_page1 = Mock()
        mock_page1.markdown = "# Page 1"
        mock_page1.images = []
        mock_page2 = Mock()
        mock_page2.markdown = "# Page 2"
        mock_page2.images = []
        mock_page3 = Mock()
        mock_page3.markdown = "# Page 3"
        mock_page3.images = []
        mock_response.pages = [mock_page1, mock_page2, mock_page3]

        import mistral_ocr_mcp.extraction

        monkeypatch.setattr(
            mistral_ocr_mcp.extraction,
            "process_local_file",
            lambda path, **kwargs: mock_response,
        )

        # Mock save_images
        monkeypatch.setattr(
            mistral_ocr_mcp.extraction, "save_images", lambda out_dir, images: []
        )

        # Mock config
        mock_config = Mock()
        mock_config.allowed_dir_resolved = tmp_path
        mock_config.allowed_dir_original = str(tmp_path)
        monkeypatch.setattr(
            mistral_ocr_mcp.extraction, "load_config", lambda: mock_config
        )

        result = extract_markdown_with_images(str(test_file), str(output_dir))

        # Verify all pages are in the markdown
        content = Path(result["markdown_file"]).read_text()
        assert "# Page 1" in content
        assert "# Page 2" in content
        assert "# Page 3" in content


class TestExtractFromUrl:
    """Tests for extract_from_url function."""

    def test_calls_process_url(self, monkeypatch):
        """Test that extract_from_url calls process_url."""
        mock_response = Mock()
        mock_page = Mock()
        mock_page.markdown = "# URL Content"
        mock_response.pages = [mock_page]

        calls = []

        def mock_process(url, **kwargs):
            calls.append((url, kwargs))
            return mock_response

        import mistral_ocr_mcp.extraction

        monkeypatch.setattr(mistral_ocr_mcp.extraction, "process_url", mock_process)

        result = extract_from_url("https://example.com/doc.pdf")

        assert result == "# URL Content"
        assert len(calls) == 1
        assert calls[0][0] == "https://example.com/doc.pdf"

    def test_joins_multiple_pages(self, monkeypatch):
        """Test that multiple pages are joined with double newline."""
        mock_response = Mock()
        mock_page1 = Mock()
        mock_page1.markdown = "# Page 1"
        mock_page2 = Mock()
        mock_page2.markdown = "# Page 2"
        mock_response.pages = [mock_page1, mock_page2]

        import mistral_ocr_mcp.extraction

        monkeypatch.setattr(
            mistral_ocr_mcp.extraction,
            "process_url",
            lambda url, **kwargs: mock_response,
        )

        result = extract_from_url("https://example.com/doc.pdf")

        assert "# Page 1" in result
        assert "# Page 2" in result
        assert "\n\n" in result

    def test_passes_include_image_base64(self, monkeypatch):
        """Test that include_image_base64 is passed to process_url."""
        mock_response = Mock()
        mock_page = Mock()
        mock_page.markdown = "Content"
        mock_response.pages = [mock_page]

        calls = []

        def mock_process(url, **kwargs):
            calls.append((url, kwargs))
            return mock_response

        import mistral_ocr_mcp.extraction

        monkeypatch.setattr(mistral_ocr_mcp.extraction, "process_url", mock_process)

        extract_from_url("https://example.com/img.png", include_image_base64=True)

        assert len(calls) == 1
        assert calls[0][1]["include_image_base64"] is True


class TestExtractMarkdownAdvanced:
    """Tests for extract_markdown_advanced function."""

    def test_calls_process_local_file_advanced(self, tmp_path, monkeypatch):
        """Test that extract_markdown_advanced calls process_local_file_advanced."""
        test_file = tmp_path / "test.pdf"
        test_file.write_bytes(b"%PDF-1.4\n%EOF")

        mock_response = Mock()
        mock_page = Mock()
        mock_page.markdown = "# Advanced Content"
        mock_response.pages = [mock_page]

        calls = []

        def mock_process(path, **kwargs):
            calls.append((path, kwargs))
            return mock_response

        import mistral_ocr_mcp.extraction

        monkeypatch.setattr(
            mistral_ocr_mcp.extraction, "process_local_file_advanced", mock_process
        )

        result = extract_markdown_advanced(str(test_file))

        assert result == "# Advanced Content"
        assert len(calls) == 1
        assert calls[0][1]["include_image_base64"] is False

    def test_passes_optional_params(self, tmp_path, monkeypatch):
        """Test that optional params are passed to process_local_file_advanced."""
        test_file = tmp_path / "test.pdf"
        test_file.write_bytes(b"%PDF-1.4\n%EOF")

        mock_response = Mock()
        mock_page = Mock()
        mock_page.markdown = "Content"
        mock_response.pages = [mock_page]

        calls = []

        def mock_process(path, **kwargs):
            calls.append((path, kwargs))
            return mock_response

        import mistral_ocr_mcp.extraction

        monkeypatch.setattr(
            mistral_ocr_mcp.extraction, "process_local_file_advanced", mock_process
        )

        extract_markdown_advanced(
            str(test_file),
            pages=[1, 3],
            table_format_="html",
            model="mistral-ocr-latest",
        )

        assert len(calls) == 1
        assert calls[0][1]["pages"] == [1, 3]
        assert calls[0][1]["table_format"] == "html"
        assert calls[0][1]["model"] == "mistral-ocr-latest"

    def test_validates_file_path(self, tmp_path):
        """Test that invalid file path raises PathValidationError."""
        with pytest.raises(PathValidationError):
            extract_markdown_advanced("/nonexistent/file.pdf")


class TestExtractTables:
    """Tests for extract_tables function."""

    def test_returns_tables_from_all_pages(self, tmp_path, monkeypatch):
        """Test that tables from all pages are returned."""
        test_file = tmp_path / "test.pdf"
        test_file.write_bytes(b"%PDF-1.4\n%EOF")

        mock_response = Mock()

        # Page 1 with two tables
        mock_tbl1 = Mock()
        mock_tbl1.id = "tbl_1"
        mock_tbl1.content = "| A | B |"
        mock_tbl1.format_ = "markdown"

        mock_tbl2 = Mock()
        mock_tbl2.id = "tbl_2"
        mock_tbl2.content = "| C | D |"
        mock_tbl2.format_ = "markdown"

        mock_page1 = Mock()
        mock_page1.index = 0
        mock_page1.tables = [mock_tbl1, mock_tbl2]

        # Page 2 with one table
        mock_tbl3 = Mock()
        mock_tbl3.id = "tbl_3"
        mock_tbl3.content = "| E | F |"
        mock_tbl3.format_ = "html"

        mock_page2 = Mock()
        mock_page2.index = 1
        mock_page2.tables = [mock_tbl3]

        mock_response.pages = [mock_page1, mock_page2]

        import mistral_ocr_mcp.extraction

        monkeypatch.setattr(
            mistral_ocr_mcp.extraction,
            "process_local_file_advanced",
            lambda path, **kwargs: mock_response,
        )

        result = extract_tables(str(test_file))

        assert len(result) == 3
        assert result[0] == {"page_index": 0, "table_id": "tbl_1", "content": "| A | B |", "format": "markdown"}
        assert result[1] == {"page_index": 0, "table_id": "tbl_2", "content": "| C | D |", "format": "markdown"}
        assert result[2] == {"page_index": 1, "table_id": "tbl_3", "content": "| E | F |", "format": "html"}

    def test_handles_page_without_tables(self, tmp_path, monkeypatch):
        """Test that pages without tables are skipped."""
        test_file = tmp_path / "test.pdf"
        test_file.write_bytes(b"%PDF-1.4\n%EOF")

        mock_response = Mock()
        mock_page1 = Mock()
        mock_page1.index = 0
        mock_page1.tables = None
        mock_page2 = Mock()
        mock_page2.index = 1
        mock_page2.tables = []
        mock_response.pages = [mock_page1, mock_page2]

        import mistral_ocr_mcp.extraction

        monkeypatch.setattr(
            mistral_ocr_mcp.extraction,
            "process_local_file_advanced",
            lambda path, **kwargs: mock_response,
        )

        result = extract_tables(str(test_file))

        assert result == []

    def test_passes_table_format(self, tmp_path, monkeypatch):
        """Test that table_format is passed to process_local_file_advanced."""
        test_file = tmp_path / "test.pdf"
        test_file.write_bytes(b"%PDF-1.4\n%EOF")

        mock_response = Mock()
        mock_response.pages = []

        calls = []

        def mock_process(path, **kwargs):
            calls.append((path, kwargs))
            return mock_response

        import mistral_ocr_mcp.extraction

        monkeypatch.setattr(
            mistral_ocr_mcp.extraction, "process_local_file_advanced", mock_process
        )

        extract_tables(str(test_file), table_format="html")

        assert len(calls) == 1
        assert calls[0][1]["table_format"] == "html"

    def test_validates_file_path(self, tmp_path):
        """Test that invalid file path raises PathValidationError."""
        with pytest.raises(PathValidationError):
            extract_tables("/nonexistent/file.pdf")


class TestOCRStatus:
    """Tests for ocr_status function."""

    def test_calls_check_api_status(self, monkeypatch):
        """Test that ocr_status calls check_api_status."""
        mock_result = {"status": "ok", "message": "API key is working"}

        import mistral_ocr_mcp.extraction

        monkeypatch.setattr(
            mistral_ocr_mcp.extraction, "check_api_status", lambda: mock_result
        )

        result = ocr_status()

        assert result == mock_result

    def test_returns_expected_keys(self, monkeypatch):
        """Test that ocr_status returns a dict with status and message."""
        mock_result = {"status": "ok", "message": "API key is working"}

        import mistral_ocr_mcp.extraction

        monkeypatch.setattr(
            mistral_ocr_mcp.extraction, "check_api_status", lambda: mock_result
        )

        result = ocr_status()

        assert isinstance(result, dict)
        assert "status" in result
        assert "message" in result
