"""Tests for MCP server module."""

import sys
from pathlib import Path
from unittest.mock import Mock

import pytest

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mistral_ocr_mcp.path_sandbox import PathValidationError
from mistral_ocr_mcp.server import call_tool_impl, list_tools_impl


class TestMCPToolRegistration:
    """Tests for actual MCP tool registration and schemas."""

    def test_extract_markdown_tool_has_correct_name(self):
        """Test that extract_markdown tool has the correct MCP name."""
        import asyncio

        from mistral_ocr_mcp.server import mcp

        async def get_tool_names():
            tools = await mcp.list_tools()
            return [tool.name for tool in tools]

        tool_names = asyncio.run(get_tool_names())
        assert "extract_markdown" in tool_names
        assert "extract_markdown_tool" not in tool_names

    def test_extract_markdown_tool_schema(self):
        """Test that extract_markdown tool has the correct schema with all params."""
        import asyncio

        from mistral_ocr_mcp.server import mcp

        async def find_extract_markdown_tool():
            tools = await mcp.list_tools()
            for tool in tools:
                if tool.name == "extract_markdown":
                    return tool
            return None

        tool = asyncio.run(find_extract_markdown_tool())
        assert tool is not None
        assert hasattr(tool, "input_schema")
        properties = tool.input_schema.get("properties", {})
        assert "file_path" in properties
        assert properties["file_path"]["type"] == "string"
        assert "output_dir" in properties
        assert "include_images" in properties

    def test_extract_markdown_from_url_tool_has_correct_name(self):
        """Test that extract_markdown_from_url tool has the correct MCP name."""
        import asyncio

        from mistral_ocr_mcp.server import mcp

        async def get_tool_names():
            tools = await mcp.list_tools()
            return [tool.name for tool in tools]

        tool_names = asyncio.run(get_tool_names())
        assert "extract_markdown_from_url" in tool_names

    def test_extract_markdown_from_url_tool_schema(self):
        """Test that extract_markdown_from_url tool has the correct schema."""
        import asyncio

        from mistral_ocr_mcp.server import mcp

        async def find_tool():
            tools = await mcp.list_tools()
            for tool in tools:
                if tool.name == "extract_markdown_from_url":
                    return tool
            return None

        tool = asyncio.run(find_tool())
        assert tool is not None
        assert hasattr(tool, "input_schema")
        properties = tool.input_schema.get("properties", {})
        assert "file_url" in properties
        assert properties["file_url"]["type"] == "string"
        assert "output_dir" in properties
        assert "include_images" in properties

    def test_extract_markdown_advanced_tool_has_correct_name(self):
        """Test that extract_markdown_advanced tool has the correct MCP name."""
        import asyncio

        from mistral_ocr_mcp.server import mcp

        async def get_tool_names():
            tools = await mcp.list_tools()
            return [tool.name for tool in tools]

        tool_names = asyncio.run(get_tool_names())
        assert "extract_markdown_advanced" in tool_names

    def test_extract_markdown_advanced_tool_schema(self):
        """Test that extract_markdown_advanced tool has the correct schema."""
        import asyncio

        from mistral_ocr_mcp.server import mcp

        async def find_tool():
            tools = await mcp.list_tools()
            for tool in tools:
                if tool.name == "extract_markdown_advanced":
                    return tool
            return None

        tool = asyncio.run(find_tool())
        assert tool is not None
        assert hasattr(tool, "input_schema")
        properties = tool.input_schema.get("properties", {})
        assert "file_path" in properties
        assert properties["file_path"]["type"] == "string"
        assert "pages" in properties
        assert "table_format_" in properties
        assert "model" in properties

    def test_ocr_status_tool_has_correct_name(self):
        """Test that ocr_status tool has the correct MCP name."""
        import asyncio

        from mistral_ocr_mcp.server import mcp

        async def get_tool_names():
            tools = await mcp.list_tools()
            return [tool.name for tool in tools]

        tool_names = asyncio.run(get_tool_names())
        assert "ocr_status" in tool_names

    def test_ocr_status_tool_no_required_params(self):
        """Test that ocr_status has no required parameters."""
        import asyncio

        from mistral_ocr_mcp.server import mcp

        async def find_tool():
            tools = await mcp.list_tools()
            for tool in tools:
                if tool.name == "ocr_status":
                    return tool
            return None

        tool = asyncio.run(find_tool())
        assert tool is not None
        assert (
            "required" not in tool.input_schema or tool.input_schema["required"] == []
        )


class TestListToolsImpl:
    """Tests for list_tools_impl function."""

    def test_returns_all_tool_names(self):
        """Test that all tool names are returned."""
        tools = list_tools_impl()

        assert len(tools) == 4
        assert "extract_markdown" in tools
        assert "extract_markdown_from_url" in tools
        assert "extract_markdown_advanced" in tools
        assert "ocr_status" in tools


class TestCallToolImpl:
    """Tests for call_tool_impl function."""

    def test_extract_markdown_calls_function(self, tmp_path, monkeypatch):
        """Test that extract_markdown calls the extraction function without images."""
        test_file = tmp_path / "test.pdf"
        test_file.write_bytes(b"%PDF-1.4\n%EOF")

        mock_markdown = "# Test Document\n\nContent here"
        monkeypatch.setattr(
            "mistral_ocr_mcp.server.extract_markdown",
            lambda path, **kwargs: mock_markdown,
        )

        result = call_tool_impl("extract_markdown", {"file_path": str(test_file)})

        assert result == mock_markdown

    def test_extract_markdown_with_images_calls_function(self, tmp_path, monkeypatch):
        """Test that extract_markdown with include_images=True saves images."""
        test_file = tmp_path / "test.pdf"
        test_file.write_bytes(b"%PDF-1.4\n%EOF")

        output_dir = tmp_path / "output"
        output_dir.mkdir()

        expected_result = {
            "output_directory": str(tmp_path / "output" / "test"),
            "markdown_file": str(tmp_path / "output" / "test" / "content.md"),
            "images": ["img_abc123.png"],
        }
        monkeypatch.setattr(
            "mistral_ocr_mcp.server.extract_markdown",
            lambda path, **kwargs: expected_result,
        )

        result = call_tool_impl(
            "extract_markdown",
            {
                "file_path": str(test_file),
                "include_images": True,
                "output_dir": str(output_dir),
            },
        )

        assert result == expected_result

    def test_extract_markdown_with_images_missing_output_dir(self):
        """Test that include_images=True without output_dir raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            call_tool_impl(
                "extract_markdown",
                {"file_path": "/tmp/test.pdf", "include_images": True},
            )

        assert "output_dir is required" in str(exc_info.value)

    def test_extract_markdown_missing_file_path(self):
        """Test that missing file_path raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            call_tool_impl("extract_markdown", {})

        assert "Missing required argument: file_path" in str(exc_info.value)

    def test_unknown_tool_raises_error(self):
        """Test that unknown tool name raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            call_tool_impl("unknown_tool", {})

        assert "Unknown tool: unknown_tool" in str(exc_info.value)


class TestExtractMarkdownTool:
    """Tests for extract_markdown tool behavior via call_tool_impl."""

    def test_extract_markdown_returns_markdown_string(self, tmp_path, monkeypatch):
        """Test that extract_markdown returns a string."""
        test_file = tmp_path / "document.pdf"
        test_file.write_bytes(b"%PDF-1.4\n%EOF")

        expected_markdown = "# Extracted Content\n\nSome text content"
        monkeypatch.setattr(
            "mistral_ocr_mcp.server.extract_markdown",
            lambda path, **kwargs: expected_markdown,
        )

        result = call_tool_impl("extract_markdown", {"file_path": str(test_file)})

        assert isinstance(result, str)
        assert result == expected_markdown

    def test_extract_markdown_passes_file_path_correctly(self, tmp_path, monkeypatch):
        """Test that file_path is passed correctly to extraction function."""
        test_file = tmp_path / "mydocument.pdf"
        test_file.write_bytes(b"%PDF-1.4\n%EOF")

        captured_path = None

        def capture_path(path):
            nonlocal captured_path
            captured_path = path
            return "# Result"

        monkeypatch.setattr(
            "mistral_ocr_mcp.server.extract_markdown",
            lambda path, **kwargs: capture_path(path),
        )

        call_tool_impl("extract_markdown", {"file_path": str(test_file)})

        assert captured_path == str(test_file)


class TestExtractMarkdownToolWithImages:
    """Tests for extract_markdown tool with include_images via call_tool_impl."""

    def test_extract_markdown_with_images_returns_dict(self, tmp_path, monkeypatch):
        """Test that extract_markdown with include_images=True returns a dict."""
        test_file = tmp_path / "document.pdf"
        test_file.write_bytes(b"%PDF-1.4\n%EOF")

        output_dir = tmp_path / "output"
        output_dir.mkdir()

        expected_result = {
            "output_directory": str(output_dir / "document"),
            "markdown_file": str(output_dir / "document" / "content.md"),
            "images": [],
        }
        monkeypatch.setattr(
            "mistral_ocr_mcp.server.extract_markdown",
            lambda path, **kwargs: expected_result,
        )

        result = call_tool_impl(
            "extract_markdown",
            {
                "file_path": str(test_file),
                "include_images": True,
                "output_dir": str(output_dir),
            },
        )

        assert isinstance(result, dict)
        assert "output_directory" in result
        assert "markdown_file" in result
        assert "images" in result

    def test_extract_markdown_with_images_passes_arguments_correctly(
        self, tmp_path, monkeypatch
    ):
        """Test that file_path and output_dir are passed correctly with images."""
        test_file = tmp_path / "document.pdf"
        test_file.write_bytes(b"%PDF-1.4\n%EOF")

        output_dir = tmp_path / "output"
        output_dir.mkdir()

        captured_kwargs = None

        def capture_kwargs(path, **kwargs):
            nonlocal captured_kwargs
            captured_kwargs = (path, kwargs)
            return {
                "output_directory": str(output_dir / "document"),
                "markdown_file": str(output_dir / "document" / "content.md"),
                "images": [],
            }

        monkeypatch.setattr("mistral_ocr_mcp.server.extract_markdown", capture_kwargs)

        call_tool_impl(
            "extract_markdown",
            {
                "file_path": str(test_file),
                "include_images": True,
                "output_dir": str(output_dir),
            },
        )

        assert captured_kwargs is not None
        assert captured_kwargs[0] == str(test_file)
        assert captured_kwargs[1]["output_dir"] == str(output_dir)
        assert captured_kwargs[1]["include_images"] is True

    def test_extract_markdown_from_url_calls_function(self, monkeypatch):
        """Test that extract_markdown_from_url tool calls the URL extraction function."""
        mock_markdown = {"result": "# URL Content"}
        monkeypatch.setattr(
            "mistral_ocr_mcp.server.extract_from_url",
            lambda url, **kwargs: mock_markdown,
        )

        result = call_tool_impl(
            "extract_markdown_from_url",
            {"file_url": "https://example.com/doc.pdf"},
        )

        assert result == mock_markdown

    def test_extract_markdown_from_url_missing_file_url(self):
        """Test that missing file_url raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            call_tool_impl("extract_markdown_from_url", {})

        assert "Missing required argument: file_url" in str(exc_info.value)

    def test_extract_markdown_from_url_with_images_calls_function(
        self, tmp_path, monkeypatch
    ):
        """Test that extract_markdown_from_url with include_images saves images."""
        mock_config = Mock()
        mock_config.allowed_dirs_resolved = [tmp_path]
        mock_config.allowed_dirs_original = str(tmp_path)
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        monkeypatch.setattr(
            "mistral_ocr_mcp.extraction.load_config", lambda: mock_config
        )

        mock_response = Mock()
        mock_page = Mock()
        mock_page.markdown = "# Page 1\n\nContent"
        mock_page.images = []
        mock_response.pages = [mock_page]
        monkeypatch.setattr(
            "mistral_ocr_mcp.extraction.process_url",
            lambda url, **kwargs: mock_response,
        )
        monkeypatch.setattr(
            "mistral_ocr_mcp.extraction.save_images", lambda out_dir, images: []
        )

        result = call_tool_impl(
            "extract_markdown_from_url",
            {
                "file_url": "https://example.com/doc.pdf",
                "include_images": True,
                "output_dir": str(output_dir),
            },
        )

        assert "output_directory" in result
        assert "markdown_file" in result
        assert "images" in result

    def test_extract_markdown_from_url_with_images_missing_output_dir(self):
        """Test that include_images=True without output_dir raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            call_tool_impl(
                "extract_markdown_from_url",
                {"file_url": "https://example.com/doc.pdf", "include_images": True},
            )

        assert "output_dir is required" in str(exc_info.value)

    def test_extract_markdown_advanced_calls_function(self, tmp_path, monkeypatch):
        """Test that extract_markdown_advanced calls the extraction function."""
        test_file = tmp_path / "test.pdf"
        test_file.write_bytes(b"%PDF-1.4\n%EOF")

        mock_markdown = "# Advanced Result"
        monkeypatch.setattr(
            "mistral_ocr_mcp.server.extract_markdown_advanced",
            lambda path, **kwargs: mock_markdown,
        )

        result = call_tool_impl(
            "extract_markdown_advanced", {"file_path": str(test_file)}
        )

        assert result == mock_markdown

    def test_extract_markdown_advanced_missing_file_path(self):
        """Test that missing file_path raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            call_tool_impl("extract_markdown_advanced", {})

        assert "Missing required argument: file_path" in str(exc_info.value)

    def test_extract_markdown_advanced_passes_optional_params(
        self, tmp_path, monkeypatch
    ):
        """Test that optional params are passed to extraction function."""
        test_file = tmp_path / "test.pdf"
        test_file.write_bytes(b"%PDF-1.4\n%EOF")

        captured = {}

        def capture(path, **kwargs):
            captured["path"] = path
            captured["pages"] = kwargs.get("pages")
            captured["table_format_"] = kwargs.get("table_format_")
            captured["model"] = kwargs.get("model", "mistral-ocr-latest")
            return "# Result"

        monkeypatch.setattr("mistral_ocr_mcp.server.extract_markdown_advanced", capture)

        call_tool_impl(
            "extract_markdown_advanced",
            {
                "file_path": str(test_file),
                "pages": [1, 3],
                "table_format_": "html",
                "model": "mistral-ocr-latest",
            },
        )

        assert captured["path"] == str(test_file)
        assert captured["pages"] == [1, 3]
        assert captured["table_format_"] == "html"
        assert captured["model"] == "mistral-ocr-latest"

    def test_ocr_status_calls_function(self, monkeypatch):
        """Test that ocr_status calls the status function."""
        mock_status = {"status": "ok", "message": "API key is working"}
        monkeypatch.setattr("mistral_ocr_mcp.server.ocr_status", lambda: mock_status)

        result = call_tool_impl("ocr_status", {})

        assert result == mock_status

    def test_ocr_status_returns_dict(self, monkeypatch):
        """Test that ocr_status returns a dict with expected keys."""
        mock_status = {"status": "ok", "message": "API key is working"}
        monkeypatch.setattr("mistral_ocr_mcp.server.ocr_status", lambda: mock_status)

        result = call_tool_impl("ocr_status", {})

        assert isinstance(result, dict)
        assert "status" in result
        assert "message" in result

    def test_extract_markdown_with_images_handles_extraction_errors(
        self, tmp_path, monkeypatch
    ):
        """Test that extraction errors are properly raised."""
        test_file = tmp_path / "document.pdf"
        test_file.write_bytes(b"%PDF-1.4\n%EOF")

        output_dir = tmp_path / "output"
        output_dir.mkdir()

        # Mock extraction function to raise PathValidationError
        def raise_error(path, **kwargs):
            raise PathValidationError("File not found")

        monkeypatch.setattr("mistral_ocr_mcp.server.extract_markdown", raise_error)

        with pytest.raises(PathValidationError) as exc_info:
            call_tool_impl(
                "extract_markdown",
                {
                    "file_path": str(test_file),
                    "include_images": True,
                    "output_dir": str(output_dir),
                },
            )

        assert "File not found" in str(exc_info.value)


class TestServerIntegration:
    """Integration tests for server functionality."""

    def test_tools_are_properly_defined(self):
        """Test that all tools are properly defined in the module."""
        tools = list_tools_impl()
        assert len(tools) == 4
        assert all(isinstance(tool, str) for tool in tools)

    def test_call_tool_impl_handles_empty_arguments(self):
        """Test that call_tool_impl handles empty arguments dict."""
        with pytest.raises(ValueError):
            call_tool_impl("extract_markdown", {})

    def test_call_tool_impl_passes_only_required_arguments(self, tmp_path, monkeypatch):
        """Test that only required arguments are passed to extraction function."""
        test_file = tmp_path / "test.pdf"
        test_file.write_bytes(b"%PDF-1.4\n%EOF")

        # Mock function to capture all arguments
        captured_args = None

        def mock_extract(path, **kwargs):
            nonlocal captured_args
            captured_args = kwargs
            return "# Result"

        monkeypatch.setattr("mistral_ocr_mcp.server.extract_markdown", mock_extract)

        # Call with extra argument that should be ignored
        call_tool_impl(
            "extract_markdown", {"file_path": str(test_file), "extra": "value"}
        )

        # The function receives output_dir=None and include_images=False
        assert captured_args == {"output_dir": None, "include_images": False}

    def test_mcp_call_tool_returns_inline_result(self, tmp_path, monkeypatch):
        """Test that MCP call_tool returns correct result (no output_dir)."""
        import asyncio

        from mistral_ocr_mcp.server import mcp

        mock_response = Mock()
        mock_page = Mock()
        mock_page.markdown = "# Page 1\n\nContent"
        mock_response.pages = [mock_page]
        monkeypatch.setattr(
            "mistral_ocr_mcp.extraction.process_local_file",
            lambda path, **kwargs: mock_response,
        )
        monkeypatch.setattr(
            "mistral_ocr_mcp.extraction.validate_file_path",
            lambda path: Path(path).resolve(),
        )

        async def call():
            return await mcp.call_tool(
                "extract_markdown",
                {"file_path": str(tmp_path / "test.pdf")},
            )

        result = asyncio.run(call())
        assert not result.is_error, (
            f"MCP call failed: {result.content[0].text if result.content else 'no content'}"
        )
        assert result.structured_content is not None

    def test_mcp_call_tool_saves_markdown_to_disk(self, tmp_path, monkeypatch):
        """Test that MCP call_tool saves content.md when output_dir is set."""
        import asyncio

        from mistral_ocr_mcp.server import mcp

        test_file = tmp_path / "test.pdf"
        test_file.write_bytes(b"%PDF-1.4\n%EOF")

        mock_response = Mock()
        mock_page = Mock()
        mock_page.markdown = "# Page 1\n\nContent"
        mock_response.pages = [mock_page]
        monkeypatch.setattr(
            "mistral_ocr_mcp.extraction.process_local_file",
            lambda path, **kwargs: mock_response,
        )

        mock_config = Mock()
        mock_config.allowed_dirs_resolved = [tmp_path]
        mock_config.allowed_dirs_original = str(tmp_path)
        monkeypatch.setattr(
            "mistral_ocr_mcp.extraction.load_config", lambda: mock_config
        )

        output_dir = tmp_path / "output"
        output_dir.mkdir()

        async def call():
            return await mcp.call_tool(
                "extract_markdown",
                {
                    "file_path": str(test_file),
                    "output_dir": str(output_dir),
                },
            )

        result = asyncio.run(call())
        assert not result.is_error, (
            f"MCP call failed: {result.content[0].text if result.content else 'no content'}"
        )
        content_files = list(output_dir.rglob("content.md"))
        assert len(content_files) > 0, (
            "content.md should be saved when output_dir is set"
        )

    def test_mcp_call_tool_with_images_saves_content_and_images(
        self, tmp_path, monkeypatch
    ):
        """Test that MCP call_tool saves content.md and images when include_images=True."""
        import asyncio

        from mistral_ocr_mcp.server import mcp

        test_file = tmp_path / "doc.pdf"
        test_file.write_bytes(b"%PDF-1.4\n%EOF")

        mock_response = Mock()
        mock_page = Mock()
        mock_page.markdown = "# Page 1\n\nContent"
        mock_page.images = []
        mock_response.pages = [mock_page]
        monkeypatch.setattr(
            "mistral_ocr_mcp.extraction.process_local_file",
            lambda path, **kwargs: mock_response,
        )

        mock_config = Mock()
        mock_config.allowed_dirs_resolved = [tmp_path]
        mock_config.allowed_dirs_original = str(tmp_path)
        monkeypatch.setattr(
            "mistral_ocr_mcp.extraction.load_config", lambda: mock_config
        )
        monkeypatch.setattr(
            "mistral_ocr_mcp.extraction.save_images", lambda out_dir, images: []
        )

        output_dir = tmp_path / "output"
        output_dir.mkdir()

        async def call():
            return await mcp.call_tool(
                "extract_markdown",
                {
                    "file_path": str(test_file),
                    "output_dir": str(output_dir),
                    "include_images": True,
                },
            )

        result = asyncio.run(call())
        assert not result.is_error, (
            f"MCP call failed: {result.content[0].text if result.content else 'no content'}"
        )
        content_files = list(output_dir.rglob("content.md"))
        assert len(content_files) > 0, "content.md should be saved"
        assert result.structured_content is not None
