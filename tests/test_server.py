"""Tests for MCP server module."""

import sys
from pathlib import Path
import pytest
from unittest.mock import Mock, MagicMock, patch

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mistral_ocr_mcp.server import list_tools_impl, call_tool_impl
from mistral_ocr_mcp.extraction import extract_markdown, extract_markdown_with_images
from mistral_ocr_mcp.path_sandbox import PathValidationError
import mcp.types


class TestMCPToolRegistration:
    """Tests for actual MCP tool registration and schemas."""

    def test_extract_markdown_tool_has_correct_name(self):
        """Test that extract_markdown tool has the correct MCP name."""
        from mistral_ocr_mcp.server import mcp
        import asyncio

        async def get_tool_names():
            tools = await mcp.list_tools()
            return [tool.name for tool in tools]

        tool_names = asyncio.run(get_tool_names())
        assert "extract_markdown" in tool_names
        assert "extract_markdown_tool" not in tool_names

    def test_extract_markdown_with_images_tool_has_correct_name(self):
        """Test that extract_markdown_with_images tool has the correct MCP name."""
        from mistral_ocr_mcp.server import mcp
        import asyncio

        async def get_tool_names():
            tools = await mcp.list_tools()
            return [tool.name for tool in tools]

        tool_names = asyncio.run(get_tool_names())
        assert "extract_markdown_with_images" in tool_names
        assert "extract_markdown_with_images_tool" not in tool_names

    def test_extract_markdown_tool_schema(self):
        """Test that extract_markdown tool has the correct schema."""
        from mistral_ocr_mcp.server import mcp
        import asyncio

        async def find_extract_markdown_tool():
            tools = await mcp.list_tools()
            for tool in tools:
                if tool.name == "extract_markdown":
                    return tool
            return None

        tool = asyncio.run(find_extract_markdown_tool())
        assert tool is not None
        assert hasattr(tool, "inputSchema")
        assert "file_path" in tool.inputSchema.get("properties", {})
        assert tool.inputSchema["properties"]["file_path"]["type"] == "string"

    def test_extract_markdown_with_images_tool_schema(self):
        """Test that extract_markdown_with_images tool has the correct schema."""
        from mistral_ocr_mcp.server import mcp
        import asyncio

        async def find_extract_markdown_with_images_tool():
            tools = await mcp.list_tools()
            for tool in tools:
                if tool.name == "extract_markdown_with_images":
                    return tool
            return None

        tool = asyncio.run(find_extract_markdown_with_images_tool())
        assert tool is not None
        assert hasattr(tool, "inputSchema")
        properties = tool.inputSchema.get("properties", {})
        assert "file_path" in properties
        assert "output_dir" in properties
        assert properties["file_path"]["type"] == "string"
        assert properties["output_dir"]["type"] == "string"

    def test_extract_markdown_with_images_tool_description_includes_existence_hint(
        self,
    ):
        """Test that extract_markdown_with_images tool description includes existence hint."""
        from mistral_ocr_mcp.server import mcp
        import asyncio

        async def find_extract_markdown_with_images_tool():
            tools = await mcp.list_tools()
            for tool in tools:
                if tool.name == "extract_markdown_with_images":
                    return tool
            return None

        tool = asyncio.run(find_extract_markdown_with_images_tool())
        assert tool is not None
        assert tool.description is not None
        assert "exist" in tool.description.lower()


class TestListToolsImpl:
    """Tests for list_tools_impl function."""

    def test_returns_both_tool_names(self):
        """Test that both tool names are returned."""
        tools = list_tools_impl()

        assert len(tools) == 2
        assert "extract_markdown" in tools
        assert "extract_markdown_with_images" in tools


class TestCallToolImpl:
    """Tests for call_tool_impl function."""

    def test_extract_markdown_calls_function(self, tmp_path, monkeypatch):
        """Test that extract_markdown tool calls the extraction function."""
        # Create a test PDF file
        test_file = tmp_path / "test.pdf"
        test_file.write_bytes(b"%PDF-1.4\n%EOF")

        # Mock extraction function
        mock_markdown = "# Test Document\n\nContent here"
        monkeypatch.setattr(
            "mistral_ocr_mcp.server.extract_markdown", lambda path: mock_markdown
        )

        result = call_tool_impl("extract_markdown", {"file_path": str(test_file)})

        assert result == mock_markdown

    def test_extract_markdown_missing_file_path(self):
        """Test that missing file_path raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            call_tool_impl("extract_markdown", {})

        assert "Missing required argument: file_path" in str(exc_info.value)

    def test_extract_markdown_with_images_calls_function(self, tmp_path, monkeypatch):
        """Test that extract_markdown_with_images tool calls the extraction function."""
        # Create test files
        test_file = tmp_path / "test.pdf"
        test_file.write_bytes(b"%PDF-1.4\n%EOF")

        output_dir = tmp_path / "output"
        output_dir.mkdir()

        # Mock extraction function
        expected_result = {
            "output_directory": str(tmp_path / "output" / "test"),
            "markdown_file": str(tmp_path / "output" / "test" / "content.md"),
            "images": ["img_abc123.png"],
        }
        monkeypatch.setattr(
            "mistral_ocr_mcp.server.extract_markdown_with_images",
            lambda path, output: expected_result,
        )

        result = call_tool_impl(
            "extract_markdown_with_images",
            {"file_path": str(test_file), "output_dir": str(output_dir)},
        )

        assert result == expected_result

    def test_extract_markdown_with_images_missing_file_path(self):
        """Test that missing file_path raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            call_tool_impl("extract_markdown_with_images", {"output_dir": "/tmp"})

        assert "Missing required argument: file_path" in str(exc_info.value)

    def test_extract_markdown_with_images_missing_output_dir(self, tmp_path):
        """Test that missing output_dir raises ValueError."""
        test_file = tmp_path / "test.pdf"
        test_file.write_bytes(b"%PDF-1.4\n%EOF")

        with pytest.raises(ValueError) as exc_info:
            call_tool_impl(
                "extract_markdown_with_images", {"file_path": str(test_file)}
            )

        assert "Missing required argument: output_dir" in str(exc_info.value)

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
            lambda path: expected_markdown,
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

        monkeypatch.setattr("mistral_ocr_mcp.server.extract_markdown", capture_path)

        call_tool_impl("extract_markdown", {"file_path": str(test_file)})

        assert captured_path == str(test_file)


class TestExtractMarkdownWithImagesTool:
    """Tests for extract_markdown_with_images tool behavior via call_tool_impl."""

    def test_extract_markdown_with_images_returns_dict(self, tmp_path, monkeypatch):
        """Test that extract_markdown_with_images returns a dictionary."""
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
            "mistral_ocr_mcp.server.extract_markdown_with_images",
            lambda path, output: expected_result,
        )

        result = call_tool_impl(
            "extract_markdown_with_images",
            {"file_path": str(test_file), "output_dir": str(output_dir)},
        )

        assert isinstance(result, dict)
        assert "output_directory" in result
        assert "markdown_file" in result
        assert "images" in result

    def test_extract_markdown_with_images_passes_arguments_correctly(
        self, tmp_path, monkeypatch
    ):
        """Test that both file_path and output_dir are passed correctly."""
        test_file = tmp_path / "document.pdf"
        test_file.write_bytes(b"%PDF-1.4\n%EOF")

        output_dir = tmp_path / "output"
        output_dir.mkdir()

        captured_args = None

        def capture_args(path, output):
            nonlocal captured_args
            captured_args = (path, output)
            return {
                "output_directory": str(output_dir / "document"),
                "markdown_file": str(output_dir / "document" / "content.md"),
                "images": [],
            }

        monkeypatch.setattr(
            "mistral_ocr_mcp.server.extract_markdown_with_images", capture_args
        )

        call_tool_impl(
            "extract_markdown_with_images",
            {"file_path": str(test_file), "output_dir": str(output_dir)},
        )

        # Verify capture function was called
        assert captured_args is not None
        assert captured_args[0] == str(test_file)
        assert captured_args[1] == str(output_dir)

    def test_extract_markdown_with_images_handles_extraction_errors(
        self, tmp_path, monkeypatch
    ):
        """Test that extraction errors are properly raised."""
        test_file = tmp_path / "document.pdf"
        test_file.write_bytes(b"%PDF-1.4\n%EOF")

        output_dir = tmp_path / "output"
        output_dir.mkdir()

        # Mock extraction function to raise PathValidationError
        def raise_error(path, output):
            raise PathValidationError("File not found")

        monkeypatch.setattr(
            "mistral_ocr_mcp.server.extract_markdown_with_images", raise_error
        )

        # The error should propagate through call_tool_impl
        with pytest.raises(PathValidationError) as exc_info:
            call_tool_impl(
                "extract_markdown_with_images",
                {"file_path": str(test_file), "output_dir": str(output_dir)},
            )

        assert "File not found" in str(exc_info.value)


class TestServerIntegration:
    """Integration tests for server functionality."""

    def test_tools_are_properly_defined(self):
        """Test that both tools are properly defined in the module."""
        tools = list_tools_impl()
        assert len(tools) == 2
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

        # The extra argument should be ignored (kwargs should be empty)
        assert captured_args == {}
