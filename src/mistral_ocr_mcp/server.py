"""MCP server implementation for Mistral OCR."""

import contextlib
from typing import Any

from mcp.server.mcpserver import MCPServer

from .extraction import (
    ExtractMarkdownWithImagesResult,
    extract_from_url,
    extract_markdown,
    extract_markdown_advanced,
    ocr_status,
)

# Create the MCP server instance
mcp = MCPServer("Mistral OCR")


@mcp.tool(name="extract_markdown")
def extract_markdown_tool(
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
    """
    return extract_markdown(
        file_path, output_dir=output_dir, include_images=include_images
    )


@mcp.tool(name="extract_markdown_from_url")
def extract_markdown_from_url_tool(
    file_url: str,
    output_dir: str | None = None,
    include_images: bool = False,
) -> ExtractMarkdownWithImagesResult:
    """Extract markdown text from a publicly accessible URL.

    Processes a PDF or image directly from a URL without uploading
    a local file first. When ``output_dir`` is provided, saves the
    extracted markdown to ``content.md`` inside a named subdirectory.
    When ``include_images`` is also ``True``, saves embedded images
    alongside the markdown file.

    Args:
        file_url: Publicly accessible URL to a PDF or image
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
    """
    return extract_from_url(
        file_url, output_dir=output_dir, include_images=include_images
    )


@mcp.tool(name="extract_markdown_advanced")
def extract_markdown_advanced_tool(
    file_path: str,
    pages: list[int] | None = None,
    table_format_: str | None = None,
    model: str = "mistral-ocr-latest",
) -> str:
    """Extract markdown with advanced OCR options.

    Args:
        file_path: Absolute path to the input file (PDF or image)
        pages: Specific page numbers to process (1-indexed, e.g. [1, 3, 5])
        table_format_: Output format for tables ("markdown" or "html")
        model: OCR model to use (default: "mistral-ocr-latest")

    Returns:
        Extracted markdown content as a string
    """
    return extract_markdown_advanced(
        file_path,
        pages=pages,
        table_format_=table_format_,
        model=model,
    )


@mcp.tool(name="ocr_status")
def ocr_status_tool() -> dict[str, str]:
    """Check Mistral API connectivity and key validity.

    Makes a lightweight API call to verify the configured
    API key is working correctly.

    Returns:
        Dictionary with status ("ok" or "error") and message
    """
    return ocr_status()


def list_tools_impl() -> list[str]:
    """List available tool names for testing purposes."""
    return [
        "extract_markdown",
        "extract_markdown_from_url",
        "extract_markdown_advanced",
        "ocr_status",
    ]


def call_tool_impl(name: str, arguments: dict[str, Any]) -> Any:
    """Call a tool implementation for testing purposes.

    Args:
        name: Tool name to call
        arguments: Tool arguments as a dictionary

    Returns:
        Tool result or raises an error

    Raises:
        ValueError: If tool name is unknown
    """
    if name == "extract_markdown":
        if "file_path" not in arguments:
            raise ValueError("Missing required argument: file_path")
        if arguments.get("include_images") and not arguments.get("output_dir"):
            raise ValueError("output_dir is required when include_images is True")
        return extract_markdown(
            arguments["file_path"],
            output_dir=arguments.get("output_dir"),
            include_images=arguments.get("include_images", False),
        )
    elif name == "extract_markdown_from_url":
        if "file_url" not in arguments:
            raise ValueError("Missing required argument: file_url")
        if arguments.get("include_images") and not arguments.get("output_dir"):
            raise ValueError("output_dir is required when include_images is True")
        return extract_from_url(
            arguments["file_url"],
            output_dir=arguments.get("output_dir"),
            include_images=arguments.get("include_images", False),
        )
    elif name == "extract_markdown_advanced":
        if "file_path" not in arguments:
            raise ValueError("Missing required argument: file_path")
        return extract_markdown_advanced(
            arguments["file_path"],
            pages=arguments.get("pages"),
            table_format_=arguments.get("table_format_"),
            model=arguments.get("model", "mistral-ocr-latest"),
        )
    elif name == "ocr_status":
        return ocr_status()
    else:
        raise ValueError(f"Unknown tool: {name}")


def run() -> None:
    """Run the MCP server.

    This is a synchronous wrapper that starts the stdio server.
    """
    with contextlib.suppress(KeyboardInterrupt):
        mcp.run()
