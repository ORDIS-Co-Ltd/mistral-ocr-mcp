"""MCP server implementation for Mistral OCR."""

from typing import Any, Optional

from mcp.server.mcpserver import MCPServer

from .extraction import (
    ExtractMarkdownWithImagesResult,
    extract_from_url,
    extract_from_url_with_images,
    extract_markdown,
    extract_markdown_advanced,
    extract_markdown_with_images,
    ocr_status,
)


# Create the MCP server instance
mcp = MCPServer("Mistral OCR")


@mcp.tool(name="extract_markdown")
def extract_markdown_tool(
    file_path: str,
    output_dir: Optional[str] = None,
    include_images: bool = False,
) -> ExtractMarkdownWithImagesResult:
    """Extract markdown text from a PDF or image file.

    When ``include_images`` is True and ``output_dir`` is provided, saves
    extracted images to disk and returns metadata. Otherwise returns the
    markdown text under the ``result`` key.

    Args:
        file_path: Absolute path to the input file (PDF or image)
        output_dir: Absolute path to an existing output directory (must be
            within allowed dir). Required when include_images is True.
        include_images: When True and output_dir is set, save images to disk

    Returns:
        When include_images is False:
            result: Extracted markdown content
        When include_images is True:
            output_directory: Absolute path to the output subdirectory
            markdown_file: Absolute path to the content.md file
            images: List of saved image filenames
    """
    if include_images and output_dir:
        return extract_markdown_with_images(file_path, output_dir)
    return extract_markdown(file_path)


@mcp.tool(name="extract_markdown_from_url")
def extract_markdown_from_url_tool(
    file_url: str,
    include_image_base64: bool = False,
) -> str:
    """Extract markdown text from a publicly accessible URL.

    Processes a PDF or image directly from a URL without uploading
    a local file first.

    Args:
        file_url: Publicly accessible URL to a PDF or image
        include_image_base64: Whether to include base64 images in the response

    Returns:
        Extracted markdown content as a string
    """
    return extract_from_url(file_url, include_image_base64=include_image_base64)


@mcp.tool(name="extract_markdown_from_url_with_images")
def extract_markdown_from_url_with_images_tool(
    file_url: str,
    output_dir: str,
) -> ExtractMarkdownWithImagesResult:
    """Extract markdown from a URL and save embedded images to disk.

    Processes a PDF or image from a URL, saves extracted images to
    the output directory, and rewrites the markdown with relative
    image paths.

    Args:
        file_url: Publicly accessible URL to a PDF or image
        output_dir: Absolute path to an existing output directory (must be within
            the allowed directory from config)

    Returns:
        Dictionary with:
            - output_directory: Absolute path to the output subdirectory
            - markdown_file: Absolute path to the content.md file
            - images: List of saved image filenames (not full paths)
    """
    return extract_from_url_with_images(file_url, output_dir)


@mcp.tool(name="extract_markdown_advanced")
def extract_markdown_advanced_tool(
    file_path: str,
    pages: Optional[list[int]] = None,
    table_format_: Optional[str] = None,
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
        "extract_markdown_from_url_with_images",
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
        if arguments.get("include_images") and "output_dir" not in arguments:
            raise ValueError("output_dir is required when include_images is True")
        if arguments.get("include_images") and arguments.get("output_dir"):
            return extract_markdown_with_images(
                arguments["file_path"], arguments["output_dir"]
            )
        return extract_markdown(arguments["file_path"])
    elif name == "extract_markdown_from_url":
        if "file_url" not in arguments:
            raise ValueError("Missing required argument: file_url")
        return extract_from_url(
            arguments["file_url"],
            include_image_base64=arguments.get("include_image_base64", False),
        )
    elif name == "extract_markdown_from_url_with_images":
        if "file_url" not in arguments:
            raise ValueError("Missing required argument: file_url")
        if "output_dir" not in arguments:
            raise ValueError("Missing required argument: output_dir")
        return extract_from_url_with_images(
            arguments["file_url"], arguments["output_dir"]
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
    try:
        mcp.run()
    except KeyboardInterrupt:
        pass
