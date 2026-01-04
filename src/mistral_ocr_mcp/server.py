"""MCP server implementation for Mistral OCR."""

from typing import Any

from mcp.server.fastmcp import FastMCP

from .extraction import extract_markdown, extract_markdown_with_images


# Create the MCP server instance
mcp = FastMCP("Mistral OCR")


@mcp.tool(name="extract_markdown")
def extract_markdown_tool(file_path: str) -> str:
    """Extract markdown text from a PDF or image file.

    Args:
        file_path: Absolute path to the input file (PDF or image)

    Returns:
        Extracted markdown content as a string
    """
    return extract_markdown(file_path)


@mcp.tool(name="extract_markdown_with_images")
def extract_markdown_with_images_tool(
    file_path: str, output_dir: str
) -> dict[str, Any]:
    """Extract markdown with embedded images and save them as separate files.

    Args:
        file_path: Absolute path to the input file (PDF or image)
        output_dir: Absolute path to an existing output directory (must be within allowed dir)

    Returns:
        Dictionary with:
            - output_directory: Absolute path to the output subdirectory
            - markdown_file: Absolute path to the content.md file
            - images: List of saved image filenames (not full paths)
    """
    result = extract_markdown_with_images(file_path, output_dir)
    return result


def list_tools_impl() -> list[str]:
    """List available tool names for testing purposes."""
    return ["extract_markdown", "extract_markdown_with_images"]


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
        return extract_markdown(arguments["file_path"])
    elif name == "extract_markdown_with_images":
        if "file_path" not in arguments:
            raise ValueError("Missing required argument: file_path")
        if "output_dir" not in arguments:
            raise ValueError("Missing required argument: output_dir")
        return extract_markdown_with_images(
            arguments["file_path"], arguments["output_dir"]
        )
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
