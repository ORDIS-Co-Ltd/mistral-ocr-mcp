# Mistral OCR MCP Server

A Model Context Protocol (MCP) server that provides tools for extracting text and images from PDF and image files using the Mistral OCR API.

## Installation

```bash
pip install mistral-ocr-mcp
```

Or use with uvx (recommended for zero-install deployment):

```bash
uvx mistral-ocr-mcp
```

## Usage

The server will provide two tools (implementation in progress):
- `extract_markdown`: Extract markdown content from a document
- `extract_markdown_with_images`: Extract markdown and save embedded images to disk

## Configuration

Set the following environment variables:
- `MISTRAL_API_KEY`: Your Mistral API key
- `MISTRAL_OCR_ALLOWED_DIR`: The allowed directory for file write operations

## Development

Install with development dependencies:

```bash
pip install -e '.[dev]'
```

Run tests:

```bash
pytest
```
