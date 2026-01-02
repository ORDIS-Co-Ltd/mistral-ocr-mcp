# Mistral OCR MCP Server

A Model Context Protocol (MCP) server that provides tools for extracting text and images from PDF and image files using the Mistral OCR API.

## Features

- **Simple Text Extraction**: Extract markdown content from documents without handling images
- **Full Extraction with Images**: Extract markdown and save embedded images to disk with proper relative links
- **Security Sandbox**: Restricts file writes to a configured allowed directory
- **Zero-Install Deployment**: Run with `uvx` without prior installation
- **Supported Formats**: PDF (`.pdf`), PNG (`.png`), JPEG (`.jpg`, `.jpeg`), WebP (`.webp`), GIF (`.gif`)

---

## Client Configuration

### Claude Desktop

Add this to your `claude_desktop_config.json`:

- **macOS**: `~/Library/Application Support/Claude/claude_desktop_config.json`
- **Windows**: `%APPDATA%\Claude\claude_desktop_config.json`

```json
{
  "mcpServers": {
    "mistral-ocr": {
      "command": "uvx",
      "args": ["mistral-ocr-mcp"],
      "env": {
        "MISTRAL_API_KEY": "your-api-key-here",
        "MISTRAL_OCR_ALLOWED_DIR": "/absolute/path/to/allowed/directory"
      }
    }
  }
}
```

### OpenCode

Add this to the `mcp` section of your configuration file:

```json
{
  "mcp": {
    "mistral-ocr": {
      "type": "local",
      "command": ["uvx", "mistral-ocr-mcp"],
      "enabled": true,
      "environment": {
        "MISTRAL_API_KEY": "your-api-key-here",
        "MISTRAL_OCR_ALLOWED_DIR": "/absolute/path/to/allowed/directory"
      }
    }
  }
}
```

### Codex

If you use the Codex CLI, you can add the server with:

```bash
codex mcp add mistral-ocr -- uvx mistral-ocr-mcp
```

Make sure the environment variables `MISTRAL_API_KEY` and `MISTRAL_OCR_ALLOWED_DIR` are set in your shell environment.

---

## Configuration

### Required Environment Variables

| Variable | Description | Example |
|----------|-------------|---------|
| `MISTRAL_API_KEY` | Your Mistral API key (never logged) | `sk-abc123...` |
| `MISTRAL_OCR_ALLOWED_DIR` | Absolute path to allowed write directory | `/Users/username/workdir` |

### Security Sandbox

The server enforces a **write directory sandbox** to prevent unauthorized file writes:

- **`extract_markdown`**: No write restrictions (read-only operation)
- **`extract_markdown_with_images`**: The `output_dir` parameter **must** be within `MISTRAL_OCR_ALLOWED_DIR`

**Validation Examples:**

| `MISTRAL_OCR_ALLOWED_DIR` | `output_dir` | Result |
|---------------------------|--------------|--------|
| `/Users/username/workdir` | `/Users/username/workdir/project/output` | ✅ Allowed |
| `/Users/username/workdir` | `/Users/username/workdir` | ✅ Allowed (exact match) |
| `/Users/username/workdir` | `/Users/username/documents` | ❌ Rejected |
| `/Users/username/workdir` | `/Users/username/workdir/../documents` | ❌ Rejected (resolves outside) |

**Security Notes:**
- All paths are canonicalized (symlinks resolved, `..` eliminated) before validation
- Image filenames are sanitized to prevent path traversal attacks

---

## Tool Reference

### Tool 1: `extract_markdown`

Extract markdown content from a document **without** saving images.

**Arguments:**

```json
{
  "file_path": "/absolute/path/to/document.pdf"
}
```

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `file_path` | `string` | Yes | Absolute path to input file (PDF or image) |

**Returns:**

```json
"# Document Title\n\nExtracted markdown content from all pages..."
```

Returns a single string containing concatenated markdown from all pages.

**Example:**

```json
{
  "tool": "extract_markdown",
  "arguments": {
    "file_path": "/Users/username/documents/report.pdf"
  }
}
```

---

### Tool 2: `extract_markdown_with_images`

Extract markdown content **and** save embedded images to disk.

**Arguments:**

```json
{
  "file_path": "/absolute/path/to/document.pdf",
  "output_dir": "/absolute/path/to/output/parent"
}
```

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `file_path` | `string` | Yes | Absolute path to input file (PDF or image) |
| `output_dir` | `string` | Yes | Absolute path to output parent directory (must exist and be writable, must be within `MISTRAL_OCR_ALLOWED_DIR`) |

**Returns:**

```json
{
  "output_directory": "/absolute/path/to/output/parent/document",
  "markdown_file": "/absolute/path/to/output/parent/document/content.md",
  "images": ["img_abc123.png", "img_def456.jpeg"]
}
```

| Field | Type | Description |
|-------|------|-------------|
| `output_directory` | `string` | Absolute path to created subdirectory |
| `markdown_file` | `string` | Absolute path to `content.md` file |
| `images` | `array[string]` | List of saved image filenames (not full paths) |

**Behavior:**

1. Creates a subdirectory named after the input file stem (e.g., `report` for `report.pdf`)
2. If the subdirectory already exists, appends a timestamp: `report_20260102_143022`
3. Saves all extracted images as `<sanitized_id>.<ext>` (e.g., `img_abc123.png`)
4. Saves markdown to `content.md` with relative image links (e.g., `![](./img_abc123.png)`)

**Example:**

```json
{
  "tool": "extract_markdown_with_images",
  "arguments": {
    "file_path": "/Users/username/documents/quarterly-report.pdf",
    "output_dir": "/Users/username/workdir/extracted"
  }
}
```

**Output Structure:**

```
/Users/username/workdir/extracted/
  quarterly-report/
    content.md          # Markdown with relative image links
    img_abc123.png      # First extracted image
    img_def456.jpeg     # Second extracted image
```

---

## Example Client Usage

Here's a minimal Python example using the MCP SDK to call the tools:

```python
import asyncio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

async def extract_document():
    server_params = StdioServerParameters(
        command="mistral-ocr-mcp",
        env={
            "MISTRAL_API_KEY": "your-api-key",
            "MISTRAL_OCR_ALLOWED_DIR": "/Users/username/workdir"
        }
    )
    
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            
            # Simple extraction
            result = await session.call_tool(
                "extract_markdown",
                arguments={"file_path": "/path/to/document.pdf"}
            )
            print(result.content[0].text)
            
            # Extraction with images
            result = await session.call_tool(
                "extract_markdown_with_images",
                arguments={
                    "file_path": "/path/to/document.pdf",
                    "output_dir": "/Users/username/workdir/output"
                }
            )
            print(result.content[0].text)

asyncio.run(extract_document())
```

---

## Troubleshooting

| Error | Cause | Solution |
|-------|-------|----------|
| `Missing required environment variable: MISTRAL_API_KEY` | `MISTRAL_API_KEY` not set | Set the environment variable before running the server |
| `Missing required environment variable: MISTRAL_OCR_ALLOWED_DIR` | `MISTRAL_OCR_ALLOWED_DIR` not set | Set the environment variable to an absolute path |
| `MISTRAL_OCR_ALLOWED_DIR must be an absolute path` | Relative path provided (e.g., `~/documents`) | Use an absolute path (e.g., `/Users/username/documents`) |
| `MISTRAL_OCR_ALLOWED_DIR does not exist` | Directory does not exist on filesystem | Create the directory first: `mkdir -p /path/to/dir` |
| `MISTRAL_OCR_ALLOWED_DIR is not a directory` | Path points to a file, not a directory | Ensure the path is a directory |
| `validate file_path: must be an absolute path: {path}` | Relative path provided for input file | Use an absolute path (e.g., `/Users/username/file.pdf`) |
| `validate file_path: resolve failed, path does not exist: {path}` | Input file does not exist | Check the file path and ensure the file exists |
| `validate file_path: unsupported file type '{suffix}'. Supported types: ...` | File extension not supported | Use `.pdf`, `.png`, `.jpg`, `.jpeg`, `.webp`, or `.gif` |
| `validate output_dir: resolve failed, path does not exist: {path}` | Output directory does not exist | Create the directory first: `mkdir -p {path}` |
| `validate output_dir: path is not a directory: {path}` | Path points to a file, not a directory | Ensure the path is a directory |
| `validate output_dir: writability check failed, directory not writable: {path}` | Output directory exists but is not writable | Check directory permissions: `chmod u+w {path}` |
| `output_dir must be within the allowed directory` | `output_dir` is outside `MISTRAL_OCR_ALLOWED_DIR` | Use a path within the allowed directory |
| `Mistral OCR request failed (status=401): {message}` | Invalid API key | Check your `MISTRAL_API_KEY` |
| `Mistral OCR request failed (status=429): {message}` | Rate limit exceeded | Wait and retry, or check your API quota |

---

## Development

### Setup

Clone the repository and install with development dependencies:

```bash
git clone https://github.com/ORDIS-Co-Ltd/mistral-ocr-mcp
cd mistral-ocr-mcp
pip install -e '.[dev]'
```

Run the server locally:

```bash
MISTRAL_API_KEY="your-key" \
MISTRAL_OCR_ALLOWED_DIR="/path/to/allowed/dir" \
python -m mistral_ocr_mcp
```

### Run Tests

```bash
pytest
```

### Project Structure

```
mistral-ocr-mcp/
├── src/
│   └── mistral_ocr_mcp/
│       ├── __init__.py
│       ├── __main__.py          # Entry point
│       ├── server.py            # MCP server and tool definitions
│       ├── config.py            # Configuration loading and validation
│       ├── extraction.py        # OCR orchestration logic
│       ├── mistral_client.py    # Mistral API client
│       ├── images.py            # Image parsing and saving
│       ├── markdown_rewrite.py  # Markdown link rewriting
│       └── path_sandbox.py      # Path validation and sandbox enforcement
├── tests/                       # Unit tests
├── pyproject.toml              # Package configuration
└── README.md                   # This file
```

---

## License

MIT

---

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.

---

## Links

- **GitHub Repository**: https://github.com/ORDIS-Co-Ltd/mistral-ocr-mcp
- **MCP Specification**: https://modelcontextprotocol.io
- **Mistral AI**: https://mistral.ai
