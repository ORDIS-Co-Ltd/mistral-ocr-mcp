# Software Requirements Specification: Mistral OCR MCP Server

## Background

Large Language Models (LLMs) operating within the Model Context Protocol (MCP) ecosystem need the ability to extract text and images from PDF and image files. The Mistral OCR API provides high-quality document understanding, but LLMs cannot directly consume large base64-encoded images in responses due to context window limitations and inefficiency.

This MCP server bridges that gap by:
1. Providing a simple tool to extract markdown text from documents
2. Providing an advanced tool that extracts both text and images, saving them to disk with properly linked markdown

This enables LLMs to process document content without being overwhelmed by raw image data.

---

## User Stories

### US-1: Simple Text Extraction
> As an LLM agent, I want to extract markdown content from a PDF or image file so that I can analyze and respond to questions about the document's text content.

### US-2: Full Extraction with Images
> As an LLM agent, I want to extract markdown content AND save embedded images to a specified directory so that I can reference both text and images without consuming base64 data in my context window.

### US-3: Easy Deployment
> As a developer, I want to run this MCP server with a single `uvx` command so that I can quickly integrate it into my MCP client without complex setup.

---

## Functional Requirements

### FR-1: MCP Server Implementation

| ID | Requirement |
|----|-------------|
| FR-1.1 | The server MUST implement the Model Context Protocol (MCP) specification |
| FR-1.2 | The server MUST expose exactly two tools (see FR-2 and FR-3) |
| FR-1.3 | The server MUST be implemented in Python |
| FR-1.4 | The server MUST be runnable via `uvx` for zero-install deployment |

### FR-2: Tool 1 - `extract_markdown`

**Purpose:** Extract markdown content from a document without saving images.

| ID | Requirement |
|----|-------------|
| FR-2.1 | The tool MUST accept a single parameter: `file_path` (string, required) - absolute path to the input file |
| FR-2.2 | The tool MUST support PDF files (`.pdf`) |
| FR-2.3 | The tool MUST support image files: `.png`, `.jpg`, `.jpeg`, `.webp`, `.gif` |
| FR-2.4 | The tool MUST call the Mistral OCR API with `include_image_base64=false` |
| FR-2.5 | The tool MUST return the extracted markdown text as a string |
| FR-2.6 | The tool MUST return an error if the file does not exist |
| FR-2.7 | The tool MUST return an error if the file type is not supported |

**Example Invocation:**
```json
{
  "tool": "extract_markdown",
  "arguments": {
    "file_path": "/Users/ben/documents/report.pdf"
  }
}
```

**Example Response:**
```json
{
  "content": "# Annual Report 2025\n\nThis document outlines..."
}
```

### FR-3: Tool 2 - `extract_markdown_with_images`

**Purpose:** Extract markdown content and save embedded images to disk.

| ID | Requirement |
|----|-------------|
| FR-3.1 | The tool MUST accept two parameters: |
|        | - `file_path` (string, required) - absolute path to the input file |
|        | - `output_dir` (string, required) - absolute path to the parent directory where output will be saved |
| FR-3.2 | The tool MUST support the same file types as `extract_markdown` (FR-2.2, FR-2.3) |
| FR-3.3 | The tool MUST call the Mistral OCR API with `include_image_base64=true` |
| FR-3.4 | The tool MUST create a subdirectory named after the input file (without extension) inside `output_dir` |
| FR-3.5 | If the subdirectory already exists, the tool MUST append a timestamp in the format `_YYYYMMDD_HHMMSS` to the directory name |
| FR-3.6 | The tool MUST save extracted images to the subdirectory using the image `id` from the API response as the filename (e.g., `img_12345.png`) |
| FR-3.7 | The tool MUST determine the image format from the base64 data header or default to `.png` |
| FR-3.8 | The tool MUST save a `content.md` file containing the markdown text |
| FR-3.9 | The tool MUST replace all base64 image references in the markdown with relative paths to the saved image files |
| FR-3.10 | The tool MUST return a JSON object containing: |
|         | - `output_directory`: absolute path to the created directory |
|         | - `markdown_file`: absolute path to the saved markdown file |
|         | - `images`: list of saved image filenames |
| FR-3.11 | The tool MUST return an error if `output_dir` does not exist or is not writable |

**Example Invocation:**
```json
{
  "tool": "extract_markdown_with_images",
  "arguments": {
    "file_path": "/Users/ben/documents/report.pdf",
    "output_dir": "/Users/ben/extracted"
  }
}
```

**Example Response:**
```json
{
  "output_directory": "/Users/ben/extracted/report",
  "markdown_file": "/Users/ben/extracted/report/content.md",
  "images": ["img_abc123.png", "img_def456.png"]
}
```

### FR-4: Authentication & Configuration

| ID | Requirement |
|----|-------------|
| FR-4.1 | The server MUST read the Mistral API key from the `MISTRAL_API_KEY` environment variable |
| FR-4.2 | The server MUST refuse to start if `MISTRAL_API_KEY` is not set |
| FR-4.3 | The server MUST NOT log or expose the API key in error messages |

### FR-5: Write Directory Sandbox

| ID | Requirement |
|----|-------------|
| FR-5.1 | The server MUST read the allowed write directory from the `MISTRAL_OCR_ALLOWED_DIR` environment variable |
| FR-5.2 | The server MUST refuse to start if `MISTRAL_OCR_ALLOWED_DIR` is not set |
| FR-5.3 | The server MUST validate that `MISTRAL_OCR_ALLOWED_DIR` is an absolute path |
| FR-5.4 | The server MUST validate that `MISTRAL_OCR_ALLOWED_DIR` exists and is a directory |
| FR-5.5 | For `extract_markdown_with_images`, the server MUST reject any `output_dir` that is not at or below `MISTRAL_OCR_ALLOWED_DIR` |
| FR-5.6 | Path validation MUST canonicalize paths (resolve symlinks, eliminate `..`) before comparison to prevent traversal attacks |
| FR-5.7 | If `output_dir` is outside the allowed directory, the tool MUST return an error: `"output_dir must be within the allowed directory: {MISTRAL_OCR_ALLOWED_DIR}"` |

**Example Configuration:**
```bash
export MISTRAL_API_KEY="your-api-key"
export MISTRAL_OCR_ALLOWED_DIR="/Users/ben/Development"
```

**Validation Examples:**

| `MISTRAL_OCR_ALLOWED_DIR` | `output_dir` | Result |
|---------------------------|--------------|--------|
| `/Users/ben/Development` | `/Users/ben/Development/project/output` | ✅ Allowed |
| `/Users/ben/Development` | `/Users/ben/Development` | ✅ Allowed (exact match) |
| `/Users/ben/Development` | `/Users/ben/Documents` | ❌ Rejected |
| `/Users/ben/Development` | `/Users/ben/Development/../Documents` | ❌ Rejected (resolves outside) |

### FR-6: Error Handling

| ID | Requirement |
|----|-------------|
| FR-5.1 | All errors MUST be returned as MCP tool errors with descriptive messages |
| FR-5.2 | API errors from Mistral MUST include the HTTP status code and error message |
| FR-5.3 | File system errors MUST indicate the specific path and operation that failed |

---

## Non-Functional Requirements

### NFR-1: Deployment & Packaging

| ID | Requirement |
|----|-------------|
| NFR-1.1 | The package MUST be installable via `pip install` |
| NFR-1.2 | The package MUST be runnable via `uvx mistral-ocr-mcp` without prior installation |
| NFR-1.3 | The package MUST use `pyproject.toml` for project configuration |
| NFR-1.4 | The package MUST specify Python 3.10+ as the minimum version |

### NFR-2: Dependencies

| ID | Requirement |
|----|-------------|
| NFR-2.1 | The server MUST use the official `mistralai` Python client |
| NFR-2.2 | The server MUST use `mcp` (Model Context Protocol SDK) for server implementation |
| NFR-2.3 | Dependencies MUST be minimal to reduce install time for `uvx` |

### NFR-3: Performance

| ID | Requirement |
|----|-------------|
| NFR-3.1 | The server SHOULD handle documents up to 50 pages without timeout |
| NFR-3.2 | Image decoding and file I/O SHOULD be performed efficiently to minimize memory usage |

### NFR-4: Security

| ID | Requirement |
|----|-------------|
| NFR-4.1 | The server MUST validate that `file_path` is an absolute path |
| NFR-4.2 | The server MUST validate that `output_dir` is an absolute path |
| NFR-4.3 | The server MUST canonicalize all paths before validation to prevent traversal attacks |
| NFR-4.4 | The server MUST enforce the `MISTRAL_OCR_ALLOWED_DIR` sandbox for all write operations (see FR-5) |

---

## Out of Scope

The following are explicitly **NOT** part of this project:

1. **Multi-file batch processing** - Only single files are supported per tool invocation
2. **URL input** - Only local file paths are accepted; the LLM must download remote files first
3. **Page selection** - All pages are processed; no `pages` parameter is exposed
4. **Custom OCR models** - The server uses `mistral-ocr-latest` exclusively
5. **Streaming responses** - Results are returned in full after processing completes
6. **Image format conversion** - Images are saved in their original format as returned by the API
7. **Thumbnail generation** - Full-size images only

---

## API Model Reference

The server uses the Mistral OCR API with the following key structures:

### Request
```python
client.ocr.process(
    model="mistral-ocr-latest",
    document={
        "type": "document_url" | "image_url",  # or base64 variants
        "document_url": "..." | "image_url": "..."
    },
    include_image_base64=True | False
)
```

### Response Structure
```python
OCRResponse:
  pages: List[OCRPageObject]
    - index: int
    - markdown: str
    - images: List[OCRImageObject]
        - id: str  # e.g., "img_abc123"
        - image_base64: str  # base64-encoded image data
        - top_left_x, top_left_y, bottom_right_x, bottom_right_y: int
```

---

## Appendix: Directory Structure Example

After running `extract_markdown_with_images` on `/path/to/quarterly-report.pdf` with `output_dir=/output`:

```
/output/
  quarterly-report/
    content.md          # Markdown with image references like ![](./img_abc123.png)
    img_abc123.png      # First extracted image
    img_def456.png      # Second extracted image
    img_ghi789.jpeg     # Third extracted image (format from API)
```

If run again, the directory would be:
```
/output/
  quarterly-report/
    ...
  quarterly-report_20260102_143022/
    ...
```
