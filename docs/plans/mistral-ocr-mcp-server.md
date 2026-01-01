# Implementation Plan: Mistral OCR MCP Server

## Target Files

- `pyproject.toml`
- `src/mistral_ocr_mcp/__init__.py`
- `src/mistral_ocr_mcp/__main__.py`
- `src/mistral_ocr_mcp/server.py`
- `src/mistral_ocr_mcp/config.py`
- `src/mistral_ocr_mcp/path_sandbox.py`
- `src/mistral_ocr_mcp/mistral_client.py`
- `src/mistral_ocr_mcp/extraction.py`
- `src/mistral_ocr_mcp/images.py`
- `src/mistral_ocr_mcp/markdown_rewrite.py`
- `tests/test_config.py`
- `tests/test_path_sandbox.py`
- `tests/test_output_dir_naming.py`
- `tests/test_images.py`
- `tests/test_markdown_rewrite.py`

## Plan

1. **Create the Python package skeleton (src layout).**
   - Add `pyproject.toml` with Python `>=3.10`, minimal runtime deps (`mcp`, `mistralai`), and a console script entrypoint named `mistral-ocr-mcp`.
   - Create `src/mistral_ocr_mcp/` package with `__init__.py` and `__main__.py`.
   - Ensure `uvx mistral-ocr-mcp` is supported via `[project.scripts]` (for local testing: `uvx --from . mistral-ocr-mcp`).

2. **Implement startup configuration validation (FR-4, FR-5).**
   - In `config.py`, load `MISTRAL_API_KEY` and `MISTRAL_OCR_ALLOWED_DIR` from environment.
   - Refuse to start if either is missing.
   - Validate `MISTRAL_OCR_ALLOWED_DIR`:
     - Must be an absolute path.
     - Must exist and be a directory.
     - Must be canonicalized (`Path(...).resolve(strict=True)`) and stored for later comparisons.
   - Ensure error messages never include the API key value.

3. **Add canonical path validation helpers for tool inputs (NFR-4, FR-2, FR-3).**
   - In `path_sandbox.py`, implement helpers that:
     - Validate `file_path` is absolute, exists, and has a supported extension (`.pdf`, `.png`, `.jpg`, `.jpeg`, `.webp`, `.gif`).
     - Validate `output_dir` is absolute, exists, is a directory, and is writable.
     - Canonicalize all paths before use (`resolve(strict=True)`) to eliminate symlinks and `..`.
     - Enforce sandbox: reject any `output_dir` that is not equal to, or a descendant of, `MISTRAL_OCR_ALLOWED_DIR`.
       - On violation, return the exact error message:
         - `output_dir must be within the allowed directory: {MISTRAL_OCR_ALLOWED_DIR}`

4. **Wrap the Mistral OCR call behind a small client adapter (FR-2.4, FR-3.3).**
   - In `mistral_client.py`, create a small function/class to:
     - Initialize the official `mistralai` client using `MISTRAL_API_KEY`.
     - For local files, upload via `client.files.upload(...)` with `purpose="ocr"` using a file handle (`input_path.open("rb")`) to avoid loading the full document into memory, then call `client.ocr.process(...)` with `document={"type": "file", "file_id": uploaded_file.id}`.
     - Always use `model="mistral-ocr-latest"`.
     - Toggle `include_image_base64` based on tool.
     - Normalize API errors into a single exception type that includes HTTP status code + error message (FR-6).

5. **Implement markdown extraction logic (US-1, FR-2.5).**
   - In `extraction.py`, implement a function that:
     - Calls the OCR adapter with `include_image_base64=False`.
     - Concatenates page markdown into a single markdown string in a predictable way (e.g., `"\n\n".join(p.markdown for p in pages)` if the response is page-based).

6. **Implement output directory creation and collision handling (FR-3.4, FR-3.5).**
    - In `extraction.py` (or a small helper module), implement:
      - `output_subdir = output_dir / input_path.stem`.
      - If it already exists, append a timestamp in the exact format `_YYYYMMDD_HHMMSS` (e.g., `_20260102_143022`).
      - Use a loop to guarantee uniqueness if the timestamped directory already exists.

7. **Implement image decoding and saving (FR-3.6, FR-3.7).**
   - In `images.py`, implement:
     - Per-image decoding from the SDK response (`id`, `image_base64`).
     - Determine file extension from a data-URI header (e.g., `data:image/jpeg;base64,...`) and default to `.png` if not present/recognized.
     - Save each image to `<output_subdir>/{image_id}{ext}`.
     - Return a list of filenames (not full paths) for the tool response.
   - Ensure filesystem errors include the failing path and operation (FR-6).

8. **Rewrite markdown to avoid base64 image data (FR-3.8, FR-3.9).**
   - In `markdown_rewrite.py`, implement a rewrite function that:
     - Locates markdown image links that embed base64 (`data:image/...;base64,....`).
     - Replaces each embedded base64 reference with a relative path `./{image_id}{ext}` (matching the appendix examples).
     - Prefer a deterministic mapping strategy:
       - First try exact-match replacement using the `image_base64` strings returned by the API.
       - If the markdown/image relationship is not 1:1, fall back to sequential replacement in document order.

9. **Implement Tool 2 end-to-end: `extract_markdown_with_images` (US-2, FR-3).**
    - In `extraction.py`, implement the orchestrator:
      - Validate `file_path` and `output_dir`.
      - Create output subdirectory.
      - Call OCR with `include_image_base64=True`.
      - Save images.
      - Rewrite markdown.
      - Save markdown to `content.md` (exact filename required per FR-3.8).
      - Return `{output_directory, markdown_file, images}` with absolute paths for the first two and filenames for `images`.

10. **Implement the MCP server with exactly two tools (FR-1, FR-2, FR-3).**
    - In `server.py`, use the official MCP Python SDK server over stdio (`mcp.server.stdio`).
    - Expose exactly these two tools via `@server.list_tools()`:
      - `extract_markdown` with input schema `{file_path}` and output schema `{content: string}`.
      - `extract_markdown_with_images` with input schema `{file_path, output_dir}` and output schema `{output_directory, markdown_file, images}`.
    - Implement `@server.call_tool()`:
      - Route to the correct handler based on tool name (e.g., `if tool_name == "extract_markdown":`)
      - Reuse `path_sandbox.py` validators for file-type checks so errors are consistent (e.g., `"Unsupported file type. Supported types: .pdf, .png, .jpg, .jpeg, .webp, .gif"`).
      - Convert validation, API, and filesystem exceptions into MCP tool errors with descriptive messages (FR-6).
    - In `__main__.py`, validate env vars (FR-4/5) before starting the server.

11. **Add focused unit tests (no network).**
    - Use `pytest` as a dev dependency.
    - Test in isolation (monkeypatch/mock the OCR adapter):
      - Missing env vars refusal.
      - Allowed-dir sandbox behavior including canonicalization cases (`..`, symlinks).
      - Output subdirectory naming + timestamp collision behavior.
      - Base64 header parsing to extension + defaulting to `.png`.
      - Markdown rewrite replacing base64 images with `./img_id.ext`.

## Verification

- `python -m pip install -e ".[dev]"`
- `pytest`
- Manual smoke (requires credentials):
  - `export MISTRAL_API_KEY=...`
  - `export MISTRAL_OCR_ALLOWED_DIR="/absolute/allowed/dir"`
  - `uvx --from . mistral-ocr-mcp` (server starts and refuses to start if env is missing)
  - Run a client tool call against stdio server to confirm:
    - `extract_markdown` returns markdown text.
    - `extract_markdown_with_images` creates `<output_dir>/<stem>[_timestamp]/content.md` and saves images, with base64 removed from markdown.
