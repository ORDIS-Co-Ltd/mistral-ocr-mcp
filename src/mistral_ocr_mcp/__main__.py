"""MCP server implementation for Mistral OCR."""

import sys
from .server import run


def main() -> int:
    """Main entry point for the MCP server."""
    try:
        run()
        return 0
    except NotImplementedError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
