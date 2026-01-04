"""MCP server implementation for Mistral OCR."""

import sys
from .config import ConfigurationError, load_config
from .server import run


def main() -> int:
    """Main entry point for the MCP server."""
    try:
        # Load and validate configuration before starting server
        load_config()
        run()
        return 0
    except KeyboardInterrupt:
        return 0
    except ConfigurationError as e:
        print(f"Configuration error: {e}", file=sys.stderr)
        return 1
    except NotImplementedError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
