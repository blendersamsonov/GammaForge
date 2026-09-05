"""Start the local browser UI only when explicitly invoked."""

from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="GammaForge local browser UI")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--no-browser", action="store_true", help="do not open a browser")
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("port must be between 1 and 65535")
    try:
        from .app import start
        start(port=args.port, show=not args.no_browser)
    except ModuleNotFoundError as exc:
        if exc.name not in {"nicegui", "plotly"}:
            raise
        parser.exit(1, 'Install the browser interface first: pip install -e ".[gui]"\n')


if __name__ == "__main__":
    main()
