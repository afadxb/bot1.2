"""Debug-friendly entry point mirroring ``python -m premarket``."""

from __future__ import annotations

from typing import Optional, Sequence


def run(argv: Optional[Sequence[str]] = None) -> int:
    """Execute a single Screener run (arguments are ignored)."""

    from premarket.__main__ import main as cli_main

    cli_main()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(run())
