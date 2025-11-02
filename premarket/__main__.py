"""Legacy entrypoint retained for backward compatibility."""

from __future__ import annotations

from bots.screener.service import premarket_scan


def main() -> None:
    """Execute a single SteadyAlpha Screener run."""

    premarket_scan()


if __name__ == "__main__":  # pragma: no cover
    main()
