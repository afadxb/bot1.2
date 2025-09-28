"""Debug-friendly entry point mirroring ``python -m premarket``."""

from __future__ import annotations

from typing import Optional, Sequence


def run(argv: Optional[Sequence[str]] = None) -> int:
    """Execute the packaged CLI with optional argument overrides.

    Parameters
    ----------
    argv:
        Optional sequence of command-line arguments. If omitted, the CLI
        consumes ``sys.argv`` just like ``python -m premarket``.
    """

    from premarket.__main__ import main as cli_main

    if argv is None:
        return cli_main(None)

    return cli_main(list(argv))


if __name__ == "__main__":  # pragma: no cover
    from sys import argv as sys_argv

    raise SystemExit(run(sys_argv[1:]))
