"""Minimal Rich logging handler stub."""

from __future__ import annotations

import logging


class RichHandler(logging.StreamHandler):
    """Simplified handler that behaves like a standard stream handler."""

    def __init__(self, *args, **kwargs) -> None:  # noqa: D401
        super().__init__()
