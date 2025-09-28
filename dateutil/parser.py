"""Minimal parser for ISO-like dates."""

from __future__ import annotations

from datetime import datetime


class ParserError(ValueError):
    """Exception raised on parse errors."""


def parse(value: str) -> datetime:
    try:
        return datetime.fromisoformat(value)
    except ValueError as exc:
        raise ParserError(str(exc)) from exc
