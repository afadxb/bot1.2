"""Minimal requests stub."""

from __future__ import annotations


class RequestException(Exception):
    """Base request exception."""


class Response:
    def __init__(self, text: str = "", status_code: int = 200) -> None:
        self.text = text
        self.status_code = status_code

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RequestException(f"HTTP {self.status_code}")


def get(url: str, timeout: int = 10) -> Response:
    raise RequestException("Network access disabled in test environment")
