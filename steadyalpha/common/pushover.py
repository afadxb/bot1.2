"""Pushover notification helpers."""

from __future__ import annotations

import logging
from typing import Optional

import requests
from requests import Response
from tenacity import retry, stop_after_attempt, wait_exponential

LOGGER = logging.getLogger(__name__)

PUSHOVER_ENDPOINT = "https://api.pushover.net/1/messages.json"


@retry(wait=wait_exponential(multiplier=1, min=2, max=30), stop=stop_after_attempt(3))
def _post(payload: dict[str, str]) -> Response:
    response = requests.post(PUSHOVER_ENDPOINT, data=payload, timeout=10)
    if response.status_code >= 500:
        response.raise_for_status()
    return response


def send(title: str, message: str, priority: int = 0, url: Optional[str] = None) -> None:
    """Send a notification via Pushover if credentials are configured."""

    from steadyalpha.settings import get_settings

    settings = get_settings()
    if not settings.pushover_api_token or not settings.pushover_user_key:
        LOGGER.debug("Pushover credentials missing; skipping notification")
        return

    payload = {
        "token": settings.pushover_api_token,
        "user": settings.pushover_user_key,
        "title": title,
        "message": message,
        "priority": str(priority),
    }
    if url:
        payload["url"] = url

    try:
        response = _post(payload)
        LOGGER.info("Pushover response: %s", response.status_code)
    except requests.RequestException as exc:  # pragma: no cover - network errors
        LOGGER.warning("Failed to send Pushover notification: %s", exc)


__all__ = ["send"]
