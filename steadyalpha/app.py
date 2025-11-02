"""Entrypoint for the SteadyAlpha Screener application."""

from __future__ import annotations

import logging
import os
import signal
import sys
from typing import Callable

from steadyalpha.scheduler import create_scheduler
from steadyalpha.settings import get_settings

LOGGER = logging.getLogger(__name__)


def _handle_shutdown(scheduler) -> Callable[[int, object], None]:
    def _inner(signum, frame) -> None:  # pragma: no cover - signal handlers are not test-friendly
        LOGGER.info("Received signal %s, shutting down", signum)
        scheduler.shutdown(wait=False)
        sys.exit(0)

    return _inner


def main() -> None:
    settings = get_settings()
    bot_role = os.getenv("BOT_ROLE", "screener").lower()
    if bot_role != "screener":
        LOGGER.info("BOT_ROLE=%s - nothing to schedule", bot_role)
        return

    scheduler = create_scheduler(settings.tz)

    from bots.screener.jobs import schedule_jobs

    schedule_jobs(scheduler)
    scheduler.start()

    signal.signal(signal.SIGINT, _handle_shutdown(scheduler))
    signal.signal(signal.SIGTERM, _handle_shutdown(scheduler))

    LOGGER.info("Scheduler started. Press Ctrl+C to exit.")
    signal.pause()  # pragma: no cover - blocks main thread


if __name__ == "__main__":  # pragma: no cover
    main()
