"""Scheduler job definitions for the SteadyAlpha Screener."""

from __future__ import annotations

import logging

from apscheduler.schedulers.background import BackgroundScheduler

from bots.screener.service import premarket_scan
from steadyalpha.scheduler import add_windowed_job
from steadyalpha.settings import get_settings

LOGGER = logging.getLogger(__name__)


def schedule_jobs(scheduler: BackgroundScheduler) -> None:
    """Register jobs with the provided scheduler."""

    settings = get_settings()
    LOGGER.info(
        "Scheduling premarket scan every %s minutes between %s-%s",
        settings.job_screener_pm_every_min,
        settings.job_screener_pm_start,
        settings.job_screener_pm_end,
    )
    add_windowed_job(
        scheduler,
        premarket_scan,
        job_id="screener_premarket",
        minutes=settings.job_screener_pm_every_min,
        start=settings.job_screener_pm_start,
        end=settings.job_screener_pm_end,
        tz_name=settings.tz,
    )


__all__ = ["schedule_jobs"]
