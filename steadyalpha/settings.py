"""Application settings loaded from environment variables."""

from __future__ import annotations

from functools import lru_cache
from typing import Optional

from pydantic import BaseSettings, Field


class AppSettings(BaseSettings):
    """Pydantic settings for the SteadyAlpha Screener."""

    tz: str = Field(default="America/New_York", alias="TZ")
    db_host: str = Field(alias="DB_HOST")
    db_port: int = Field(default=3306, alias="DB_PORT")
    db_user: str = Field(alias="DB_USER")
    db_password: str = Field(alias="DB_PASS")
    db_name: str = Field(alias="DB_NAME")
    finviz_export_url: str = Field(alias="FINVIZ_EXPORT_URL")

    job_screener_pm_start: str = Field(default="08:30", alias="JOB_SCREENER_PM_START")
    job_screener_pm_end: str = Field(default="09:15", alias="JOB_SCREENER_PM_END")
    job_screener_pm_every_min: int = Field(default=5, alias="JOB_SCREENER_PM_EVERY_MIN")

    min_gap_pct: float = Field(default=2.0, alias="MIN_GAP_PCT")
    min_pm_volume: int = Field(default=200000, alias="MIN_PM_VOL")
    top_n: int = Field(default=40, alias="TOP_N")

    pushover_user_key: Optional[str] = Field(default=None, alias="PUSHOVER_USER_KEY")
    pushover_api_token: Optional[str] = Field(default=None, alias="PUSHOVER_API_TOKEN")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


@lru_cache(maxsize=1)
def get_settings() -> AppSettings:
    """Return cached application settings."""

    return AppSettings()  # type: ignore[call-arg]


__all__ = ["AppSettings", "get_settings"]
