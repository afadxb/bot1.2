"""Command-line entry point reading environment configuration."""

from __future__ import annotations

import argparse
import os
from datetime import date, datetime
from pathlib import Path
from typing import Optional

from dateutil import tz

from . import orchestrate
from . import utils

_BOOL_TRUE = {"1", "true", "yes", "y", "on"}
_BOOL_FALSE = {"0", "false", "no", "n", "off"}


def _load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


def _parse_bool(value: Optional[str], key: str) -> Optional[bool]:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    normalized = value.strip().lower()
    if normalized in _BOOL_TRUE:
        return True
    if normalized in _BOOL_FALSE:
        return False
    raise ValueError(f"{key} must be a boolean-like value")


def _today_in_timezone(tz_name: str) -> date:
    tzinfo = tz.gettz(tz_name)
    if tzinfo is None:
        tzinfo = tz.gettz(utils.DEFAULT_TZ_NAME)
    now = datetime.now(tz=tzinfo)
    return now.date()


def _parse_date(value: str, key: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as exc:  # pragma: no cover - invalid user input
        raise SystemExit(f"{key} must be in YYYY-MM-DD format") from exc


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Premarket Top-N watchlist runner")
    parser.add_argument("--config", help="Override PREMARKET_CONFIG_PATH")
    parser.add_argument("--out", help="Override PREMARKET_OUT_DIR")
    parser.add_argument("--top-n", type=int, help="Override PREMARKET_TOP_N")
    parser.add_argument("--use-cache", help="Override PREMARKET_USE_CACHE (true/false)")
    parser.add_argument("--news", help="Override PREMARKET_NEWS_ENABLED (true/false)")
    parser.add_argument("--log-file", help="Override PREMARKET_LOG_FILE")
    parser.add_argument("--date", help="Override PREMARKET_DATE (YYYY-MM-DD)")
    parser.add_argument("--tz", help="Override PREMARKET_TZ")
    parser.add_argument("--max-per-sector", type=float, help="Override PREMARKET_MAX_PER_SECTOR")
    parser.add_argument("--fail-on-empty", help="Override PREMARKET_FAIL_ON_EMPTY (true/false)")
    parser.add_argument("--env-file", help="Path to .env file", default=".env")
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    env_path = Path(args.env_file)
    _load_env_file(env_path)

    overrides: set[str] = set()

    timezone = utils.DEFAULT_TZ_NAME
    timezone_env = utils.env_str("PREMARKET_TZ")
    if timezone_env is not None:
        timezone = timezone_env
        overrides.add("PREMARKET_TZ")
    if args.tz:
        timezone = args.tz
        overrides.add("PREMARKET_TZ")

    run_date = _today_in_timezone(timezone)
    env_date = utils.env_str("PREMARKET_DATE")
    if env_date is not None:
        run_date = _parse_date(env_date, "PREMARKET_DATE")
        overrides.add("PREMARKET_DATE")
    if args.date:
        run_date = _parse_date(args.date, "PREMARKET_DATE")
        overrides.add("PREMARKET_DATE")

    config_value = "config/strategy.yaml"
    config_env = utils.env_str("PREMARKET_CONFIG_PATH")
    if config_env is not None:
        config_value = config_env
        overrides.add("PREMARKET_CONFIG_PATH")
    if args.config:
        config_value = args.config
        overrides.add("PREMARKET_CONFIG_PATH")

    out_value = utils.env_str("PREMARKET_OUT_DIR")
    output_base = Path(out_value) if out_value is not None else Path("data/watchlists")
    if out_value is not None:
        overrides.add("PREMARKET_OUT_DIR")
    if args.out:
        output_base = Path(args.out)
        overrides.add("PREMARKET_OUT_DIR")

    top_n: Optional[int] = None
    top_env = utils.env_str("PREMARKET_TOP_N")
    if top_env is not None:
        try:
            top_n = int(top_env)
        except ValueError as exc:
            raise SystemExit("PREMARKET_TOP_N must be an integer") from exc
        overrides.add("PREMARKET_TOP_N")
    if args.top_n is not None:
        top_n = args.top_n
        overrides.add("PREMARKET_TOP_N")

    use_cache = True
    try:
        env_use_cache = _parse_bool(utils.env_str("PREMARKET_USE_CACHE"), "PREMARKET_USE_CACHE")
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    if env_use_cache is not None:
        use_cache = env_use_cache
        overrides.add("PREMARKET_USE_CACHE")
    try:
        cli_use_cache = _parse_bool(args.use_cache, "PREMARKET_USE_CACHE") if args.use_cache else None
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    if cli_use_cache is not None:
        use_cache = cli_use_cache
        overrides.add("PREMARKET_USE_CACHE")

    try:
        news_override = _parse_bool(utils.env_str("PREMARKET_NEWS_ENABLED"), "PREMARKET_NEWS_ENABLED")
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    if news_override is not None:
        overrides.add("PREMARKET_NEWS_ENABLED")
    try:
        cli_news = _parse_bool(args.news, "PREMARKET_NEWS_ENABLED") if args.news else None
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    if cli_news is not None:
        news_override = cli_news
        overrides.add("PREMARKET_NEWS_ENABLED")

    log_file: Optional[Path] = None
    log_env = utils.env_str("PREMARKET_LOG_FILE")
    if log_env is not None:
        log_file = Path(log_env)
        overrides.add("PREMARKET_LOG_FILE")
    if args.log_file:
        log_file = Path(args.log_file)
        overrides.add("PREMARKET_LOG_FILE")

    max_per_sector = None
    max_per_sector_env = utils.env_str("PREMARKET_MAX_PER_SECTOR")
    if max_per_sector_env is not None:
        try:
            max_per_sector = float(max_per_sector_env)
        except ValueError as exc:
            raise SystemExit("PREMARKET_MAX_PER_SECTOR must be numeric") from exc
        overrides.add("PREMARKET_MAX_PER_SECTOR")
    if args.max_per_sector is not None:
        max_per_sector = args.max_per_sector
        overrides.add("PREMARKET_MAX_PER_SECTOR")

    fail_on_empty = False
    try:
        env_fail = _parse_bool(utils.env_str("PREMARKET_FAIL_ON_EMPTY"), "PREMARKET_FAIL_ON_EMPTY")
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    if env_fail is not None:
        fail_on_empty = env_fail
        overrides.add("PREMARKET_FAIL_ON_EMPTY")
    try:
        cli_fail = _parse_bool(args.fail_on_empty, "PREMARKET_FAIL_ON_EMPTY") if args.fail_on_empty else None
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    if cli_fail is not None:
        fail_on_empty = cli_fail
        overrides.add("PREMARKET_FAIL_ON_EMPTY")

    params = orchestrate.RunParams(
        config_path=Path(config_value),
        output_base_dir=output_base,
        top_n=top_n,
        use_cache=use_cache,
        news_override=news_override,
        log_file=log_file,
        run_date=run_date,
        timezone=timezone,
        fail_on_empty=fail_on_empty,
        max_per_sector=max_per_sector,
        env_overrides=sorted(overrides),
    )

    return orchestrate.run(params)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
