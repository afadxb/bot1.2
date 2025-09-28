"""Fetch lightweight news metadata from Finviz and Finnhub."""

from __future__ import annotations

import csv
import io
import json
import logging
from datetime import datetime, timedelta, timezone
from typing import Dict, Iterable, List, Optional, Tuple
from urllib import request as urllib_request
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import requests
from dateutil import parser as date_parser

from . import utils

LOGGER = logging.getLogger(__name__)


def _is_stub_network_error(exc: Exception) -> bool:
    return "Network access disabled" in str(exc)


def _http_get(url: str) -> str:
    try:
        response = requests.get(url, timeout=15)
        response.raise_for_status()
        return response.text
    except requests.RequestException as exc:
        if _is_stub_network_error(exc):
            LOGGER.info("requests stub detected for %s; retrying via urllib", utils.redact_token(url))
            return _http_get_urllib(url)
        raise


def _http_get_urllib(url: str) -> str:
    try:
        with urllib_request.urlopen(url, timeout=15) as response:  # type: ignore[arg-type]
            status = getattr(response, "status", 200)
            if status and status >= 400:
                raise requests.RequestException(f"HTTP {status}")
            headers = getattr(response, "headers", None)
            encoding = headers.get_content_charset() if headers is not None else None
            content = response.read()
            return content.decode(encoding or "utf-8", errors="replace")
    except (HTTPError, URLError, OSError) as exc:  # pragma: no cover - defensive
        raise requests.RequestException(str(exc)) from exc


def _normalise_symbols(symbols: Iterable[str]) -> List[str]:
    result: List[str] = []
    seen: set[str] = set()
    for symbol in symbols:
        if not symbol:
            continue
        trimmed = symbol.strip().upper()
        if not trimmed or trimmed in seen:
            continue
        result.append(trimmed)
        seen.add(trimmed)
    return result


def _build_finviz_url(base_url: str, symbols: List[str]) -> str:
    if not base_url:
        return ""
    if not symbols:
        return base_url

    parsed = urlsplit(base_url)
    query_pairs = parse_qsl(parsed.query, keep_blank_values=True)
    existing: List[str] = []
    filtered_pairs: List[Tuple[str, str]] = []
    for key, value in query_pairs:
        if key == "t":
            existing.extend(
                [item.strip().upper() for item in value.split(",") if item.strip()]
            )
        else:
            filtered_pairs.append((key, value))

    merged = sorted({*existing, *symbols})
    if merged:
        filtered_pairs.append(("t", ",".join(merged)))

    query = urlencode(filtered_pairs)
    return urlunsplit((parsed.scheme, parsed.netloc, parsed.path, query, parsed.fragment))


def _split_tickers(raw: str) -> List[str]:
    candidates = [segment.strip().upper() for segment in raw.replace(";", ",").split(",")]
    return [candidate for candidate in candidates if candidate]


def _clean_datetime_string(value: str) -> str:
    cleaned = value.replace("ET", "").replace("EDT", "").replace("EST", "")
    cleaned = cleaned.replace("UTC", "").strip()
    return " ".join(cleaned.split())


_FINVIZ_DATETIME_FORMATS = [
    "%m/%d/%Y %I:%M %p",
    "%m/%d/%y %I:%M %p",
    "%m/%d/%Y %H:%M",
    "%m/%d/%y %H:%M",

    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%d %H:%M",
    "%Y-%m-%dT%H:%M:%S",
    "%Y-%m-%d",
]


def _parse_finviz_timestamp(row: dict) -> Optional[datetime]:
    date_value = row.get("Date") or row.get("Published") or row.get("Date & Time")
    time_value = row.get("Time") or row.get("Time (ET)")
    datetime_value = row.get("DateTime")

    candidates: List[str] = []
    if isinstance(date_value, str) and date_value:
        if isinstance(time_value, str) and time_value:
            candidates.append(f"{date_value} {time_value}")
        candidates.append(date_value)
    if isinstance(datetime_value, str) and datetime_value:
        candidates.append(datetime_value)

    for candidate in candidates:
        cleaned = _clean_datetime_string(candidate)
        for fmt in _FINVIZ_DATETIME_FORMATS:
            try:
                parsed = datetime.strptime(cleaned, fmt)
            except ValueError:
                continue
            parsed = parsed.replace(tzinfo=utils.EASTERN)
            return parsed
        try:
            parsed_iso = _dateutil_parse(cleaned)
        except ValueError:
            continue
        if parsed_iso.tzinfo is None:
            parsed_iso = parsed_iso.replace(tzinfo=utils.EASTERN)
        else:
            parsed_iso = parsed_iso.astimezone(utils.EASTERN)
        return parsed_iso
    return None


def _dateutil_parse(value: str) -> datetime:
    try:
        return date_parser.parse(value)
    except (AttributeError, TypeError, ValueError):
        raise ValueError from None
    except date_parser.ParserError as exc:  # type: ignore[attr-defined]
        raise ValueError(str(exc)) from exc


def _finviz_latest(symbols: List[str], base_url: Optional[str]) -> Dict[str, Tuple[datetime, str]]:
    if not base_url or not symbols:
        return {}

    url = _build_finviz_url(base_url, symbols)
    try:
        content = _http_get(url)
    except requests.RequestException as exc:
        LOGGER.warning("Failed to fetch Finviz news: %s", exc)
        return {}

    reader = csv.DictReader(io.StringIO(content))
    target_set = set(symbols)
    latest: Dict[str, Tuple[datetime, str]] = {}
    for row in reader:
        ticker_field = (
            row.get("Ticker")
            or row.get("Tickers")
            or row.get("Ticker(s)")
            or row.get("Symbol")
            or row.get("Symbols")
            or row.get("Related")
        )
        if not ticker_field:
            continue

        parsed_ts = _parse_finviz_timestamp(row)
        if parsed_ts is None:
            continue

        for ticker in _split_tickers(str(ticker_field)):
            if ticker not in target_set:
                continue
            current = latest.get(ticker)
            if current is None or parsed_ts > current[0]:
                latest[ticker] = (parsed_ts, "finviz")

    return latest


def _parse_finnhub_timestamp(value) -> Optional[datetime]:
    if isinstance(value, (int, float)):
        seconds = float(value)
    elif isinstance(value, str):
        try:
            seconds = float(value)
        except ValueError:
            return None
    else:
        return None

    return datetime.fromtimestamp(seconds, tz=timezone.utc).astimezone(utils.EASTERN)


def _finnhub_latest(
    symbols: List[str], token: Optional[str], lookback_days: int
) -> Dict[str, Tuple[datetime, str]]:
    if not token or not symbols:
        return {}

    now = utils.now_eastern()
    start_date = (now - timedelta(days=max(lookback_days, 1))).date().isoformat()
    end_date = now.date().isoformat()

    latest: Dict[str, Tuple[datetime, str]] = {}
    for symbol in symbols:
        params = urlencode({"symbol": symbol, "from": start_date, "to": end_date, "token": token})
        url = f"https://finnhub.io/api/v1/company-news?{params}"
        try:
            payload = _http_get(url)
            data = json.loads(payload)
        except (requests.RequestException, json.JSONDecodeError) as exc:
            LOGGER.warning("Failed to fetch Finnhub news for %s: %s", symbol, exc)
            continue

        if not isinstance(data, list):
            continue

        for item in data:
            if not isinstance(item, dict):
                continue
            timestamp = _parse_finnhub_timestamp(item.get("datetime") or item.get("time"))
            if timestamp is None:
                continue
            current = latest.get(symbol)
            if current is None or timestamp > current[0]:
                latest[symbol] = (timestamp, "finnhub")

    return latest


def _merge_sources(
    symbols: List[str],
    finviz_data: Dict[str, Tuple[datetime, str]],
    finnhub_data: Dict[str, Tuple[datetime, str]],
) -> Dict[str, Tuple[datetime, str]]:
    merged: Dict[str, Tuple[datetime, str]] = {}
    for source in (finviz_data, finnhub_data):
        for symbol, payload in source.items():
            if symbol not in symbols:
                continue
            current = merged.get(symbol)
            if current is None or payload[0] > current[0]:
                merged[symbol] = payload
    return merged


def probe(symbols: Iterable[str], cfg) -> Dict[str, dict]:
    """Probe Finviz and Finnhub news sources for the provided symbols."""

    normalized = _normalise_symbols(symbols)
    now = utils.now_eastern()

    finviz_url = getattr(cfg, "finviz_url", None) or utils.env_str("FINVIZ_NEWS_EXPORT_URL")
    finnhub_token = getattr(cfg, "finnhub_token", None) or utils.env_str("FINNHUB_API_KEY")
    finnhub_days = getattr(cfg, "finnhub_days", 3)

    finviz_data = _finviz_latest(normalized, finviz_url)
    finnhub_data = _finnhub_latest(normalized, finnhub_token, finnhub_days)
    merged = _merge_sources(normalized, finviz_data, finnhub_data)

    results: Dict[str, dict] = {}
    for symbol in normalized:
        payload = merged.get(symbol)
        if payload is None:
            results[symbol] = {
                "freshness_hours": None,
                "category": None,
                "timestamp": utils.timestamp_iso(now),
            }
            continue

        timestamp, category = payload
        delta = max(0.0, (now - timestamp).total_seconds() / 3600.0)
        results[symbol] = {
            "freshness_hours": float(delta),
            "category": category,
            "timestamp": utils.timestamp_iso(timestamp),
        }

    return results
