import json
import math
from datetime import datetime, timezone

import pytest

from premarket import news_probe
from premarket.orchestrate import NewsModel
from premarket import utils


@pytest.fixture(autouse=True)
def restore_http_get(monkeypatch):
    original = news_probe._http_get
    yield
    monkeypatch.setattr(news_probe, "_http_get", original)


def test_probe_uses_finviz_csv(monkeypatch):
    cfg = NewsModel(enabled=True, freshness_hours=24, finviz_url="https://elite.finviz.com/news_export.ashx?v=3")

    now = datetime(2024, 4, 20, 10, 0, tzinfo=utils.EASTERN)
    monkeypatch.setattr(news_probe.utils, "now_eastern", lambda: now)

    csv_content = """Date,Time,Ticker,Title,URL\n04/20/2024,09:15 AM,AAA,Foo,https://example.com\n"""

    def fake_http_get(url: str) -> str:
        assert "news_export" in url
        return csv_content

    monkeypatch.setattr(news_probe, "_http_get", fake_http_get)

    result = news_probe.probe(["AAA", "BBB"], cfg)

    assert result["AAA"]["category"] == "finviz"
    assert math.isclose(result["AAA"]["freshness_hours"], 0.75, rel_tol=1e-6)
    assert result["BBB"]["freshness_hours"] is None


def test_probe_uses_finnhub_when_no_finviz(monkeypatch):
    cfg = NewsModel(enabled=True, freshness_hours=24, finnhub_token="token")

    now = datetime(2024, 4, 20, 10, 0, tzinfo=utils.EASTERN)
    monkeypatch.setattr(news_probe.utils, "now_eastern", lambda: now)

    article_time = datetime(2024, 4, 20, 13, 0, tzinfo=timezone.utc)  # 9:00 AM ET
    payload = json.dumps([{"datetime": article_time.timestamp()}])

    def fake_http_get(url: str) -> str:
        assert "company-news" in url
        return payload

    monkeypatch.setattr(news_probe, "_http_get", fake_http_get)

    result = news_probe.probe(["CCC"], cfg)

    assert result["CCC"]["category"] == "finnhub"
    assert math.isclose(result["CCC"]["freshness_hours"], 1.0, rel_tol=1e-6)


def test_probe_prefers_latest_timestamp(monkeypatch):
    cfg = NewsModel(enabled=True, freshness_hours=24)

    now = datetime(2024, 4, 20, 12, 0, tzinfo=utils.EASTERN)
    monkeypatch.setattr(news_probe.utils, "now_eastern", lambda: now)

    finviz_dt = datetime(2024, 4, 20, 9, 0, tzinfo=utils.EASTERN)
    finnhub_dt = datetime(2024, 4, 20, 11, 30, tzinfo=utils.EASTERN)

    monkeypatch.setattr(
        news_probe,
        "_finviz_latest",
        lambda symbols, url: {symbols[0]: (finviz_dt, "finviz")} if symbols else {},
    )
    monkeypatch.setattr(
        news_probe,
        "_finnhub_latest",
        lambda symbols, token, days: {symbols[0]: (finnhub_dt, "finnhub")} if symbols else {},
    )

    result = news_probe.probe(["DDD"], cfg)

    assert result["DDD"]["category"] == "finnhub"
    assert math.isclose(result["DDD"]["freshness_hours"], 0.5, rel_tol=1e-6)

def test_probe_handles_finviz_24_hour_datetime(monkeypatch):
    cfg = NewsModel(enabled=True, freshness_hours=24, finviz_url="https://elite.finviz.com/news_export.ashx?v=3")

    now = datetime(2025, 9, 27, 16, 0, tzinfo=utils.EASTERN)
    monkeypatch.setattr(news_probe.utils, "now_eastern", lambda: now)

    csv_content = (
        "Title,Source,Date,Url,Category,Ticker\n"
        "AI Race Analysis,Benzinga,9/27/2025 15:11,https://example.com,Stock,\"GOOG,GOOGL\"\n"
    )

    def fake_http_get(url: str) -> str:
        assert "news_export" in url
        return csv_content

    monkeypatch.setattr(news_probe, "_http_get", fake_http_get)

    result = news_probe.probe(["goog", "googl", "msft"], cfg)

    assert result["GOOG"]["category"] == "finviz"
    assert result["GOOGL"]["category"] == "finviz"
    assert math.isclose(result["GOOG"]["freshness_hours"], 0.8166666, rel_tol=1e-6)
    assert result["MSFT"]["freshness_hours"] is None

