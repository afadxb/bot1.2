from datetime import datetime
from pathlib import Path

import pandas as pd
from dateutil import tz

from bots.screener import service


def _sample_csv(path: Path) -> None:
    rows = [
        {
            "Ticker": "AAA",
            "Company": "Alpha",
            "Sector": "Technology",
            "Industry": "Software",
            "Exchange": "NASDAQ",
            "Country": "USA",
            "Market Cap": "1,500,000,000",
            "P/E": "25",
            "Price": "25",
            "Change": "5%",
            "Gap": "4%",
            "Volume": "500000",
            "Average Volume (3m)": "2000000",
            "Relative Volume": "2.0",
            "Float": "50000000",
            "Float %": "38%",
            "Short Float": "10%",
            "After-Hours Change": "1%",
            "52-Week Range": "10 - 30",
            "Earnings Date": "2099-01-01",
            "Analyst Recom.": "Buy",
            "Insider Transactions": "1%",
            "Institutional Transactions": "2%",
            "Previous Close": "24",
        },
        {
            "Ticker": "BBB",
            "Company": "Beta",
            "Sector": "Healthcare",
            "Industry": "Biotech",
            "Exchange": "NYSE",
            "Country": "USA",
            "Market Cap": "2,000,000,000",
            "P/E": "30",
            "Price": "45",
            "Change": "3%",
            "Gap": "2.5%",
            "Volume": "600000",
            "Average Volume (3m)": "3000000",
            "Relative Volume": "1.6",
            "Float": "15000000",
            "Float %": "42%",
            "Short Float": "12%",
            "After-Hours Change": "0.5%",
            "52-Week Range": "20 - 60",
            "Earnings Date": "2099-01-01",
            "Analyst Recom.": "Strong Buy",
            "Insider Transactions": "0%",
            "Institutional Transactions": "3%",
            "Previous Close": "44",
        },
    ]
    pd.DataFrame(rows).to_csv(path, index=False)


def test_premarket_scan_smoke(monkeypatch, tmp_path, fake_connection):
    csv_path = tmp_path / "finviz.csv"
    _sample_csv(csv_path)

    tzinfo = tz.gettz("America/New_York")
    fixed_now = datetime(2024, 1, 2, 8, 45, tzinfo=tzinfo)

    config_src = Path("config/strategy.yaml")
    config_path = tmp_path / "config" / "strategy.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(config_src.read_text(), encoding="utf-8")

    monkeypatch.chdir(tmp_path)

    monkeypatch.setenv("TZ", "America/New_York")
    monkeypatch.setenv("DB_HOST", "localhost")
    monkeypatch.setenv("DB_PORT", "3306")
    monkeypatch.setenv("DB_USER", "tester")
    monkeypatch.setenv("DB_PASS", "secret")
    monkeypatch.setenv("DB_NAME", "screener")
    monkeypatch.setenv("FINVIZ_EXPORT_URL", "https://example.com/export")
    monkeypatch.setenv("JOB_SCREENER_PM_START", "08:30")
    monkeypatch.setenv("JOB_SCREENER_PM_END", "09:15")
    monkeypatch.setenv("JOB_SCREENER_PM_EVERY_MIN", "5")
    monkeypatch.setenv("MIN_GAP_PCT", "2.0")
    monkeypatch.setenv("MIN_PM_VOL", "400000")
    monkeypatch.setenv("TOP_N", "2")

    monkeypatch.setattr(service, "CONFIG_PATH", config_path)
    monkeypatch.setattr(service.loader_finviz, "download_csv", lambda url, out, use_cache: csv_path)
    monkeypatch.setattr(service.utils, "now_eastern", lambda: fixed_now)
    monkeypatch.setattr(service, "push_notification", lambda *args, **kwargs: None)
    monkeypatch.setattr(service, "news_probe", lambda symbols, cfg: {symbol: {"freshness_hours": 1} for symbol in symbols})

    monkeypatch.setattr(service, "init_pool", lambda: None)
    monkeypatch.setattr(service, "get_connection", lambda: fake_connection)

    result = service.premarket_scan()

    assert result.top_symbols
    assert fake_connection.run_summary
    assert fake_connection.candidates
    output_dir = Path("data/watchlists") / result.run_date.isoformat()
    assert (output_dir / "watchlist.csv").exists()
    assert len(fake_connection.candidates) == len(result.top_symbols)
