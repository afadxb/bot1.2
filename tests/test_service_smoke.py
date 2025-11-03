from datetime import datetime
from pathlib import Path
import math

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
            "Price": "$25.00",
            "Change": "5%",
            "Gap": "4%",
            "Volume": "500000",
            "Average Volume (3m)": "2.0M",
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
            "Price": "$45.50",
            "Change": "3%",
            "Gap": "2.5%",
            "Volume": "600000",
            "Average Volume (3m)": "3.5M",
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

    fake_connection.securities = {"AAA": 101, "BBB": 202}
    fake_connection.security_metadata = {
        "AAA": {"name": "Legacy", "sector": "Old"},
        "BBB": {"name": "Beta", "sector": "Healthcare"},
    }

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
    assert fake_connection.shortlists
    output_dir = Path("data/watchlists") / result.run_date.isoformat()
    assert (output_dir / "watchlist.csv").exists()
    assert len(fake_connection.shortlists) == len(result.top_symbols)
    stored_symbol_ids = {row[1] for row in fake_connection.shortlists}
    assert stored_symbol_ids == {101, 202}
    assert fake_connection.security_metadata["AAA"]["name"] == "Alpha"
    assert fake_connection.security_metadata["AAA"]["sector"] == "Technology"
    assert fake_connection.security_metadata["BBB"]["name"] == "Beta"
    assert fake_connection.security_metadata["BBB"]["sector"] == "Healthcare"
    diversified = result.diversified_df.to_dict(orient="records")
    rows_by_symbol = {row["ticker"]: row for row in diversified}
    id_to_symbol = {identifier: symbol for symbol, identifier in fake_connection.securities.items()}
    for run_date, symbol_id, liquidity_score, price, average_volume, _created_at in fake_connection.shortlists:
        assert run_date == result.run_date
        symbol = id_to_symbol[symbol_id]
        shortlist_row = rows_by_symbol[symbol]
        expected_price = shortlist_row.get("price")
        expected_avg = (
            shortlist_row.get("average_volume")
            if shortlist_row.get("average_volume") is not None
            else shortlist_row.get("avg_volume_3m")
        )
        if expected_price is not None:
            assert price is not None
            assert math.isclose(price, float(expected_price), rel_tol=1e-6)
        if expected_avg is not None:
            assert average_volume is not None
            assert math.isclose(average_volume, float(expected_avg), rel_tol=1e-6)
        if expected_price is not None and expected_avg is not None:
            expected_score = float(expected_price) * float(expected_avg)
        else:
            fallback = shortlist_row.get("liquidity_score") or shortlist_row.get("score") or 0.0
            expected_score = float(fallback)
        assert math.isclose(liquidity_score, expected_score, rel_tol=1e-6)
