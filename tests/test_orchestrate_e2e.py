import json
import sqlite3
from pathlib import Path

from datetime import date

import pandas as pd

from premarket import orchestrate


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
            "Gap": "2%",
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
        {
            "Ticker": "CCC",
            "Company": "Gamma",
            "Sector": "Technology",
            "Industry": "Hardware",
            "Exchange": "OTC",
            "Country": "USA",
            "Market Cap": "500,000,000",
            "P/E": "15",
            "Price": "10",
            "Change": "1%",
            "Gap": "0.5%",
            "Volume": "300000",
            "Average Volume (3m)": "1500000",
            "Relative Volume": "1.7",
            "Float": "5000000",
            "Float %": "55%",
            "Short Float": "5%",
            "After-Hours Change": "0.1%",
            "52-Week Range": "5 - 12",
            "Earnings Date": "2099-01-01",
            "Analyst Recom.": "Hold",
            "Insider Transactions": "-1%",
            "Institutional Transactions": "-2%",
            "Previous Close": "9.5",
        },
    ]
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)


def _fetch_rows(conn: sqlite3.Connection, query: str) -> list[dict[str, object]]:
    cursor = conn.execute(query)
    columns = [desc[0] for desc in cursor.description]
    return [dict(zip(columns, row)) for row in cursor.fetchall()]


def test_orchestrate_end_to_end(tmp_path, monkeypatch):
    csv_path = tmp_path / "finviz.csv"
    _sample_csv(csv_path)

    monkeypatch.setenv("FINVIZ_EXPORT_URL", "https://example.com/export")
    monkeypatch.setenv("CACHE_TTL_MIN", "1440")

    monkeypatch.setattr(orchestrate.loader_finviz, "download_csv", lambda url, out_path, use_cache: csv_path)

    run_date = date(2024, 1, 2)
    out_base = tmp_path / "out"
    params = orchestrate.RunParams(
        config_path=Path("config/strategy.yaml"),
        output_base_dir=out_base,
        top_n=2,
        use_cache=True,
        news_override=False,
        log_file=tmp_path / "run.log",
        run_date=run_date,
        timezone="America/New_York",
        env_overrides=["PREMARKET_TOP_N"],
    )

    code = orchestrate.run(params)

    assert code == 0
    out_dir = out_base / run_date.isoformat()
    rejection_path = csv_path.with_name("finviz_reject.csv")
    db_path = out_dir / "watchlist.db"
    assert db_path.exists()
    assert rejection_path.exists()

    conn = sqlite3.connect(db_path)
    watchlist_rows = _fetch_rows(conn, "SELECT * FROM watchlist")
    assert watchlist_rows
    assert "Why" in watchlist_rows[0]
    assert "TopFeature5" in watchlist_rows[0]
    assert "AIConfidence" in watchlist_rows[0]
    assert float(watchlist_rows[0]["AIConfidence"]) >= 0.0

    topn_rows = _fetch_rows(conn, "SELECT * FROM top_rankings")
    assert len(topn_rows) == 2
    assert "ai_confidence" in topn_rows[0]

    metadata_rows = _fetch_rows(conn, "SELECT * FROM metadata")
    assert int(metadata_rows[0]["top_n"]) == 2
    insights = json.loads(metadata_rows[0]["insights"])
    assert "news" in insights
    assert "ai_confidence" in insights
    assert "sector_focus" in insights
    schemas = json.loads(metadata_rows[0]["table_schemas"])
    assert "watchlist" in schemas
    assert "AIConfidence" in schemas["watchlist"]
    assert "ai_confidence" in schemas["top_rankings"]

    run_summary_rows = _fetch_rows(conn, "SELECT * FROM run_summary")
    run_summary = json.loads(run_summary_rows[0]["payload"])
    assert run_summary["row_counts"]["topN"] == 2
    assert "csv_hash" in run_summary
    assert run_summary["env_overrides_used"] == sorted(params.env_overrides)
    assert "news_signal" in run_summary

    full_watchlist_rows = _fetch_rows(conn, "SELECT * FROM full_watchlist")
    assert full_watchlist_rows
    assert "ai_confidence" in full_watchlist_rows[0]

    rejections_rows = _fetch_rows(conn, "SELECT * FROM rejections")
    if rejections_rows:
        assert "rejection_reasons" in rejections_rows[0]
    conn.close()

    rejected_df = pd.read_csv(rejection_path)
    assert "ticker" in rejected_df.columns
    reason_row = rejected_df.loc[rejected_df["ticker"] == "CCC"]
    assert not reason_row.empty
    reason = str(reason_row.iloc[0]["rejection_reasons"])
    parsed_reasons = [part.strip() for part in reason.split("|") if part.strip()]
    assert "exchange_excluded" in parsed_reasons


def test_run_emits_empty_outputs_when_download_fails(tmp_path, monkeypatch):
    monkeypatch.setenv("FINVIZ_EXPORT_URL", "https://example.com/export")
    out_base = tmp_path / "out"
    run_date = date(2024, 1, 2)

    def boom(*_, **__):
        raise RuntimeError("network disabled")

    monkeypatch.setattr(orchestrate.loader_finviz, "download_csv", boom)

    params = orchestrate.RunParams(
        config_path=Path("config/strategy.yaml"),
        output_base_dir=out_base,
        use_cache=True,
        news_override=False,
        run_date=run_date,
        timezone="America/New_York",
    )

    exit_code = orchestrate.run(params)

    assert exit_code == 0

    out_dir = out_base / run_date.isoformat()
    db_path = out_dir / "watchlist.db"
    assert db_path.exists()

    conn = sqlite3.connect(db_path)

    watchlist_rows = _fetch_rows(conn, "SELECT * FROM watchlist")
    assert watchlist_rows == []

    topn_rows = _fetch_rows(conn, "SELECT * FROM top_rankings")
    assert topn_rows == []

    summary_rows = _fetch_rows(conn, "SELECT * FROM run_summary")
    assert len(summary_rows) == 1
    summary = json.loads(summary_rows[0]["payload"])
    metadata_rows = _fetch_rows(conn, "SELECT * FROM metadata")
    schemas = json.loads(metadata_rows[0]["table_schemas"])
    assert schemas["watchlist"][-1] == "TopFeature5"
    conn.close()

    assert summary["row_counts"] == {"raw": 0, "qualified": 0, "rejected": 0, "topN": 0}
    assert "download_failed_no_cache" in summary["notes"]
    assert summary["used_cached_csv"] is False
