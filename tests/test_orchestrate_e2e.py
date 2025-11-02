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


def test_orchestrate_end_to_end(tmp_path, monkeypatch):
    db_path = Path("premarket.db")
    if db_path.exists():
        db_path.unlink()

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
    assert (out_dir / "full_watchlist.json").exists()
    assert (out_dir / "topN.json").exists()
    assert (out_dir / "watchlist.csv").exists()
    assert (out_dir / "run_summary.json").exists()
    assert rejection_path.exists()

    topn = json.loads((out_dir / "topN.json").read_text())
    assert topn["top_n"] == 2
    assert len(topn["symbols"]) == 2
    top_symbols_list = topn["symbols"]

    watchlist_df = pd.read_csv(out_dir / "watchlist.csv")
    assert "Why" in watchlist_df.columns
    assert "TopFeature5" in watchlist_df.columns

    run_summary = json.loads((out_dir / "run_summary.json").read_text())
    assert run_summary["row_counts"]["topN"] == 2
    assert "csv_hash" in run_summary
    assert run_summary["env_overrides_used"] == sorted(params.env_overrides)

    rejected_df = pd.read_csv(rejection_path)
    assert "ticker" in rejected_df.columns
    reason_row = rejected_df.loc[rejected_df["ticker"] == "CCC"]
    assert not reason_row.empty
    reason = str(reason_row.iloc[0]["rejection_reasons"])
    parsed_reasons = [part.strip() for part in reason.split("|") if part.strip()]
    assert "exchange_excluded" in parsed_reasons

    assert db_path.exists()
    with sqlite3.connect(db_path) as conn:
        full_rows = conn.execute(
            "SELECT symbol, score FROM full_watchlist WHERE run_date = ?",
            (run_date.isoformat(),),
        ).fetchall()
        assert full_rows
        first_symbol, first_score = full_rows[0]
        assert first_symbol in {"AAA", "BBB"}
        assert first_score is not None

        top_rows = conn.execute(
            "SELECT rank, symbol, score FROM top_n WHERE run_date = ? ORDER BY rank",
            (run_date.isoformat(),),
        ).fetchall()
        assert len(top_rows) == 2
        assert [row[0] for row in top_rows] == [1, 2]
        assert {row[1] for row in top_rows} == set(top_symbols_list)

        watch_rows = conn.execute(
            """
            SELECT rank, symbol, why, tags_json
            FROM watchlist
            WHERE run_date = ?
            ORDER BY rank
            """,
            (run_date.isoformat(),),
        ).fetchall()
        assert len(watch_rows) == 2
        assert watch_rows[0][0] == 1
        assert isinstance(watch_rows[0][2], str)
        tags = json.loads(watch_rows[0][3]) if watch_rows[0][3] else []
        assert isinstance(tags, list)

        summary_row = conn.execute(
            """
            SELECT row_counts_json, used_cached_csv
            FROM run_summary
            WHERE run_date = ?
            """,
            (run_date.isoformat(),),
        ).fetchone()
        assert summary_row is not None
        summary_payload = json.loads(summary_row[0])
        assert summary_payload["topN"] == 2
        assert summary_row[1] in (0, 1)


def test_run_emits_empty_outputs_when_download_fails(tmp_path, monkeypatch):
    db_path = Path("premarket.db")
    if db_path.exists():
        db_path.unlink()

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
    assert (out_dir / "full_watchlist.json").exists()
    assert (out_dir / "topN.json").exists()
    assert (out_dir / "watchlist.csv").exists()
    assert (out_dir / "run_summary.json").exists()

    topn = json.loads((out_dir / "topN.json").read_text())
    assert topn["symbols"] == []

    summary = json.loads((out_dir / "run_summary.json").read_text())
    assert summary["row_counts"] == {"raw": 0, "qualified": 0, "rejected": 0, "topN": 0}
    assert "download_failed_no_cache" in summary["notes"]
    assert summary["used_cached_csv"] is False

    assert db_path.exists()
    with sqlite3.connect(db_path) as conn:
        top_rows = conn.execute(
            "SELECT rank FROM top_n WHERE run_date = ?",
            (run_date.isoformat(),),
        ).fetchall()
        assert top_rows == []

        summary_row = conn.execute(
            "SELECT row_counts_json FROM run_summary WHERE run_date = ?",
            (run_date.isoformat(),),
        ).fetchone()
        assert summary_row is not None
        payload = json.loads(summary_row[0])
        assert payload["topN"] == 0
