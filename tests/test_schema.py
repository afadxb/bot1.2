from steadyalpha.storage.migrations import SCHEMA_VERSION, apply_migrations


def test_migrations_are_idempotent(fake_connection):
    apply_migrations(fake_connection)
    apply_migrations(fake_connection)

    assert "run_summary" in fake_connection.tables
    assert "candidates" in fake_connection.tables

    candidate_columns = fake_connection.tables["candidates"]
    for column in {
        "run_id",
        "symbol",
        "gap_pct",
        "pre_mkt_vol",
        "catalyst_flag",
        "pm_high",
        "pm_low",
        "prev_high",
        "prev_low",
        "pm_vwap",
        "tags",
        "created_at",
    }:
        assert column in candidate_columns

    summary_columns = fake_connection.tables["run_summary"]
    assert {"run_id", "started_at", "finished_at", "notes"}.issubset(summary_columns)

    assert fake_connection.schema_versions == {SCHEMA_VERSION}
    assert fake_connection.indexes["candidates"] == {"idx_symbol"}
