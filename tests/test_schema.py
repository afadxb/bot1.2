from steadyalpha.storage.migrations import SCHEMA_VERSION, apply_migrations


def test_migrations_are_idempotent(fake_connection):
    apply_migrations(fake_connection)
    apply_migrations(fake_connection)

    assert "run_summary" in fake_connection.tables
    assert "shortlists" in fake_connection.tables

    shortlist_columns = fake_connection.tables["shortlists"]
    for column in {
        "id",
        "run_date",
        "symbol_id",
        "liquidity_score",
        "price",
        "average_volume",
        "created_at",
    }:
        assert column in shortlist_columns

    summary_columns = fake_connection.tables["run_summary"]
    assert {"run_id", "started_at", "finished_at", "notes"}.issubset(summary_columns)

    assert fake_connection.schema_versions == {SCHEMA_VERSION}
    assert fake_connection.indexes["shortlists"] == {"idx_run_date", "uq_shortlist_symbol_date"}
