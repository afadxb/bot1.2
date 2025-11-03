import re
import sys
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest

from steadyalpha.settings import get_settings


class FakeCursor:
    def __init__(self, connection: "FakeConnection") -> None:
        self.connection = connection
        self._results: list[tuple[Any, ...]] = []

    def __enter__(self) -> "FakeCursor":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self._results = []

    def execute(self, query: str, params: Iterable[Any] | None = None) -> None:
        params = tuple(params or ())
        normalized = query.strip()
        self._results = []
        if normalized.upper().startswith("CREATE TABLE"):
            self._handle_create_table(normalized)
        elif normalized.upper().startswith("ALTER TABLE"):
            self._handle_alter_table(normalized)
        elif normalized.upper().startswith("SHOW COLUMNS"):
            table = re.search(r"FROM `([^`]+)`", normalized).group(1)
            columns = sorted(self.connection.tables.get(table, set()))
            self._results = [(name,) for name in columns]
        elif normalized.upper().startswith("SHOW INDEX"):
            table = re.search(r"FROM `([^`]+)`", normalized).group(1)
            indexes = sorted(self.connection.indexes.get(table, set()))
            self._results = [(None, None, name) for name in indexes]
        elif normalized.upper().startswith("CREATE INDEX"):
            match = re.search(r"CREATE INDEX `([^`]+)` ON `([^`]+)`", normalized, re.IGNORECASE)
            if match:
                name, table = match.groups()
                self.connection.indexes.setdefault(table, set()).add(name)
        elif normalized.upper().startswith("SELECT VERSION"):
            version = params[0] if params else None
            if version in self.connection.schema_versions:
                self._results = [(version,)]
        elif normalized.upper().startswith("SELECT SYMBOL, ID FROM SECURITIES"):
            symbols = params or ()
            results = []
            for symbol in symbols:
                identifier = self.connection.securities.get(symbol)
                if identifier is not None:
                    results.append((symbol, identifier))
            self._results = results
        elif normalized.upper().startswith("INSERT INTO SECURITIES"):
            self._handle_insert_securities([params])
        elif normalized.upper().startswith("INSERT INTO SCHEMA_VERSION"):
            if params:
                self.connection.schema_versions.add(params[0])
        elif normalized.upper().startswith("INSERT INTO RUN_SUMMARY"):
            self.connection.run_summary.append(params)
        else:
            self.connection.statements.append((normalized, params))

    def executemany(self, query: str, seq: Iterable[Iterable[Any]]) -> None:
        normalized = query.strip()
        if normalized.upper().startswith("INSERT INTO CANDIDATES"):
            for params in seq:
                self.connection.candidates.append(tuple(params))
        elif normalized.upper().startswith("INSERT INTO SHORTLISTS"):
            for params in seq:
                self.connection.shortlists.append(tuple(params))
        elif normalized.upper().startswith("INSERT INTO SECURITIES"):
            rows = [tuple(params) for params in seq]
            self._handle_insert_securities(rows)
        else:
            for params in seq:
                self.connection.statements.append((normalized, tuple(params)))

    def fetchall(self) -> list[tuple[Any, ...]]:
        return list(self._results)

    def fetchone(self) -> tuple[Any, ...] | None:
        return self._results[0] if self._results else None

    def _handle_create_table(self, query: str) -> None:
        match = re.search(r"CREATE TABLE IF NOT EXISTS `([^`]+)`", query, re.IGNORECASE)
        if not match:
            return
        table = match.group(1)
        columns = set(self.connection.tables.get(table, set()))
        body = query.split("(", 1)[1].rsplit(")", 1)[0]
        for line in body.splitlines():
            line = line.strip().strip(",")
            col_match = re.match(r"`([^`]+)`", line)
            if col_match:
                columns.add(col_match.group(1))
                continue
            unique_match = re.match(r"UNIQUE KEY `([^`]+)`", line, re.IGNORECASE)
            if unique_match:
                self.connection.indexes.setdefault(table, set()).add(unique_match.group(1))
                continue
            key_match = re.match(r"KEY `([^`]+)`", line, re.IGNORECASE)
            if key_match:
                self.connection.indexes.setdefault(table, set()).add(key_match.group(1))
        self.connection.tables[table] = columns

    def _handle_alter_table(self, query: str) -> None:
        match = re.search(r"ALTER TABLE `([^`]+)` ADD COLUMN ([^ ]+)", query, re.IGNORECASE)
        if not match:
            return
        table = match.group(1)
        column = match.group(2).strip("`")
        self.connection.tables.setdefault(table, set()).add(column)

    def _handle_insert_securities(self, rows: Iterable[Iterable[Any]]) -> None:
        next_id = max(self.connection.securities.values(), default=0)
        for params in rows:
            values = tuple(params)
            if not values:
                continue
            symbol = str(values[0])
            name = values[1] if len(values) > 1 else None
            sector = values[2] if len(values) > 2 else None
            if symbol not in self.connection.securities:
                next_id += 1
                self.connection.securities[symbol] = next_id
            self.connection.security_metadata[symbol] = {
                "name": name,
                "sector": sector,
            }


class FakeConnection:
    def __init__(self) -> None:
        self.tables: dict[str, set[str]] = {}
        self.indexes: dict[str, set[str]] = {}
        self.schema_versions: set[int] = set()
        self.run_summary: list[tuple[Any, ...]] = []
        self.candidates: list[tuple[Any, ...]] = []
        self.shortlists: list[tuple[Any, ...]] = []
        self.statements: list[tuple[str, tuple[Any, ...]]] = []
        self.commits: int = 0
        self.closed: bool = False
        self.securities: dict[str, int] = {}
        self.security_metadata: dict[str, dict[str, Any]] = {}

    def cursor(self) -> FakeCursor:
        return FakeCursor(self)

    def commit(self) -> None:
        self.commits += 1

    def close(self) -> None:
        self.closed = True


@pytest.fixture(autouse=True)
def clear_settings_cache():
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture
def fake_connection() -> FakeConnection:
    return FakeConnection()
