# SteadyAlpha Screener Bot

The SteadyAlpha Screener Bot ingests Finviz Elite pre-market exports, normalises
and scores equities with the existing Top-N ranking pipeline, and publishes
results to a shared MySQL database. Optional CSV and JSON artifacts are still
produced for ad-hoc review, but downstream automation should consume the MySQL
`candidates` table.

## Features

- Finviz Elite ingest with cached CSV reuse.
- Deterministic filters, feature engineering, scoring, and sector diversification
  inherited from the original pre-market bot.
- APScheduler driven pre-market cadence (default 08:30–09:15 ET).
- MySQL persistence with idempotent migrations and a pooled connection.
- Optional CSV/JSON artifacts under `data/watchlists/<YYYY-MM-DD>` for manual
  inspection.
- Pushover summary notifications after each run.

## Configuration

All configuration is sourced from `.env` using `pydantic.BaseSettings`. Copy
`.env.example` and populate the required fields:

```dotenv
# Core runtime
TZ="America/New_York"
FINVIZ_EXPORT_URL="https://elite.finviz.com/export.ashx?..."
TOP_N=40
MIN_GAP_PCT=2.0
MIN_PM_VOL=200000

# MySQL target
DB_HOST=localhost
DB_PORT=3306
DB_USER=screener
DB_PASS=secret
DB_NAME=steadyalpha

# Scheduler window
JOB_SCREENER_PM_START="08:30"
JOB_SCREENER_PM_END="09:15"
JOB_SCREENER_PM_EVERY_MIN=5

# Optional notifications
PUSHOVER_USER_KEY="..."
PUSHOVER_API_TOKEN="..."
```

`config/strategy.yaml` retains all scoring weights, penalties, filters, and
sector caps. Adjust the YAML to tune the screener behaviour; the values are
combined with the `.env` settings at runtime.

## Running locally

Install dependencies and execute a single scan:

```bash
pip install -e .[dev]
python -m premarket  # legacy entry point that now delegates to the screener
```

To run continuously, launch the scheduler:

```bash
python -m steadyalpha.app
```

The scheduler respects the configured window (`JOB_SCREENER_PM_START`–
`JOB_SCREENER_PM_END`) and interval (`JOB_SCREENER_PM_EVERY_MIN`). Jobs outside
the window are skipped.

## MySQL schema

Migrations ensure the following tables exist:

- `schema_version(version INT PRIMARY KEY, applied_at DATETIME)`
- `run_summary(run_id VARCHAR(26) PRIMARY KEY, started_at DATETIME, finished_at DATETIME, notes VARCHAR(255))`
- `candidates(run_id VARCHAR(26), symbol VARCHAR(16), gap_pct DECIMAL(8,3), pre_mkt_vol INT,
  catalyst_flag TINYINT, pm_high DECIMAL(16,6), pm_low DECIMAL(16,6), prev_high DECIMAL(16,6),
  prev_low DECIMAL(16,6), pm_vwap DECIMAL(16,6), tags VARCHAR(255), created_at DATETIME,
  PRIMARY KEY(run_id, symbol), KEY idx_symbol(symbol))`

Each screener run inserts a ULID-based `run_id`, all ranked candidates, and the
run summary metadata. Re-running the screener for the same `run_id` is safe due
to `ON DUPLICATE KEY UPDATE` semantics.

## Notifications

Set `PUSHOVER_USER_KEY` and `PUSHOVER_API_TOKEN` to receive a succinct Pushover
message after each run, e.g. `Screener TopN: AAPL, MSFT, NVDA … (N=40)`.

## Testing

```bash
pytest -q
```

Tests mock the MySQL layer to verify migrations and the end-to-end screener
service without requiring a live database.
