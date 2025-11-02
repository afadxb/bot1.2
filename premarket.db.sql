BEGIN TRANSACTION;
CREATE TABLE IF NOT EXISTS "full_watchlist" (
	"run_date"	TEXT NOT NULL,
	"generated_at"	TEXT NOT NULL,
	"symbol"	TEXT,
	"company"	TEXT,
	"sector"	TEXT,
	"industry"	TEXT,
	"exchange"	TEXT,
	"market_cap"	TEXT,
	"pe"	TEXT,
	"price"	TEXT,
	"change_pct"	TEXT,
	"gap_pct"	TEXT,
	"volume"	TEXT,
	"avg_volume_3m"	TEXT,
	"rel_volume"	TEXT,
	"float_shares"	TEXT,
	"short_float_pct"	TEXT,
	"after_hours_change_pct"	TEXT,
	"week52_range"	TEXT,
	"week52_pos"	TEXT,
	"earnings_date"	TEXT,
	"analyst_recom"	TEXT,
	"features_json"	TEXT,
	"score"	TEXT,
	"tier"	TEXT,
	"tags_json"	TEXT,
	"rejection_reasons_json"	TEXT,
	"insider_transactions"	TEXT,
	"institutional_transactions"	TEXT
);
CREATE TABLE IF NOT EXISTS "top_n" (
	"run_date"	TEXT NOT NULL,
	"generated_at"	TEXT NOT NULL,
	"rank"	INTEGER,
	"symbol"	TEXT,
	"score"	TEXT
);
CREATE TABLE IF NOT EXISTS "watchlist" (
	"run_date"	TEXT NOT NULL,
	"generated_at"	TEXT NOT NULL,
	"rank"	INTEGER,
	"symbol"	TEXT,
	"score"	TEXT,
	"tier"	TEXT,
	"gap_pct"	TEXT,
	"rel_volume"	TEXT,
	"tags_json"	TEXT,
	"why"	TEXT,
	"top_feature1"	TEXT,
	"top_feature2"	TEXT,
	"top_feature3"	TEXT,
	"top_feature4"	TEXT,
	"top_feature5"	TEXT
);
CREATE TABLE IF NOT EXISTS "run_summary" (
	"run_date"	TEXT NOT NULL,
	"generated_at"	TEXT NOT NULL,
	"summary_date"	TEXT,
	"filters_json"	TEXT,
	"timings_json"	TEXT,
	"notes_json"	TEXT,
	"row_counts_json"	TEXT,
	"tiers_json"	TEXT,
	"env_overrides_json"	TEXT,
	"weights_version"	TEXT,
	"csv_hash"	TEXT,
	"sector_cap_applied"	INTEGER,
	"used_cached_csv"	INTEGER,
	"week52_warning_count"	INTEGER
);
COMMIT;
