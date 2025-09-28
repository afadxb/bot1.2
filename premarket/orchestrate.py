"""Runtime orchestration for the premarket workflow."""

from __future__ import annotations

import hashlib
import logging
import time
from dataclasses import dataclass, field
from datetime import date, datetime, time as dtime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import yaml
from dateutil import tz
from pydantic import BaseModel, Field

from . import features, filters, loader_finviz, news_ai, normalize, persist, ranker, utils
from .news_probe import probe as news_probe

LOGGER = logging.getLogger(__name__)

_FEATURE_LABELS = {
    "relvol": "RelVol",
    "gap": "Gap",
    "avgvol": "AvgVol",
    "float_band": "FloatBand",
    "short_float": "ShortFloat",
    "after_hours": "AfterHours",
    "change": "Change",
    "w52pos": "52Wpos",
    "news_fresh": "News",
    "analyst": "Analyst",
    "insider_inst": "Ins/Inst",
}

WATCHLIST_COLUMNS = [
    "rank",
    "symbol",
    "score",
    "AIConfidence",
    "tier",
    "gap_pct",
    "rel_volume",
    "tags",
    "Why",
    "TopFeature1",
    "TopFeature2",
    "TopFeature3",
    "TopFeature4",
    "TopFeature5",
]

TOP_RANKINGS_COLUMNS = ["rank", "symbol", "score", "ai_confidence"]

FULL_WATCHLIST_COLUMNS = [
    "symbol",
    "company",
    "sector",
    "industry",
    "exchange",
    "market_cap",
    "pe",
    "price",
    "change_pct",
    "gap_pct",
    "volume",
    "avg_volume_3m",
    "rel_volume",
    "float_shares",
    "short_float_pct",
    "after_hours_change_pct",
    "week52_range",
    "week52_pos",
    "earnings_date",
    "analyst_recom",
    "insider_transactions",
    "institutional_transactions",
    "features",
    "score",
    "ai_confidence",
    "tier",
    "tags",
    "rejection_reasons",
    "generated_at",
]

def _table_schemas_metadata() -> dict[str, list[str]]:
    """Return a mapping of table names to their column order for metadata."""

    return {
        "watchlist": WATCHLIST_COLUMNS,
        "top_rankings": TOP_RANKINGS_COLUMNS,
        "full_watchlist": FULL_WATCHLIST_COLUMNS,
    }


class WeightsModel(BaseModel):
    relvol: float
    gap: float
    avgvol: float
    float_band: float
    short_float: float
    after_hours: float
    change: float
    w52pos: float
    news_fresh: float
    analyst: float
    insider_inst: float


class PenaltiesModel(BaseModel):
    earnings_near: float
    pe_outlier: float


class CapsModel(BaseModel):
    max_single_negative: float


class PremarketModel(BaseModel):
    price_min: float
    price_max: float
    avg_vol_min: int
    rel_vol_min: float
    float_min: int
    earnings_exclude_window_days: int
    max_per_sector: float
    top_n: int
    exclude_exchanges: list[str] = Field(default_factory=list)
    exclude_countries: list[str] = Field(default_factory=list)
    weights: WeightsModel
    penalties: PenaltiesModel
    caps: CapsModel
    weights_version: Optional[str] = None


class NewsModel(BaseModel):
    enabled: bool = False
    freshness_hours: int = 24
    finviz_url: Optional[str] = None
    finnhub_token: Optional[str] = None
    finnhub_days: int = 3


class StrategyConfig(BaseModel):
    premarket: PremarketModel
    news: NewsModel


@dataclass
class RunParams:
    """Runtime settings derived from environment variables."""

    config_path: Path
    run_date: date
    output_base_dir: Path = field(default_factory=lambda: Path("data/watchlists"))
    top_n: Optional[int] = None
    use_cache: bool = True
    news_override: Optional[bool] = None
    log_file: Optional[Path] = None
    timezone: str = utils.DEFAULT_TZ_NAME
    fail_on_empty: bool = False
    max_per_sector: Optional[float] = None
    env_overrides: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if isinstance(self.config_path, str):
            self.config_path = Path(self.config_path)
        if isinstance(self.output_base_dir, str):
            self.output_base_dir = Path(self.output_base_dir)
        if isinstance(self.log_file, str):
            self.log_file = Path(self.log_file)
        if isinstance(self.run_date, str):
            self.run_date = date.fromisoformat(self.run_date)
        if self.top_n is not None and self.top_n <= 0:
            raise ValueError("top_n must be a positive integer")
        if self.max_per_sector is not None and not (0 < self.max_per_sector <= 1):
            raise ValueError("max_per_sector must be between 0 and 1")
        if not self.timezone or tz.gettz(self.timezone) is None:
            self.timezone = utils.DEFAULT_TZ_NAME
        self.env_overrides = sorted(set(self.env_overrides))

    def resolved_output_dir(self) -> Path:
        path = self.output_base_dir / self.run_date.isoformat()
        utils.ensure_directory(path)
        return path

    def resolved_log_file(self) -> Path:
        if self.log_file is not None:
            return self.log_file
        return Path("logs") / f"premarket_{self.run_date.isoformat()}.log"

def _load_config(path: Path) -> StrategyConfig:
    data = yaml.safe_load(path.read_text())
    return StrategyConfig.model_validate(data)


def _build_filter_config(cfg: PremarketModel) -> filters.FilterConfig:
    return filters.FilterConfig(
        price_min=cfg.price_min,
        price_max=cfg.price_max,
        avg_vol_min=cfg.avg_vol_min,
        rel_vol_min=cfg.rel_vol_min,
        float_min=cfg.float_min,
        earnings_exclude_window_days=cfg.earnings_exclude_window_days,
        exclude_exchanges=cfg.exclude_exchanges,
        exclude_countries=cfg.exclude_countries,
    )


def _build_ranker_config(cfg: PremarketModel) -> ranker.RankerConfig:
    return ranker.RankerConfig(
        weights=ranker.RankerWeights(**cfg.weights.model_dump()),
        penalties=ranker.RankerPenalties(**cfg.penalties.model_dump()),
        caps=ranker.RankerCaps(**cfg.caps.model_dump()),
        earnings_window_days=cfg.earnings_exclude_window_days,
    )


def _determine_output_dir(base_dir: Path, today: str) -> Path:
    path = base_dir / today
    utils.ensure_directory(path)
    return path


def _determine_log_path(log_file: Optional[Path], today: str) -> Path:
    if log_file is not None:
        return log_file
    return Path("logs") / f"premarket_{today}.log"


def _news_scores(symbols: list[str], news_cfg: NewsModel) -> Dict[str, float]:
    if not news_cfg.enabled or not symbols:
        return {symbol: 0.0 for symbol in symbols}
    raw = news_probe(symbols, news_cfg)
    horizon = news_cfg.freshness_hours or 24
    return news_ai.score_batch(raw, horizon_hours=horizon)


def _tags_for_row(row: pd.Series) -> list[str]:
    tags: list[str] = []
    float_shares = row.get("float_shares")
    if float_shares is not None and float_shares < 20_000_000:
        tags.append("LOW_FLOAT")
    gap_pct = row.get("gap_pct")
    if gap_pct is not None and gap_pct > 20:
        tags.append("EXTREME_GAP")
    earnings_date = row.get("earnings_date")
    if hasattr(earnings_date, "date"):
        if abs((earnings_date.date() - utils.now_eastern().date()).days) <= 1:
            tags.append("EARNINGS_TODAY")
    if row.get("f_52w_pos", 0.0) >= 0.80:
        tags.append("FIFTY_TWO_WEEK_BREAKOUT")
    return tags


def _build_feature_dict(row: pd.Series) -> Dict[str, float]:
    features_map: Dict[str, float] = {}
    for col in row.index:
        if not col.startswith("f_"):
            continue
        value = row[col]
        if value is None or (isinstance(value, float) and np.isnan(value)):
            features_map[col.replace("f_", "")] = 0.0
        else:
            features_map[col.replace("f_", "")] = float(value)
    return features_map


def _feature_contributions(row: pd.Series, weights: ranker.RankerWeights) -> list[tuple[str, float]]:
    contributions: list[tuple[str, float]] = []
    for key, label in _FEATURE_LABELS.items():
        column = f"f_{key}"
        if column not in row:
            continue
        value = row.get(column)
        if value is None or (isinstance(value, float) and np.isnan(value)):
            contribution = 0.0
        else:
            weight = getattr(weights, key)
            contribution = float(value) * float(weight)
        contributions.append((label, contribution))
    contributions.sort(key=lambda item: item[1], reverse=True)
    return contributions


def _summarize_ai_confidence(df: pd.DataFrame) -> dict[str, object]:
    if df.empty or "ai_confidence" not in df.columns:
        return {"average": 0.0, "leaders": [], "count": 0}

    confidence_series = pd.to_numeric(df["ai_confidence"], errors="coerce")
    confidence_series = confidence_series.dropna()
    values = list(zip(confidence_series.index, confidence_series.to_list()))
    filtered = [(idx, float(val)) for idx, val in values if not pd.isna(val)]
    if not filtered:
        return {"average": 0.0, "leaders": [], "count": 0}

    total = sum(val for _, val in filtered)
    count = len(filtered)
    average = total / count if count else 0.0
    filtered.sort(key=lambda item: item[1], reverse=True)

    ticker_series = df.get("ticker") if "ticker" in df.columns else None
    ticker_lookup = {}
    if ticker_series is not None:
        ticker_lookup = dict(zip(ticker_series.index, ticker_series.to_list()))

    leaders: list[dict[str, object]] = []
    for idx, score in filtered[:3]:
        symbol = ticker_lookup.get(idx)
        if not symbol:
            continue
        leaders.append({"symbol": str(symbol), "confidence": round(float(score), 2)})

    return {
        "average": round(float(average), 2),
        "leaders": leaders,
        "count": int(count),
    }


def _sector_distribution(df: pd.DataFrame) -> dict[str, int]:
    if df.empty or "sector" not in df.columns:
        return {}
    sector_series = df.get("sector")
    if sector_series is None:
        return {}
    cleaned: list[str] = []
    for value in sector_series.to_list():
        if value is None:
            continue
        text = str(value).strip()
        if text:
            cleaned.append(text)
    if not cleaned:
        return {}
    counts: dict[str, int] = {}
    for item in cleaned:
        counts[item] = counts.get(item, 0) + 1
    sorted_counts = sorted(counts.items(), key=lambda entry: (-entry[1], entry[0]))
    top_five = sorted_counts[:5]
    return {sector: int(count) for sector, count in top_five}


def _build_metadata_insights(
    df: pd.DataFrame, news_summary: dict[str, object]
) -> dict[str, object]:
    return {
        "news": news_summary,
        "ai_confidence": _summarize_ai_confidence(df),
        "sector_focus": _sector_distribution(df),
    }


def _timezone_label(tz_name: str, run_day: date) -> str:
    tzinfo = tz.gettz(tz_name)
    if tzinfo is None:
        return tz_name
    tz_dt = datetime.combine(run_day, dtime.min, tzinfo)
    label = tz_dt.tzname() or tz_name
    if label in {"EST", "EDT"}:
        return "ET"
    return label


def _build_run_summary(
    date_str: str,
    cfg: StrategyConfig,
    timings: Dict[str, float],
    notes: list[str],
    row_counts: Dict[str, int],
    tier_counts: Dict[str, int],
    env_overrides: List[str],
    weights_version: Optional[str],
    csv_hash: str,
    used_cached_csv: bool,
    sector_cap_applied: bool,
    week52_warnings: int,
    news_signal: Optional[dict[str, object]] = None,
) -> Dict[str, object]:
    summary = {
        "date": date_str,
        "filters": cfg.premarket.model_dump(),
        "timings_sec": {k: round(v, 3) for k, v in timings.items()},
        "notes": notes,
        "row_counts": row_counts,
        "tiers": tier_counts,
        "env_overrides_used": sorted(env_overrides),
        "weights_version": weights_version or "default",
        "csv_hash": csv_hash,
        "sector_cap_applied": bool(sector_cap_applied),
        "used_cached_csv": bool(used_cached_csv),
        "week52_warning_count": int(week52_warnings),
    }
    if news_signal is not None:
        summary["news_signal"] = news_signal
    return summary


def _emit_empty_outputs(
    output_dir: Path,
    generated_at: str,
    requested_top_n: int,
    run_summary: Dict[str, object],
) -> None:
    empty_table = pd.DataFrame(columns=WATCHLIST_COLUMNS)
    empty_full_watchlist = pd.DataFrame(columns=FULL_WATCHLIST_COLUMNS)
    empty_rankings = pd.DataFrame(columns=TOP_RANKINGS_COLUMNS)
    empty_rejections = pd.DataFrame(columns=["ticker", "rejection_reasons"])
    empty_insights = _build_metadata_insights(
        pd.DataFrame(columns=["ai_confidence", "ticker", "sector"]),
        news_ai.summarize_scores({}),
    )
    persist.write_outputs(
        output_dir / "watchlist.db",
        watchlist=empty_table,
        top_rankings=empty_rankings,
        full_watchlist=empty_full_watchlist,
        run_summary=run_summary,
        metadata={
            "generated_at": generated_at,
            "top_n": requested_top_n,
            "insights": empty_insights,
            "table_schemas": _table_schemas_metadata(),
        },
        rejections=empty_rejections,
    )


def _format_tier_counts(tier_counts: Dict[str, int]) -> str:
    order = ["A", "B", "C"]
    parts = []
    for tier in order:
        value = tier_counts.get(tier, 0)
        parts.append(str(value) if value > 0 else "—")
    return "/".join(parts)


def run(params: RunParams) -> int:
    """Execute the full workflow."""

    utils.configure_timezone(params.timezone)
    today = params.run_date.isoformat()
    log_path = _determine_log_path(params.log_file, today)
    logger = utils.setup_logging(log_path)

    try:
        cfg = _load_config(Path(params.config_path))
    except Exception as exc:  # pragma: no cover - config errors
        logger.error("Failed to load config: %s", exc)
        return 1

    finviz_url = utils.env_str("FINVIZ_EXPORT_URL", "")
    if not finviz_url:
        logger.error("FINVIZ_EXPORT_URL is not set. Provide the export URL with auth token.")
        return 3

    top_n_value = params.top_n if params.top_n is not None else cfg.premarket.top_n
    max_per_sector = (
        params.max_per_sector if params.max_per_sector is not None else cfg.premarket.max_per_sector
    )

    news_enabled = params.news_override if params.news_override is not None else cfg.news.enabled
    news_cfg = cfg.news.copy()
    news_cfg.enabled = bool(news_enabled)
    weights_version = cfg.premarket.weights_version or "default"

    output_dir = _determine_output_dir(params.output_base_dir, today)
    raw_csv_path = Path("data/raw") / today / "finviz_elite.csv"

    timings: Dict[str, float] = {}
    notes: list[str] = []
    row_counts: Dict[str, int] = {"raw": 0, "qualified": 0, "rejected": 0, "topN": 0}

    start = time.perf_counter()
    try:
        csv_path = loader_finviz.download_csv(finviz_url, raw_csv_path, use_cache=params.use_cache)
    except RuntimeError:
        timings["download"] = time.perf_counter() - start
        logger.error("Failed to download CSV and no cache available.")
        notes.append("used_cached_csv: False")
        notes.append("download_failed_no_cache")
        generated_at = utils.timestamp_iso()
        run_summary = _build_run_summary(
            today,
            cfg,
            timings,
            notes,
            row_counts,
            {},
            params.env_overrides,
            weights_version,
            "",
            False,
            False,
            0,
        )
        _emit_empty_outputs(output_dir, generated_at, top_n_value, run_summary)
        tier_display = _format_tier_counts({})
        summary_line = (
            f"Date={today} {_timezone_label(params.timezone, params.run_date)} | "
            f"TopN=0 | A/B/C={tier_display} | SectorCap=False | "
            f"Cache=False | Out={output_dir}"
        )
        logger.info(summary_line)
        return 2 if params.fail_on_empty else 0
    timings["download"] = time.perf_counter() - start
    used_cached_csv = csv_path != raw_csv_path
    notes.append(f"used_cached_csv: {used_cached_csv}")

    try:
        csv_hash = hashlib.sha256(csv_path.read_bytes()).hexdigest()
    except OSError:
        csv_hash = ""
        logger.warning("Unable to hash CSV at %s", csv_path)

    start = time.perf_counter()
    df = loader_finviz.read_csv(csv_path)
    raw_rows = len(df)
    df = normalize.normalize_columns(df)
    df, week52_warnings = normalize.coerce_types(df)
    if week52_warnings:
        notes.append(f"week52_warnings: {week52_warnings}")
    timings["normalize"] = time.perf_counter() - start

    filter_cfg = _build_filter_config(cfg.premarket)
    qualified_df, rejected_df = filters.apply_hard_filters(df, filter_cfg)

    rejection_report_path = csv_path.with_name("finviz_reject.csv")
    utils.ensure_directory(rejection_report_path.parent)
    rejected_for_csv = rejected_df.copy()
    if "rejection_reasons" in rejected_for_csv.columns:
        rejected_for_csv["rejection_reasons"] = rejected_for_csv["rejection_reasons"].apply(
            lambda reasons: " | ".join(map(str, reasons))
            if isinstance(reasons, (list, tuple))
            else ("" if pd.isna(reasons) else str(reasons))
        )
    rejected_for_csv.to_csv(rejection_report_path, index=False)

    row_counts = {
        "raw": int(raw_rows) if "raw_rows" in locals() else 0,
        "qualified": int(len(qualified_df)) if "qualified_df" in locals() else 0,
        "rejected": int(len(rejected_df)) if "rejected_df" in locals() else 0,
        "topN": 0,
    }
    logger.info(row_counts)

    generated_at = utils.timestamp_iso()

    if qualified_df.empty:
        logger.warning("No candidates qualified after filters.")
        run_summary = _build_run_summary(
            today,
            cfg,
            timings,
            notes,
            row_counts,
            {},
            params.env_overrides,
            weights_version,
            csv_hash,
            used_cached_csv,
            False,
            week52_warnings,
            news_ai.summarize_scores({}),
        )
        _emit_empty_outputs(output_dir, generated_at, top_n_value, run_summary)
        empty_tiers: Dict[str, int] = {}
        tier_display = _format_tier_counts(empty_tiers)
        summary_line = (
            f"Date={today} {_timezone_label(params.timezone, params.run_date)} | "
            f"TopN=0 | A/B/C={tier_display} | SectorCap=False | "
            f"Cache={used_cached_csv} | Out={output_dir}"
        )
        logger.info(summary_line)
        return 2 if params.fail_on_empty else 0

    start = time.perf_counter()
    symbols = qualified_df.get("ticker", pd.Series(dtype=str)).fillna("").astype(str).tolist()
    news_scores = _news_scores(symbols, news_cfg)
    news_signal_summary = news_ai.summarize_scores(news_scores)
    if news_signal_summary["average"]:
        notes.append(f"news_signal_avg: {news_signal_summary['average']}")
    qualified_df["news_fresh_score"] = [news_scores.get(sym, 0.0) for sym in symbols]

    featured_df = features.build_features(qualified_df, cfg)

    rank_cfg = _build_ranker_config(cfg.premarket)
    scores = ranker.compute_score(featured_df, rank_cfg)
    featured_df["score"] = scores
    ai_confidence = scores.clip(lower=0.0, upper=1.0) * 100
    featured_df["ai_confidence"] = ai_confidence
    featured_df["tier"] = ranker.assign_tiers(scores)
    timings["score"] = time.perf_counter() - start

    featured_df.sort_values(
        by=["score", "turnover_dollar", "ticker"], ascending=[False, False, True], inplace=True
    )

    diversified_df, sector_trimmed = ranker.apply_sector_diversity(
        featured_df, top_n=top_n_value, max_fraction=max_per_sector
    )

    if diversified_df.empty:
        logger.warning("No symbols selected after sector diversity constraint.")
        run_summary = _build_run_summary(
            today,
            cfg,
            timings,
            notes,
            row_counts,
            {},
            params.env_overrides,
            weights_version,
            csv_hash,
            used_cached_csv,
            sector_trimmed,
            week52_warnings,
            news_signal_summary,
        )
        _emit_empty_outputs(output_dir, generated_at, top_n_value, run_summary)
        empty_tiers = {}
        tier_display = _format_tier_counts(empty_tiers)
        summary_line = (
            f"Date={today} {_timezone_label(params.timezone, params.run_date)} | "
            f"TopN=0 | A/B/C={tier_display} | SectorCap={sector_trimmed} | "
            f"Cache={used_cached_csv} | Out={output_dir}"
        )
        logger.info(summary_line)
        return 2 if params.fail_on_empty else 0

    diversified_df = diversified_df.head(top_n_value).copy()
    diversified_df["rank"] = list(range(1, len(diversified_df) + 1))
    diversified_df["tags"] = diversified_df.apply(_tags_for_row, axis=1)

    rank_weights = rank_cfg.weights
    why_values: list[str] = []
    feature_columns: dict[int, list[str]] = {idx: [] for idx in range(1, 6)}
    for _, row in diversified_df.iterrows():
        contributions = _feature_contributions(row, rank_weights)
        positive_labels = [label for label, value in contributions if value > 0][:3]
        why_values.append(" + ".join(positive_labels) if positive_labels else "—")
        for idx in range(5):
            if idx < len(contributions):
                label, value = contributions[idx]
                feature_columns[idx + 1].append(f"{label}={value:.3f}")
            else:
                feature_columns[idx + 1].append("")

    generated_at = utils.timestamp_iso()

    full_watchlist = []
    for _, row in featured_df.iterrows():
        features_dict = _build_feature_dict(row)
        item = {
            "symbol": row.get("ticker"),
            "company": row.get("company"),
            "sector": row.get("sector"),
            "industry": row.get("industry"),
            "exchange": row.get("exchange"),
            "market_cap": row.get("market_cap"),
            "pe": row.get("pe"),
            "price": row.get("price"),
            "change_pct": row.get("change_pct"),
            "gap_pct": row.get("gap_pct"),
            "volume": row.get("volume"),
            "avg_volume_3m": row.get("avg_volume_3m"),
            "rel_volume": row.get("rel_volume"),
            "float_shares": row.get("float_shares"),
            "short_float_pct": row.get("short_float_pct"),
            "after_hours_change_pct": row.get("after_hours_change_pct"),
            "week52_range": row.get("week52_range"),
            "week52_pos": row.get("f_52w_pos"),
            "earnings_date": row.get("earnings_date").isoformat()
            if hasattr(row.get("earnings_date"), "isoformat")
            else row.get("earnings_date"),
            "analyst_recom": row.get("analyst_recom"),
            "features": features_dict,
            "score": row.get("score"),
            "ai_confidence": row.get("ai_confidence"),
            "tier": row.get("tier"),
            "tags": _tags_for_row(row),
            "rejection_reasons": row.get("rejection_reasons", []),
            "generated_at": generated_at,
        }
        if "insider_transactions" in row:
            item["insider_transactions"] = row.get("insider_transactions")
        if "institutional_transactions" in row:
            item["institutional_transactions"] = row.get("institutional_transactions")
        if "week52_pos" in row:
            item["week52_pos"] = row.get("week52_pos")
        full_watchlist.append(item)

    full_watchlist_df = pd.DataFrame(full_watchlist, columns=FULL_WATCHLIST_COLUMNS)

    top_rankings_records: list[dict[str, object]] = []
    for _, row in diversified_df.iterrows():
        top_rankings_records.append(
            {
                "rank": row.get("rank"),
                "symbol": row.get("ticker"),
                "score": row.get("score"),
                "ai_confidence": row.get("ai_confidence"),
            }
        )
    top_rankings_df = pd.DataFrame(top_rankings_records, columns=TOP_RANKINGS_COLUMNS)

    watchlist_records: list[dict[str, object]] = []
    for idx, (_, row) in enumerate(diversified_df.iterrows()):
        record: dict[str, object] = {
            "rank": row.get("rank"),
            "symbol": row.get("ticker"),
            "score": row.get("score"),
            "AIConfidence": row.get("ai_confidence"),
            "tier": row.get("tier"),
            "gap_pct": row.get("gap_pct"),
            "rel_volume": row.get("rel_volume"),
            "tags": row.get("tags"),
            "Why": why_values[idx] if idx < len(why_values) else "",
        }
        for feature_idx in range(1, 6):
            values = feature_columns.get(feature_idx, [])
            record[f"TopFeature{feature_idx}"] = values[idx] if idx < len(values) else ""
        watchlist_records.append(record)
    watchlist_table = pd.DataFrame(watchlist_records, columns=WATCHLIST_COLUMNS)

    metadata = {
        "generated_at": generated_at,
        "top_n": top_n_value,
        "insights": _build_metadata_insights(diversified_df, news_signal_summary),
        "table_schemas": _table_schemas_metadata(),
    }

    persist_start = time.perf_counter()
    persist.write_outputs(
        output_dir / "watchlist.db",
        watchlist=watchlist_table,
        top_rankings=top_rankings_df,
        full_watchlist=full_watchlist_df,
        metadata=metadata,
        rejections=rejected_for_csv,
    )
    timings["persist"] = time.perf_counter() - persist_start

    tier_counts = diversified_df["tier"].value_counts().to_dict()
    row_counts["topN"] = int(len(diversified_df))

    run_summary = _build_run_summary(
        today,
        cfg,
        timings,
        notes,
        row_counts,
        tier_counts,
        params.env_overrides,
        weights_version,
        csv_hash,
        used_cached_csv,
        sector_trimmed,
        week52_warnings,
        news_signal_summary,
    )
    persist.write_run_summary(output_dir / "watchlist.db", run_summary)

    summary_line = (
        f"Date={today} {_timezone_label(params.timezone, params.run_date)} | "
        f"TopN={row_counts['topN']} | A/B/C={_format_tier_counts(tier_counts)} | "
        f"SectorCap={sector_trimmed} | Cache={used_cached_csv} | Out={output_dir}"
    )
    logger.info(summary_line)

    return 0


__all__ = ["RunParams", "run"]
