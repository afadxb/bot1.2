"""Core SteadyAlpha Screener workflow."""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, Optional

import pandas as pd
import yaml

from premarket import features, filters, loader_finviz, normalize, ranker, utils
from premarket.news_probe import probe as news_probe
from steadyalpha.common.pushover import send as push_notification
from steadyalpha.settings import get_settings
from steadyalpha.storage.db import get_connection, init_pool
from steadyalpha.storage.migrations import apply_migrations

try:  # pragma: no cover - fallback when ulid isn't installed
    import ulid

    def _new_run_id() -> str:
        return str(ulid.new())

except ImportError:  # pragma: no cover - fallback for environments without ulid
    import uuid

    def _new_run_id() -> str:
        return uuid.uuid4().hex[:26]

LOGGER = logging.getLogger(__name__)

CONFIG_PATH = Path("config/strategy.yaml")


@dataclass
class PipelineContext:
    run_id: str
    run_date: date
    generated_at: str
    top_symbols: list[str]
    watchlist_records: list[dict[str, object]]
    top_n_records: list[dict[str, object]]
    run_summary: dict[str, object]
    diversified_df: pd.DataFrame
    featured_df: pd.DataFrame
    settings_used: dict[str, object]
    notes: list[str]


def _load_config() -> dict[str, object]:
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"Strategy config missing at {CONFIG_PATH}")
    return yaml.safe_load(CONFIG_PATH.read_text())


def _news_scores(symbols: list[str], news_cfg: dict[str, object]) -> Dict[str, float]:
    if not news_cfg.get("enabled") or not symbols:
        return {symbol: 0.0 for symbol in symbols}
    raw = news_probe(symbols, news_cfg)  # type: ignore[arg-type]
    scores: Dict[str, float] = {}
    for symbol, payload in raw.items():
        freshness = payload.get("freshness_hours") if isinstance(payload, dict) else None
        if freshness is None:
            scores[symbol] = 0.0
        else:
            freshness = max(0.0, float(freshness))
            limit = float(news_cfg.get("freshness_hours", 24) or 24)
            normalized = max(0.0, 1 - min(freshness, limit) / limit)
            scores[symbol] = float(max(0.0, min(1.0, normalized)))
    return scores


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
        now_day = utils.now_eastern().date()
        if abs((earnings_date.date() - now_day).days) <= 1:
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
        if value is None or (isinstance(value, float) and pd.isna(value)):
            features_map[col.replace("f_", "")] = 0.0
        else:
            features_map[col.replace("f_", "")] = float(value)
    return features_map


def _feature_contributions(row: pd.Series, weights: ranker.RankerWeights) -> list[tuple[str, float]]:
    feature_labels = {
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
    contributions: list[tuple[str, float]] = []
    for key, label in feature_labels.items():
        column = f"f_{key}"
        if column not in row:
            continue
        value = row.get(column)
        if value is None or (isinstance(value, float) and pd.isna(value)):
            contribution = 0.0
        else:
            weight = getattr(weights, key)
            contribution = float(value) * float(weight)
        contributions.append((label, contribution))
    contributions.sort(key=lambda item: item[1], reverse=True)
    return contributions


def _build_ranker_config(cfg: dict[str, object]) -> ranker.RankerConfig:
    weights = cfg["premarket"]["weights"]
    penalties = cfg["premarket"]["penalties"]
    caps = cfg["premarket"]["caps"]
    return ranker.RankerConfig(
        weights=ranker.RankerWeights(**weights),
        penalties=ranker.RankerPenalties(**penalties),
        caps=ranker.RankerCaps(**caps),
        earnings_window_days=cfg["premarket"]["earnings_exclude_window_days"],
    )


def _build_filter_config(cfg: dict[str, object]) -> filters.FilterConfig:
    pm_cfg = cfg["premarket"]
    return filters.FilterConfig(
        price_min=pm_cfg["price_min"],
        price_max=pm_cfg["price_max"],
        avg_vol_min=pm_cfg["avg_vol_min"],
        rel_vol_min=pm_cfg["rel_vol_min"],
        float_min=pm_cfg["float_min"],
        earnings_exclude_window_days=pm_cfg["earnings_exclude_window_days"],
        exclude_exchanges=pm_cfg.get("exclude_exchanges", []),
        exclude_countries=pm_cfg.get("exclude_countries", []),
    )


def _persist_optional_outputs(
    output_dir: Path,
    generated_at: str,
    top_n: int,
    full_watchlist: list[dict[str, object]],
    watchlist_table: pd.DataFrame,
    top_symbols: pd.DataFrame,
    run_summary: dict[str, object],
) -> None:
    from premarket import persist

    persist.write_json(full_watchlist, output_dir / "full_watchlist.json")
    persist.write_json(
        {
            "generated_at": generated_at,
            "top_n": top_n,
            "symbols": top_symbols["symbol"].tolist(),
            "ranking": top_symbols.to_dict(orient="records"),
        },
        output_dir / "topN.json",
    )
    persist.write_csv(watchlist_table, output_dir / "watchlist.csv")
    persist.write_json(run_summary, output_dir / "run_summary.json")


def _prepare_watchlist(
    diversified_df: pd.DataFrame,
    rank_cfg: ranker.RankerConfig,
) -> tuple[pd.DataFrame, list[str], dict[int, list[str]]]:
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
    return diversified_df.copy(), why_values, feature_columns


def _build_run_summary(
    date_str: str,
    cfg: dict[str, object],
    timings: Dict[str, float],
    notes: list[str],
    row_counts: Dict[str, int],
    tier_counts: Dict[str, int],
    weights_version: str,
    csv_hash: str,
    used_cached_csv: bool,
    sector_trimmed: bool,
    week52_warnings: int,
) -> dict[str, object]:
    summary = {
        "date": date_str,
        "filters": cfg["premarket"],
        "timings_sec": {k: round(v, 3) for k, v in timings.items()},
        "notes": notes,
        "row_counts": row_counts,
        "tiers": tier_counts,
        "weights_version": weights_version or "default",
        "csv_hash": csv_hash,
        "sector_cap_applied": bool(sector_trimmed),
        "used_cached_csv": bool(used_cached_csv),
        "week52_warning_count": int(week52_warnings),
    }
    return summary


def _create_output_dir(run_date: date) -> Path:
    base_dir = Path("data/watchlists")
    output_dir = base_dir / run_date.isoformat()
    utils.ensure_directory(output_dir)
    return output_dir


def _extract_numeric(row: dict[str, object], key: str) -> Optional[float]:
    value = row.get(key)
    numeric = utils.safe_float(value)
    if numeric is None:
        return None
    return float(numeric)


def _unique_symbols(rows: Iterable[dict[str, object]]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for row in rows:
        symbol = row.get("ticker")
        if not symbol:
            continue
        symbol_str = str(symbol)
        if symbol_str in seen:
            continue
        seen.add(symbol_str)
        ordered.append(symbol_str)
    return ordered


def _fetch_symbol_ids(cursor, symbols: list[str]) -> dict[str, int]:
    if not symbols:
        return {}
    placeholders = ", ".join(["%s"] * len(symbols))
    cursor.execute(
        f"SELECT symbol, id FROM securities WHERE symbol IN ({placeholders})",
        tuple(symbols),
    )
    mapping: dict[str, int] = {}
    for symbol, identifier in cursor.fetchall():
        try:
            mapping[str(symbol)] = int(identifier)
        except (TypeError, ValueError):
            continue
    return mapping


def premarket_scan() -> PipelineContext:
    """Execute the screener workflow and persist outputs."""

    settings = get_settings()
    utils.configure_timezone(settings.tz)
    now = utils.now_eastern()
    run_day = now.date()
    run_id = _new_run_id()
    LOGGER.info("Starting SteadyAlpha Screener run %s", run_id)

    raw_cfg = _load_config()
    premarket_cfg = raw_cfg["premarket"]
    top_n_value = int(settings.top_n or premarket_cfg["top_n"])
    max_per_sector = float(premarket_cfg.get("max_per_sector", 0.4))
    news_cfg = raw_cfg.get("news", {"enabled": False})

    output_dir = _create_output_dir(run_day)
    raw_csv_path = Path("data/raw") / run_day.isoformat() / "finviz_elite.csv"
    utils.ensure_directory(raw_csv_path.parent)

    timings: Dict[str, float] = {}
    notes: list[str] = []
    row_counts: Dict[str, int] = {"raw": 0, "qualified": 0, "rejected": 0, "topN": 0}

    start_ts = time.perf_counter()
    try:
        csv_path = loader_finviz.download_csv(
            settings.finviz_export_url,
            raw_csv_path,
            use_cache=True,
        )
    except RuntimeError:
        timings["download"] = time.perf_counter() - start_ts
        LOGGER.error("Failed to download CSV and no cache available.")
        raise

    timings["download"] = time.perf_counter() - start_ts
    used_cached_csv = csv_path != raw_csv_path
    notes.append(f"used_cached_csv: {used_cached_csv}")

    try:
        csv_hash = hashlib.sha256(csv_path.read_bytes()).hexdigest()
    except OSError:
        csv_hash = ""
        LOGGER.warning("Unable to hash CSV at %s", csv_path)

    start_ts = time.perf_counter()
    df = loader_finviz.read_csv(csv_path)
    row_counts["raw"] = int(len(df))
    df = normalize.normalize_columns(df)
    df, week52_warnings = normalize.coerce_types(df)
    timings["normalize"] = time.perf_counter() - start_ts
    if week52_warnings:
        notes.append(f"week52_warnings: {week52_warnings}")

    filter_cfg = _build_filter_config(raw_cfg)
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

    qualified_df = qualified_df.copy()
    gap_series = pd.to_numeric(qualified_df.get("gap_pct"), errors="coerce").fillna(0.0)
    vol_series = pd.to_numeric(qualified_df.get("volume"), errors="coerce").fillna(0)
    filtered_records: list[dict[str, object]] = []
    for record, gap, vol in zip(qualified_df.to_dict(orient="records"), gap_series.tolist(), vol_series.tolist()):
        if (gap >= settings.min_gap_pct) and (vol >= settings.min_pm_volume):
            filtered_records.append(record)
    qualified_df = pd.DataFrame(filtered_records)

    row_counts["qualified"] = int(len(qualified_df))
    row_counts["rejected"] = int(len(rejected_df))

    if qualified_df.empty:
        raise RuntimeError("No candidates qualified for Top-N watchlist")

    symbols = qualified_df.get("ticker", pd.Series(dtype=str)).fillna("").astype(str).tolist()
    news_scores = _news_scores(symbols, news_cfg)
    qualified_df["news_fresh_score"] = [news_scores.get(sym, 0.0) for sym in symbols]

    featured_df = features.build_features(qualified_df, raw_cfg)

    rank_cfg = _build_ranker_config(raw_cfg)
    start_ts = time.perf_counter()
    scores = ranker.compute_score(featured_df, rank_cfg)
    featured_df["score"] = scores
    featured_df["tier"] = ranker.assign_tiers(scores)
    timings["score"] = time.perf_counter() - start_ts

    featured_df.sort_values(
        by=["score", "turnover_dollar", "ticker"], ascending=[False, False, True], inplace=True
    )

    diversified_df, sector_trimmed = ranker.apply_sector_diversity(
        featured_df, top_n=top_n_value, max_fraction=max_per_sector
    )

    if diversified_df.empty:
        raise RuntimeError("No symbols selected after sector diversity constraint")

    diversified_df = diversified_df.head(top_n_value).copy()
    diversified_df["rank"] = range(1, len(diversified_df) + 1)
    diversified_df["tags"] = diversified_df.apply(_tags_for_row, axis=1)

    enriched_df, why_values, feature_columns = _prepare_watchlist(diversified_df, rank_cfg)

    generated_at = utils.timestamp_iso(now)

    full_watchlist: list[dict[str, object]] = []
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

    watchlist_table = enriched_df[
        ["rank", "ticker", "score", "tier", "gap_pct", "rel_volume", "tags"]
    ].rename(columns={"ticker": "symbol"})
    watchlist_table = watchlist_table.copy()
    watchlist_table["Why"] = why_values
    for idx, values in feature_columns.items():
        watchlist_table[f"TopFeature{idx}"] = values

    top_symbols = watchlist_table[["symbol", "score"]].copy()
    top_symbols_records = top_symbols.to_dict(orient="records")
    row_counts["topN"] = int(len(watchlist_table))
    run_summary = _build_run_summary(
        run_day.isoformat(),
        raw_cfg,
        timings,
        notes,
        row_counts,
        watchlist_table["tier"].value_counts().to_dict(),
        premarket_cfg.get("weights_version", "default"),
        csv_hash,
        used_cached_csv,
        sector_trimmed,
        week52_warnings,
    )

    _persist_optional_outputs(
        output_dir,
        generated_at,
        top_n_value,
        full_watchlist,
        watchlist_table,
        top_symbols,
        run_summary,
    )

    watchlist_records = watchlist_table.to_dict(orient="records")
    for idx, record in enumerate(watchlist_records, start=1):
        record["rank"] = idx

    context = PipelineContext(
        run_id=run_id,
        run_date=run_day,
        generated_at=generated_at,
        top_symbols=top_symbols["symbol"].tolist(),
        watchlist_records=watchlist_records,
        top_n_records=top_symbols_records,
        run_summary=run_summary,
        diversified_df=enriched_df,
        featured_df=featured_df,
        settings_used={
            "min_gap_pct": settings.min_gap_pct,
            "min_pm_volume": settings.min_pm_volume,
            "top_n": top_n_value,
        },
        notes=notes,
    )

    _persist_mysql(context)
    _send_summary(context)
    LOGGER.info("Completed SteadyAlpha Screener run %s with %d symbols", run_id, len(context.top_symbols))
    return context


def _persist_mysql(context: PipelineContext) -> None:
    init_pool()
    conn = get_connection()
    try:
        apply_migrations(conn)
        with conn.cursor() as cursor:
            started_at = datetime.fromisoformat(context.generated_at)
            if started_at.tzinfo is not None:
                started_at = started_at.astimezone(timezone.utc).replace(tzinfo=None)
            finished_at = datetime.utcnow()
            cursor.execute(
                """
                INSERT INTO run_summary (run_id, started_at, finished_at, notes)
                VALUES (%s, %s, %s, %s)
                ON DUPLICATE KEY UPDATE finished_at = VALUES(finished_at), notes = VALUES(notes)
                """,
                (
                    context.run_id,
                    started_at,
                    finished_at,
                    json.dumps(context.notes)[:255],
                ),
            )
            diversified_rows = context.diversified_df.to_dict(orient="records")
            symbols = _unique_symbols(diversified_rows)
            symbol_ids = _fetch_symbol_ids(cursor, symbols)
            missing_symbols = [s for s in symbols if s not in symbol_ids]
            if missing_symbols:
                LOGGER.warning(
                    "Skipping shortlist persistence for symbols missing securities.id: %s",
                    ", ".join(sorted(missing_symbols)),
                )

            shortlist_rows = []
            generated_at = datetime.fromisoformat(context.generated_at)
            if generated_at.tzinfo is not None:
                generated_at = generated_at.astimezone(timezone.utc).replace(tzinfo=None)
            for row in diversified_rows:
                ticker = row.get("ticker")
                if not ticker:
                    continue
                symbol_id = symbol_ids.get(str(ticker))
                if symbol_id is None:
                    continue
                price_value = _extract_numeric(row, "price")
                average_volume_value = _extract_numeric(row, "average_volume")
                if average_volume_value is None:
                    average_volume_value = _extract_numeric(row, "avg_volume_3m")
                if price_value is not None and average_volume_value is not None:
                    liquidity_value = price_value * average_volume_value
                else:
                    liquidity_value = row.get("liquidity_score")
                    if liquidity_value is None:
                        liquidity_value = row.get("score", 0.0)
                shortlist_rows.append(
                    (
                        context.run_date,
                        symbol_id,
                        float(liquidity_value or 0.0),
                        price_value,
                        average_volume_value,
                        generated_at,
                    )
                )
            if shortlist_rows:
                cursor.executemany(
                    """
                    INSERT INTO shortlists (
                        run_date, symbol_id, liquidity_score, price, average_volume, created_at
                    ) VALUES (%s, %s, %s, %s, %s, %s)
                    ON DUPLICATE KEY UPDATE
                        liquidity_score = VALUES(liquidity_score),
                        price = VALUES(price),
                        average_volume = VALUES(average_volume),
                        created_at = VALUES(created_at)
                    """,
                    shortlist_rows,
                )
        conn.commit()
    finally:
        conn.close()


def _send_summary(context: PipelineContext) -> None:
    if not context.top_symbols:
        return
    preview = context.top_symbols[:5]
    body = ", ".join(preview)
    if len(context.top_symbols) > len(preview):
        body = f"{body} …"
    message = f"Screener TopN: {body} (N={len(context.top_symbols)})"
    push_notification("SteadyAlpha Screener", message)


__all__ = ["premarket_scan", "PipelineContext"]
