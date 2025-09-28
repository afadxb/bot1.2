import pandas as pd

from premarket import ranker


def _make_config():
    return ranker.RankerConfig(
        weights=ranker.RankerWeights(
            relvol=0.3,
            gap=0.2,
            avgvol=0.1,
            float_band=0.1,
            short_float=0.05,
            after_hours=0.05,
            change=0.05,
            w52pos=0.05,
            news_fresh=0.05,
            analyst=0.03,
            insider_inst=0.02,
        ),
        penalties=ranker.RankerPenalties(earnings_near=0.1, pe_outlier=0.05),
        caps=ranker.RankerCaps(max_single_negative=0.1),
        earnings_window_days=1,
    )


def test_compute_score_ordering(monkeypatch):
    cfg = _make_config()
    df = pd.DataFrame(
        {
            "f_relvol": [0.9, 0.1],
            "f_gap": [0.8, 0.2],
            "f_avgvol": [0.5, 0.5],
            "f_float_band": [0.4, 0.4],
            "f_short_float": [0.3, 0.3],
            "f_after_hours": [0.6, 0.4],
            "f_change": [0.5, 0.5],
            "f_w52pos": [0.5, 0.5],
            "f_news_fresh": [0.0, 0.0],
            "f_analyst": [0.3, 0.1],
            "f_insider_inst": [0.1, 0.1],
            "earnings_date": [None, None],
            "pe": [50, 250],
        }
    )

    scores = ranker.compute_score(df, cfg)
    assert scores.iloc[0] > scores.iloc[1]


def test_assign_tiers():
    scores = pd.Series([0.8, 0.6, 0.4])
    tiers = ranker.assign_tiers(scores)
    assert list(tiers) == ["A", "B", "C"]


def test_sector_diversity():
    df = pd.DataFrame(
        {
            "ticker": ["AAA", "BBB", "CCC"],
            "sector": ["Tech", "Tech", "Health"],
            "score": [0.9, 0.8, 0.7],
        }
    )
    capped, trimmed = ranker.apply_sector_diversity(df, top_n=3, max_fraction=0.5)
    assert len(capped) <= 3
    assert trimmed is True
    assert (capped["sector"] == "Tech").sum() <= 1
