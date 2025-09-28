import pandas as pd

from premarket import filters


def test_apply_hard_filters():
    df = pd.DataFrame(
        [
            {
                "ticker": "AAA",
                "price": 25.0,
                "avg_volume_3m": 2_000_000,
                "rel_volume": 1.6,
                "float_shares": 50_000_000,
                "exchange": "NASDAQ",
                "country": "USA",
            },
            {
                "ticker": "BBB",
                "price": 3.0,
                "avg_volume_3m": 100_000,
                "rel_volume": 1.0,
                "float_shares": 5_000_000,
                "exchange": "OTC",
                "country": "USA",
            },
        ]
    )

    cfg = filters.FilterConfig(
        price_min=5,
        price_max=150,
        avg_vol_min=1_000_000,
        rel_vol_min=1.5,
        float_min=10_000_000,
        earnings_exclude_window_days=1,
        exclude_exchanges=["OTC"],
        exclude_countries=[],
    )

    qualified, rejected = filters.apply_hard_filters(df, cfg)
    assert list(qualified["ticker"]) == ["AAA"]
    assert "price_below_min" in rejected.iloc[0]["rejection_reasons"]
    assert "exchange_excluded" in rejected.iloc[0]["rejection_reasons"]
