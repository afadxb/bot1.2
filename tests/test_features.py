import pandas as pd

from premarket import features


def test_winsorize_and_scale_range():
    series = pd.Series([1, 2, 3, 100])
    scaled = features.winsorize_and_scale(series)
    assert (scaled >= 0).all() and (scaled <= 1).all()


def test_build_features_basic():
    df = pd.DataFrame(
        {
            "rel_volume": [2.0],
            "avg_volume_3m": [2_000_000],
            "float_shares": [80_000_000],
            "gap_pct": [5.0],
            "change_pct": [4.0],
            "after_hours_change_pct": [1.0],
            "week52_pos": [0.9],
            "short_float_pct": [12.0],
            "analyst_recom": ["Buy"],
            "insider_transactions": ["1%"],
            "institutional_transactions": ["2%"],
            "price": [25.0],
        }
    )

    result = features.build_features(df, cfg=None)
    for col in [
        "f_relvol",
        "f_avgvol",
        "f_float_band",
        "f_gap",
        "f_change",
        "f_after_hours",
        "f_52w_pos",
        "f_short_float",
        "f_analyst",
        "f_insider_inst",
    ]:
        assert col in result.columns
    assert result.loc[0, "turnover_dollar"] == 25.0 * 2_000_000
