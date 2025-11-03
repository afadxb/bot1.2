import pandas as pd

from premarket import normalize


def test_normalize_columns_and_types():
    df = pd.DataFrame(
        {
            "Ticker": ["AAA"],
            "Relative Vol.": ["1.8"],
            "Average Volume (3M)": ["1,500,000"],
            "Change": ["5%"],
            "52-Week Range": ["10 - 20"],
            "Price": ["18.5"],
            "Previous Close": ["17.5"],
        }
    )

    normalized = normalize.normalize_columns(df)
    assert "rel_volume" in normalized.columns

    coerced, warnings = normalize.coerce_types(normalized)
    assert coerced.loc[0, "rel_volume"] == 1.8
    assert coerced.loc[0, "avg_volume_3m"] == 1_500_000
    assert round(coerced.loc[0, "change_pct"], 2) == 5.0
    assert round(coerced.loc[0, "gap_pct"], 2) == round((18.5 - 17.5) / 17.5 * 100, 2)

    assert 0.0 <= coerced.loc[0, "week52_pos"] <= 1.0
    assert warnings == 0


def test_gap_column_maps_to_gap_percent():
    df = pd.DataFrame({"Ticker": ["AAA"], "Gap": ["4%"]})

    normalized = normalize.normalize_columns(df)
    assert "gap_pct" in normalized.columns

    coerced, _ = normalize.coerce_types(normalized)
    assert coerced.loc[0, "gap_pct"] == 4.0



def test_coerce_types_parses_suffixes():
    df = pd.DataFrame(
        {
            "Ticker": ["ZZZ"],
            "Average Volume (3M)": ["850K"],
            "Float": ["1.4B"],
            "Float %": ["65%"],
            "Short Float": ["12.5%"],
        }
    )

    normalized = normalize.normalize_columns(df)
    coerced, warnings = normalize.coerce_types(normalized)

    assert coerced.loc[0, "avg_volume_3m"] == 850_000
    assert coerced.loc[0, "float_shares"] == 1_400_000_000
    assert coerced.loc[0, "float_pct"] == 65.0
    assert coerced.loc[0, "short_float_pct"] == 12.5
    assert warnings == 0


def test_normalize_preserves_company_and_sector():
    df = pd.DataFrame(
        {
            "Ticker": ["CCC"],
            "Company": ["Gamma Corp"],
            "Sector": ["Industrials"],
            "Price": ["$12.75"],
            "Average Volume (3M)": ["1.2M"],
            "Relative Volume": ["1.4"],
        }
    )

    normalized = normalize.normalize_columns(df)
    assert "company" in normalized.columns
    assert "sector" in normalized.columns

    coerced, warnings = normalize.coerce_types(normalized)
    assert coerced.loc[0, "company"] == "Gamma Corp"
    assert coerced.loc[0, "sector"] == "Industrials"
    assert coerced.loc[0, "price"] == 12.75
    assert coerced.loc[0, "avg_volume_3m"] == 1_200_000
    assert coerced.loc[0, "rel_volume"] == 1.4
    assert warnings == 0
