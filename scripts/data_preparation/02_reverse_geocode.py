"""Reverse geocode all unique images: lat/lon -> city, admin1, country (cc).

Then enrich with:
  - ISO3 country code (mapped from cc/ISO2)
  - World Bank income class (high / upper-middle / lower-middle / low / unknown)
  - UN Global North / Global South classification
  - Continent (broad UN region)

Output: data_processed/image_geo.parquet
"""
from pathlib import Path
import pandas as pd
import reverse_geocoder as rg

ROOT = Path(__file__).resolve().parents[2]
IN = ROOT / "data_processed/image_table.parquet"
OUT = ROOT / "data_processed/image_geo.parquet"


# --- World Bank income classification (FY2024 list, ISO2 country codes) ---
# Source: World Bank country & lending groups, 2024 fiscal year.
# H = high; UM = upper-middle; LM = lower-middle; L = low.
# Curated subset covering all PP2.0-relevant cities.
WB_INCOME = {
    # High income
    "AU": "H", "AT": "H", "BE": "H", "CA": "H", "CH": "H", "CL": "H", "CZ": "H",
    "DE": "H", "DK": "H", "EE": "H", "ES": "H", "FI": "H", "FR": "H", "GB": "H",
    "GR": "H", "HK": "H", "HR": "H", "HU": "H", "IE": "H", "IL": "H", "IS": "H",
    "IT": "H", "JP": "H", "KR": "H", "KW": "H", "LT": "H", "LU": "H", "LV": "H",
    "MC": "H", "MO": "H", "MT": "H", "NL": "H", "NO": "H", "NZ": "H", "PA": "H",
    "PL": "H", "PT": "H", "QA": "H", "SA": "H", "SE": "H", "SG": "H", "SI": "H",
    "SK": "H", "TW": "H", "US": "H", "AE": "H", "RO": "H", "UY": "H", "VA": "H",
    # Upper middle
    "AR": "UM", "BG": "UM", "BR": "UM", "BW": "UM", "BY": "UM", "CN": "UM",
    "CO": "UM", "CR": "UM", "CU": "UM", "DO": "UM", "EC": "UM", "FJ": "UM",
    "GA": "UM", "GE": "UM", "GT": "UM", "ID": "UM", "IQ": "UM", "JM": "UM",
    "JO": "UM", "KZ": "UM", "LB": "UM", "LY": "UM", "MK": "UM", "MN": "UM",
    "MX": "UM", "MY": "UM", "NA": "UM", "PE": "UM", "PY": "UM", "RS": "UM",
    "RU": "UM", "TH": "UM", "TR": "UM", "VE": "UM", "ZA": "UM", "AL": "UM",
    "AM": "UM", "AZ": "UM", "BA": "UM", "MD": "UM",
    # Lower middle
    "AO": "LM", "BD": "LM", "BO": "LM", "CI": "LM", "CM": "LM", "DZ": "LM",
    "EG": "LM", "GH": "LM", "HN": "LM", "IN": "LM", "IR": "LM", "KE": "LM",
    "KG": "LM", "KH": "LM", "LA": "LM", "LK": "LM", "MA": "LM", "MM": "LM",
    "NG": "LM", "NI": "LM", "NP": "LM", "PH": "LM", "PK": "LM", "PS": "LM",
    "SN": "LM", "SV": "LM", "TJ": "LM", "TN": "LM", "UA": "LM", "UZ": "LM",
    "VN": "LM", "YE": "LM", "ZM": "LM", "ZW": "LM", "TZ": "LM", "PG": "LM",
    "MN": "LM", "MR": "LM", "DJ": "LM", "TL": "LM", "VU": "LM", "MV": "LM",
    "BT": "LM", "ET": "LM", "MG": "LM",
    # Low income
    "AF": "L", "BF": "L", "BI": "L", "CD": "L", "CF": "L", "ER": "L", "GM": "L",
    "GN": "L", "KP": "L", "LR": "L", "ML": "L", "MW": "L", "MZ": "L", "NE": "L",
    "RW": "L", "SD": "L", "SL": "L", "SO": "L", "SS": "L", "SY": "L", "TD": "L",
    "TG": "L", "UG": "L", "YE": "L",
}


# --- UN Global North / South ---
# UNCTAD definition: Global North = developed economies (essentially OECD-style + a few).
# Everyone else = Global South.
GLOBAL_NORTH_ISO2 = {
    "AU", "AT", "BE", "CA", "CH", "CY", "CZ", "DE", "DK", "EE", "ES", "FI", "FR",
    "GB", "GR", "HU", "IE", "IL", "IS", "IT", "JP", "KR", "LT", "LU", "LV", "MT",
    "NL", "NO", "NZ", "PL", "PT", "RO", "SE", "SI", "SK", "US", "BG", "HR",
}


# --- Continent / region ---
ISO2_TO_CONTINENT = {
    # Africa
    **dict.fromkeys([
        "DZ", "AO", "BJ", "BW", "BF", "BI", "CM", "CV", "CF", "TD", "KM", "CD",
        "CG", "CI", "DJ", "EG", "GQ", "ER", "ET", "GA", "GM", "GH", "GN", "GW",
        "KE", "LS", "LR", "LY", "MG", "MW", "ML", "MR", "MU", "MA", "MZ", "NA",
        "NE", "NG", "RW", "ST", "SN", "SC", "SL", "SO", "ZA", "SS", "SD", "SZ",
        "TZ", "TG", "TN", "UG", "ZM", "ZW",
    ], "Africa"),
    # Asia
    **dict.fromkeys([
        "AF", "AM", "AZ", "BH", "BD", "BT", "BN", "KH", "CN", "GE", "HK", "IN",
        "ID", "IR", "IQ", "IL", "JP", "JO", "KZ", "KP", "KR", "KW", "KG", "LA",
        "LB", "MO", "MY", "MV", "MN", "MM", "NP", "OM", "PK", "PS", "PH", "QA",
        "SA", "SG", "LK", "SY", "TW", "TJ", "TH", "TL", "TR", "TM", "AE", "UZ",
        "VN", "YE",
    ], "Asia"),
    # Europe
    **dict.fromkeys([
        "AL", "AD", "AT", "BY", "BE", "BA", "BG", "HR", "CY", "CZ", "DK", "EE",
        "FI", "FR", "DE", "GR", "HU", "IS", "IE", "IT", "XK", "LV", "LI", "LT",
        "LU", "MT", "MD", "MC", "ME", "NL", "MK", "NO", "PL", "PT", "RO", "RU",
        "SM", "RS", "SK", "SI", "ES", "SE", "CH", "UA", "GB", "VA",
    ], "Europe"),
    # North America
    **dict.fromkeys([
        "AG", "BS", "BB", "BZ", "CA", "CR", "CU", "DM", "DO", "SV", "GD", "GT",
        "HT", "HN", "JM", "MX", "NI", "PA", "KN", "LC", "VC", "TT", "US",
    ], "North America"),
    # South America
    **dict.fromkeys([
        "AR", "BO", "BR", "CL", "CO", "EC", "GY", "PY", "PE", "SR", "UY", "VE",
    ], "South America"),
    # Oceania
    **dict.fromkeys([
        "AU", "FJ", "KI", "MH", "FM", "NR", "NZ", "PW", "PG", "WS", "SB", "TO",
        "TV", "VU",
    ], "Oceania"),
}


def main() -> None:
    print(f"reading {IN} ...")
    df = pd.read_parquet(IN)
    print(f"  {len(df):,} unique images")

    coords = list(zip(df["lat"].tolist(), df["lon"].tolist()))
    print("running reverse_geocoder (offline KD-tree on GeoNames) ...")
    results = rg.search(coords)
    print(f"  got {len(results):,} results")

    df["geo_name"] = [r["name"] for r in results]
    df["admin1"] = [r["admin1"] for r in results]
    df["admin2"] = [r["admin2"] for r in results]
    df["cc"] = [r["cc"] for r in results]

    df["income_class"] = df["cc"].map(WB_INCOME).fillna("unknown")
    df["global_south"] = (~df["cc"].isin(GLOBAL_NORTH_ISO2)).astype(int)
    df["continent"] = df["cc"].map(ISO2_TO_CONTINENT).fillna("unknown")

    # Use admin1+cc as the "city" granularity since GeoNames "name" can be a tiny suburb.
    # PP2.0 paper organizes around 56 named cities; admin1+cc is a robust proxy.
    df["city_proxy"] = df["geo_name"] + ", " + df["admin1"] + ", " + df["cc"]

    print()
    print("country distribution (top 20):")
    print(df["cc"].value_counts().head(20).to_string())
    print()
    print("continent distribution:")
    print(df["continent"].value_counts().to_string())
    print()
    print("global_south split:")
    print(df["global_south"].value_counts().to_string())
    print()
    print("income_class split:")
    print(df["income_class"].value_counts().to_string())
    print()
    n_unknown_income = int((df["income_class"] == "unknown").sum())
    if n_unknown_income > 0:
        print(f"WARNING: {n_unknown_income} images have unknown income_class. Their countries:")
        print(df[df["income_class"] == "unknown"]["cc"].value_counts().head().to_string())

    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT, index=False)
    print(f"\nwrote {OUT}  ({OUT.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
