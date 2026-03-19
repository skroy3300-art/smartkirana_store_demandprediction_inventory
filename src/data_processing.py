"""
data_processing.py
------------------
Handles all data ingestion, cleaning, feature engineering,
and preparation for model training.
"""

import pandas as pd
import numpy as np
from datetime import datetime


# ---------------------------------------------------------------------------
# Festival calendar (India) – extend as needed
# ---------------------------------------------------------------------------
FESTIVAL_DATES = {
    # Diwali approximations
    "2016-10-30": "Diwali", "2017-10-19": "Diwali", "2018-11-07": "Diwali",
    "2019-10-27": "Diwali", "2020-11-14": "Diwali", "2021-11-04": "Diwali",
    "2022-10-24": "Diwali", "2023-11-12": "Diwali", "2024-11-01": "Diwali",
    # Holi
    "2016-03-24": "Holi",   "2017-03-13": "Holi",   "2018-03-02": "Holi",
    "2019-03-21": "Holi",   "2020-03-10": "Holi",   "2021-03-29": "Holi",
    "2022-03-18": "Holi",   "2023-03-08": "Holi",   "2024-03-25": "Holi",
    # Eid (approximate)
    "2017-06-26": "Eid",    "2018-06-15": "Eid",    "2019-06-04": "Eid",
    "2020-05-24": "Eid",    "2021-05-13": "Eid",    "2022-05-02": "Eid",
    "2023-04-21": "Eid",    "2024-04-10": "Eid",
    # Christmas
    "2016-12-25": "Christmas", "2017-12-25": "Christmas",
    "2018-12-25": "Christmas", "2019-12-25": "Christmas",
    "2020-12-25": "Christmas", "2021-12-25": "Christmas",
    "2022-12-25": "Christmas", "2023-12-25": "Christmas",
}

# Days around festival that still see a spike
FESTIVAL_WINDOW = 7

# Perishable sub-categories
PERISHABLE_SUBCATEGORIES = {
    "Fresh Vegetables", "Fresh Fruits", "Dairy", "Eggs, Meat & Fish",
    "Bakery", "Organic Fruits", "Organic Vegetables",
}


def load_data(filepath: str) -> pd.DataFrame:
    """Load CSV and return a raw DataFrame."""
    df = pd.read_csv(filepath, encoding="utf-8-sig")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    • Rename columns to snake_case
    • Parse dates
    • Drop duplicates
    • Fill / drop missing values
    """
    df = df.copy()

    # Standardise column names
    df.columns = [c.strip().lower().replace(" ", "_").replace("&", "and") for c in df.columns]

    # Rename for convenience
    rename_map = {
        "order_id": "order_id",
        "customer_name": "customer_name",
        "category": "category",
        "sub_category": "sub_category",
        "city": "city",
        "order_date": "order_date",
        "region": "region",
        "sales": "sales",
        "discount": "discount",
        "profit": "profit",
        "state": "state",
    }
    df.rename(columns=rename_map, inplace=True)

    # Parse dates (handle both dd-mm-yyyy and mm/dd/yyyy)
    df["order_date"] = pd.to_datetime(df["order_date"], dayfirst=True, errors="coerce")

    # Drop rows where critical fields are missing
    df.dropna(subset=["order_date", "sales", "category", "sub_category"], inplace=True)

    # Fill remaining NaN numerics with 0
    df["discount"] = df["discount"].fillna(0)
    df["profit"] = df["profit"].fillna(0)

    # Remove duplicates
    df.drop_duplicates(inplace=True)

    # Ensure numeric types
    df["sales"] = pd.to_numeric(df["sales"], errors="coerce").fillna(0)
    df["discount"] = pd.to_numeric(df["discount"], errors="coerce").fillna(0)
    df["profit"] = pd.to_numeric(df["profit"], errors="coerce").fillna(0)

    return df


def _get_season(month: int) -> str:
    if month in (12, 1, 2):
        return "Winter"
    elif month in (3, 4, 5):
        return "Summer"
    elif month in (6, 7, 8, 9):
        return "Monsoon"
    else:
        return "Autumn"


def _is_festival_period(date: pd.Timestamp) -> tuple:
    """Return (is_festival: bool, festival_name: str)."""
    date_str = date.strftime("%Y-%m-%d")
    # Exact match
    if date_str in FESTIVAL_DATES:
        return True, FESTIVAL_DATES[date_str]
    # Window match
    for fest_date_str, name in FESTIVAL_DATES.items():
        fest_date = pd.to_datetime(fest_date_str)
        if abs((date - fest_date).days) <= FESTIVAL_WINDOW:
            return True, name
    return False, "None"


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add time-based, seasonal, festival and perishability features.
    """
    df = df.copy()

    # Date parts
    df["day"] = df["order_date"].dt.day
    df["month"] = df["order_date"].dt.month
    df["year"] = df["order_date"].dt.year
    df["day_of_week"] = df["order_date"].dt.dayofweek      # 0=Mon
    df["week_of_year"] = df["order_date"].dt.isocalendar().week.astype(int)
    df["quarter"] = df["order_date"].dt.quarter
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
    df["is_month_start"] = (df["day"] <= 5).astype(int)
    df["is_month_end"] = (df["day"] >= 25).astype(int)

    # Season
    df["season"] = df["month"].apply(_get_season)

    # Festival indicator
    festival_info = df["order_date"].apply(_is_festival_period)
    df["is_festival"] = festival_info.apply(lambda x: int(x[0]))
    df["festival_name"] = festival_info.apply(lambda x: x[1])

    # Perishable flag
    df["is_perishable"] = df["sub_category"].isin(PERISHABLE_SUBCATEGORIES).astype(int)

    # Net sales after discount
    df["net_sales"] = df["sales"] * (1 - df["discount"])

    # Profit margin
    df["profit_margin"] = np.where(
        df["sales"] > 0, df["profit"] / df["sales"], 0
    )

    return df


def aggregate_daily_sales(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate to daily level per Sub-Category for time-series modelling.
    Returns one row per (order_date, sub_category).
    """
    agg = (
        df.groupby(["order_date", "sub_category", "category",
                    "season", "is_festival", "festival_name",
                    "is_perishable", "day", "month", "year",
                    "day_of_week", "week_of_year", "quarter",
                    "is_weekend", "is_month_start", "is_month_end"])
        .agg(
            total_sales=("sales", "sum"),
            total_net_sales=("net_sales", "sum"),
            order_count=("order_id", "count"),
            avg_discount=("discount", "mean"),
            total_profit=("profit", "sum"),
        )
        .reset_index()
        .sort_values("order_date")
    )
    return agg


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """Label-encode low-cardinality categorical columns."""
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    for col in ["sub_category", "category", "season", "festival_name"]:
        if col in df.columns:
            df[col + "_enc"] = le.fit_transform(df[col].astype(str))
    return df


def get_feature_columns() -> list:
    """Return the list of feature columns used for model training."""
    return [
        "day", "month", "year", "day_of_week", "week_of_year", "quarter",
        "is_weekend", "is_month_start", "is_month_end",
        "is_festival", "is_perishable",
        "avg_discount", "order_count",
        "sub_category_enc", "category_enc", "season_enc",
    ]


def full_pipeline(filepath: str):
    """
    End-to-end convenience function: load → clean → engineer → aggregate → encode.
    Returns the processed DataFrame ready for modelling.
    """
    df_raw = load_data(filepath)
    df_clean = clean_data(df_raw)
    df_feat = engineer_features(df_clean)
    df_agg = aggregate_daily_sales(df_feat)
    df_enc = encode_categoricals(df_agg)
    return df_enc, df_feat  # return feature-engineered df for EDA (has season, is_festival, etc.)


if __name__ == "__main__":
    df, _ = full_pipeline("../data/supermart_sales.csv")
    print(df.shape)
    print(df.head(3))