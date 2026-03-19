"""
model_training.py
-----------------
Trains Linear Regression and Random Forest models on the processed
Supermart dataset. Saves models + encoders to disk.
Generates evaluation metrics and 7–30-day demand forecasts.
"""

import os
import pickle
import warnings
import numpy as np
import pandas as pd
from datetime import timedelta

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _get_season(month: int) -> str:
    if month in (12, 1, 2): return "Winter"
    elif month in (3, 4, 5): return "Summer"
    elif month in (6, 7, 8, 9): return "Monsoon"
    else: return "Autumn"


FESTIVAL_DATES = {
    "2016-10-30","2017-10-19","2018-11-07","2019-10-27",
    "2020-11-14","2021-11-04","2022-10-24","2023-11-12","2024-11-01",
}
FESTIVAL_WINDOW = 7


def _is_festival(date: pd.Timestamp) -> int:
    for fd in FESTIVAL_DATES:
        if abs((date - pd.to_datetime(fd)).days) <= FESTIVAL_WINDOW:
            return 1
    return 0


def build_future_dates(days: int = 30) -> pd.DataFrame:
    """
    Build a future date DataFrame with all engineered features
    needed for prediction (without sales/labels).
    """
    last_date = pd.Timestamp.today().normalize()
    future_dates = [last_date + timedelta(days=i+1) for i in range(days)]
    rows = []
    for d in future_dates:
        rows.append({
            "order_date": d,
            "day": d.day,
            "month": d.month,
            "year": d.year,
            "day_of_week": d.dayofweek,
            "week_of_year": d.isocalendar()[1],
            "quarter": d.quarter,
            "is_weekend": int(d.dayofweek in [5, 6]),
            "is_month_start": int(d.day <= 5),
            "is_month_end": int(d.day >= 25),
            "is_festival": _is_festival(d),
            "season": _get_season(d.month),
        })
    return pd.DataFrame(rows)


# ── Core Training ─────────────────────────────────────────────────────────────

def prepare_training_data(df: pd.DataFrame):
    """
    Expects the fully processed (encoded) DataFrame from data_processing.
    Returns X_train, X_test, y_train, y_test + feature list.
    """
    feature_cols = [
        "day", "month", "year", "day_of_week", "week_of_year", "quarter",
        "is_weekend", "is_month_start", "is_month_end",
        "is_festival", "is_perishable",
        "avg_discount", "order_count",
        "sub_category_enc", "category_enc", "season_enc",
    ]
    # Keep only columns that exist
    feature_cols = [c for c in feature_cols if c in df.columns]

    X = df[feature_cols].fillna(0)
    y = df["total_sales"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    return X_train, X_test, y_train, y_test, feature_cols


def evaluate_model(model, X_test, y_test) -> dict:
    preds = model.predict(X_test)
    preds = np.clip(preds, 0, None)          # sales cannot be negative
    mae  = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    ss_res = np.sum((y_test - preds) ** 2)
    ss_tot = np.sum((y_test - y_test.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot != 0 else 0
    return {"MAE": round(mae, 2), "RMSE": round(rmse, 2), "R2": round(r2, 4)}


def train_models(df: pd.DataFrame) -> dict:
    """
    Train Linear Regression + Random Forest.
    Returns a result dict with models, metrics, and feature importance.
    """
    X_train, X_test, y_train, y_test, feature_cols = prepare_training_data(df)

    # ── Linear Regression ──
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    lr_metrics = evaluate_model(lr, X_test, y_test)

    # ── Random Forest ──
    rf = RandomForestRegressor(
        n_estimators=200,
        max_depth=12,
        min_samples_split=5,
        random_state=42,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)
    rf_metrics = evaluate_model(rf, X_test, y_test)

    # Feature importance from RF
    importance_df = pd.DataFrame({
        "feature": feature_cols,
        "importance": rf.feature_importances_,
    }).sort_values("importance", ascending=False)

    # Save models to disk
    with open(os.path.join(MODELS_DIR, "linear_regression.pkl"), "wb") as f:
        pickle.dump(lr, f)
    with open(os.path.join(MODELS_DIR, "random_forest.pkl"), "wb") as f:
        pickle.dump(rf, f)
    with open(os.path.join(MODELS_DIR, "feature_cols.pkl"), "wb") as f:
        pickle.dump(feature_cols, f)

    print("✅  Models saved to", MODELS_DIR)
    print(f"   Linear Regression → {lr_metrics}")
    print(f"   Random Forest     → {rf_metrics}")

    return {
        "lr_model": lr,
        "rf_model": rf,
        "lr_metrics": lr_metrics,
        "rf_metrics": rf_metrics,
        "feature_cols": feature_cols,
        "feature_importance": importance_df,
        "X_test": X_test,
        "y_test": y_test,
    }


# ── Prediction ─────────────────────────────────────────────────────────────────

def load_models():
    """Load saved models and feature columns from disk."""
    with open(os.path.join(MODELS_DIR, "linear_regression.pkl"), "rb") as f:
        lr = pickle.load(f)
    with open(os.path.join(MODELS_DIR, "random_forest.pkl"), "rb") as f:
        rf = pickle.load(f)
    with open(os.path.join(MODELS_DIR, "feature_cols.pkl"), "rb") as f:
        feature_cols = pickle.load(f)
    return lr, rf, feature_cols


def predict_future_demand(
    df_processed: pd.DataFrame,
    model,
    feature_cols: list,
    days: int = 30,
    sub_categories: list = None,
) -> pd.DataFrame:
    """
    Generate demand predictions for the next `days` days.
    If sub_categories list is provided, predict for each; else predict overall.
    """
    future = build_future_dates(days)

    # Encode season
    season_map = {"Winter": 0, "Summer": 1, "Monsoon": 2, "Autumn": 3}
    future["season_enc"] = future["season"].map(season_map).fillna(0).astype(int)

    results = []

    if sub_categories is None:
        # Use median values from training for category-level features
        future["sub_category_enc"] = 0
        future["category_enc"] = 0
        future["is_perishable"] = 0
        future["avg_discount"] = df_processed["avg_discount"].median()
        future["order_count"] = df_processed["order_count"].median()

        X_future = future[[c for c in feature_cols if c in future.columns]].fillna(0)
        preds = np.clip(model.predict(X_future), 0, None)
        future["predicted_sales"] = preds
        future["sub_category"] = "All Products"
        results.append(future[["order_date", "sub_category", "predicted_sales", "is_festival", "season"]])
    else:
        for sub_cat in sub_categories:
            mask = df_processed["sub_category"] == sub_cat
            sub_df = df_processed[mask]
            if sub_df.empty:
                continue

            fut_copy = future.copy()
            fut_copy["sub_category_enc"] = sub_df["sub_category_enc"].iloc[0]
            fut_copy["category_enc"]     = sub_df["category_enc"].iloc[0]
            fut_copy["is_perishable"]    = sub_df["is_perishable"].iloc[0]
            fut_copy["avg_discount"]     = sub_df["avg_discount"].median()
            fut_copy["order_count"]      = sub_df["order_count"].median()

            X_future = fut_copy[[c for c in feature_cols if c in fut_copy.columns]].fillna(0)
            preds = np.clip(model.predict(X_future), 0, None)
            fut_copy["predicted_sales"] = preds
            fut_copy["sub_category"] = sub_cat
            results.append(fut_copy[["order_date", "sub_category", "predicted_sales", "is_festival", "season"]])

    return pd.concat(results, ignore_index=True) if results else pd.DataFrame()


# ── Standalone run ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from src.data_processing import full_pipeline

    df, _ = full_pipeline("../data/supermart_sales.csv")
    results = train_models(df)
    print("\nFeature Importance (Top 10):")
    print(results["feature_importance"].head(10).to_string(index=False))
