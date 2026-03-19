"""
inventory_optimization.py
--------------------------
Implements:
  • Economic Order Quantity (EOQ)
  • Safety Stock calculation
  • Reorder Level
  • Reorder Quantity suggestion
  • Special logic for perishable items
  • Alerts for low stock / high demand
"""

import numpy as np
import pandas as pd
from typing import Optional


# ── Constants (configurable via Streamlit sidebar) ────────────────────────────
DEFAULT_HOLDING_COST_RATE = 0.20     # 20% of unit cost per year
DEFAULT_ORDERING_COST = 500.0        # ₹500 per order
DEFAULT_LEAD_TIME_DAYS = 3           # supplier lead time
DEFAULT_SERVICE_LEVEL = 0.95         # 95% → z = 1.645
DEFAULT_UNIT_COST = 100.0            # fallback unit cost in ₹

# z-scores for common service levels
Z_SCORES = {0.90: 1.282, 0.95: 1.645, 0.98: 2.054, 0.99: 2.326}

# Perishable shelf life (days) – shorter => smaller order qty
PERISHABLE_SHELF_LIFE = {
    "Fresh Vegetables": 3,
    "Fresh Fruits": 4,
    "Dairy": 5,
    "Eggs, Meat & Fish": 2,
    "Bakery": 2,
    "Organic Fruits": 4,
    "Organic Vegetables": 3,
}


# ── Core Formulas ─────────────────────────────────────────────────────────────

def eoq(
    annual_demand: float,
    ordering_cost: float = DEFAULT_ORDERING_COST,
    holding_cost_per_unit: float = None,
    unit_cost: float = DEFAULT_UNIT_COST,
    holding_cost_rate: float = DEFAULT_HOLDING_COST_RATE,
) -> float:
    """
    Economic Order Quantity = sqrt(2 * D * S / H)
    D = annual demand (units)
    S = ordering cost (₹)
    H = holding cost per unit per year (₹)
    """
    if holding_cost_per_unit is None:
        holding_cost_per_unit = unit_cost * holding_cost_rate

    if annual_demand <= 0 or holding_cost_per_unit <= 0:
        return 0.0

    return np.sqrt(2 * annual_demand * ordering_cost / holding_cost_per_unit)


def safety_stock(
    daily_demand_std: float,
    lead_time_days: float = DEFAULT_LEAD_TIME_DAYS,
    service_level: float = DEFAULT_SERVICE_LEVEL,
    lead_time_std: float = 0.0,
    daily_demand_avg: float = 0.0,
) -> float:
    """
    Safety Stock = Z * sqrt(LT * σ_demand² + D_avg² * σ_LT²)

    If lead_time_std is 0 → simplified: SS = Z * σ_demand * sqrt(LT)
    """
    z = Z_SCORES.get(service_level, 1.645)

    variance_demand = lead_time_days * (daily_demand_std ** 2)
    variance_lt = (daily_demand_avg ** 2) * (lead_time_std ** 2)
    total_variance = variance_demand + variance_lt

    return z * np.sqrt(total_variance)


def reorder_level(
    avg_daily_demand: float,
    lead_time_days: float = DEFAULT_LEAD_TIME_DAYS,
    ss: float = 0.0,
) -> float:
    """
    Reorder Level = (Avg daily demand × Lead time) + Safety Stock
    """
    return (avg_daily_demand * lead_time_days) + ss


def perishable_order_qty(
    avg_daily_demand: float,
    shelf_life_days: int,
    safety_factor: float = 0.85,
) -> float:
    """
    For perishable items, order no more than can be sold within shelf life.
    Apply a safety factor to avoid waste.
    """
    return avg_daily_demand * shelf_life_days * safety_factor


# ── Per-Product Inventory Summary ─────────────────────────────────────────────

def compute_inventory_recommendations(
    predictions_df: pd.DataFrame,
    df_historical: pd.DataFrame,
    ordering_cost: float = DEFAULT_ORDERING_COST,
    holding_cost_rate: float = DEFAULT_HOLDING_COST_RATE,
    lead_time_days: float = DEFAULT_LEAD_TIME_DAYS,
    service_level: float = DEFAULT_SERVICE_LEVEL,
    unit_cost_map: Optional[dict] = None,
) -> pd.DataFrame:
    """
    Given a predictions DataFrame (columns: sub_category, predicted_sales)
    and historical data, compute EOQ, safety stock, reorder level, and alerts.

    Returns a DataFrame with one row per sub_category.
    """
    if unit_cost_map is None:
        unit_cost_map = {}

    results = []

    sub_cats = predictions_df["sub_category"].unique()

    for sub_cat in sub_cats:
        # ── Predicted demand ────────────────────────────────────────────
        pred_mask = predictions_df["sub_category"] == sub_cat
        pred_sales = predictions_df.loc[pred_mask, "predicted_sales"]

        avg_daily_demand = pred_sales.mean()
        demand_std       = pred_sales.std() if len(pred_sales) > 1 else 0.0
        peak_demand      = pred_sales.max()
        days_forecast    = len(pred_sales)
        annual_demand    = avg_daily_demand * 365

        # ── Unit cost ───────────────────────────────────────────────────
        if sub_cat in unit_cost_map:
            unit_cost = unit_cost_map[sub_cat]
        elif not df_historical.empty and "sub_category" in df_historical.columns:
            hist_mask = df_historical["sub_category"] == sub_cat
            unit_cost = df_historical.loc[hist_mask, "sales"].median()
            unit_cost = unit_cost if not np.isnan(unit_cost) else DEFAULT_UNIT_COST
        else:
            unit_cost = DEFAULT_UNIT_COST

        holding_cost_pu = unit_cost * holding_cost_rate

        # ── Is perishable? ──────────────────────────────────────────────
        is_perishable = sub_cat in PERISHABLE_SHELF_LIFE
        shelf_life    = PERISHABLE_SHELF_LIFE.get(sub_cat, None)

        # ── EOQ / Order Qty ─────────────────────────────────────────────
        if is_perishable and shelf_life:
            order_qty = perishable_order_qty(avg_daily_demand, shelf_life)
            eoq_note  = f"Perishable (shelf life {shelf_life}d)"
        else:
            order_qty = eoq(annual_demand, ordering_cost, holding_cost_pu, unit_cost)
            eoq_note  = "EOQ"

        # ── Safety Stock & Reorder Level ─────────────────────────────────
        ss  = safety_stock(demand_std, lead_time_days, service_level,
                           daily_demand_avg=avg_daily_demand)
        rl  = reorder_level(avg_daily_demand, lead_time_days, ss)

        # ── Festival demand spike ────────────────────────────────────────
        fest_mask = pred_mask & (predictions_df["is_festival"] == 1)
        festival_boost = 0.0
        if fest_mask.any():
            fest_avg = predictions_df.loc[fest_mask, "predicted_sales"].mean()
            festival_boost = ((fest_avg - avg_daily_demand) / avg_daily_demand * 100
                              if avg_daily_demand > 0 else 0.0)

        # ── Alerts ──────────────────────────────────────────────────────
        alerts = []
        if peak_demand > avg_daily_demand * 2:
            alerts.append("⚡ Demand spike detected")
        if is_perishable:
            alerts.append("🥦 Perishable – order frequently & small")
        if festival_boost > 30:
            alerts.append(f"🎉 Festival spike +{festival_boost:.0f}% – stock up!")
        if avg_daily_demand > 0 and order_qty < avg_daily_demand * lead_time_days:
            alerts.append("⚠️ Order qty below lead-time coverage – increase!")

        results.append({
            "Sub Category": sub_cat,
            "Avg Daily Demand (₹)": round(avg_daily_demand, 2),
            "Demand Std Dev": round(demand_std, 2),
            "Peak Demand (₹)": round(peak_demand, 2),
            "Safety Stock (₹)": round(ss, 2),
            "Reorder Level (₹)": round(rl, 2),
            "Recommended Order Qty (₹)": round(order_qty, 2),
            "EOQ Type": eoq_note,
            "Is Perishable": "Yes" if is_perishable else "No",
            "Festival Boost (%)": round(festival_boost, 1),
            "Forecast Days": days_forecast,
            "Alerts": " | ".join(alerts) if alerts else "✅ Normal",
        })

    return pd.DataFrame(results)


def generate_alert_summary(inv_df: pd.DataFrame) -> list:
    """
    Returns a flat list of alert strings for the Streamlit notification panel.
    """
    alerts = []
    for _, row in inv_df.iterrows():
        if row["Alerts"] != "✅ Normal":
            alerts.append(f"**{row['Sub Category']}**: {row['Alerts']}")
    return alerts


# ── Standalone run ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Quick smoke test
    dummy_pred = pd.DataFrame({
        "sub_category": ["Masalas", "Fresh Vegetables"] * 15,
        "predicted_sales": np.random.uniform(500, 3000, 30),
        "is_festival": [0] * 28 + [1, 1],
        "season": ["Monsoon"] * 30,
    })
    dummy_hist = pd.DataFrame({
        "sub_category": ["Masalas", "Fresh Vegetables"],
        "sales": [1200.0, 800.0],
    })

    inv = compute_inventory_recommendations(dummy_pred, dummy_hist)
    print(inv.to_string(index=False))
