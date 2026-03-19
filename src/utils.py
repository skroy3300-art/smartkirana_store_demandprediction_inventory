"""
utils.py
--------
Shared helper functions: EDA charts, report generation, formatting.
All chart functions return Plotly figures (compatible with Streamlit).

Handles DataFrames that have either 'sales' OR 'total_sales' column
(raw cleaned df vs aggregated processed df).
"""

import io
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# ── Colour Palette ─────────────────────────────────────────────────────────────
PALETTE = px.colors.qualitative.Set2
ACCENT = "#FF6B35"


# ── Internal helpers ───────────────────────────────────────────────────────────

def _resolve_sales(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure the DataFrame always has a 'sales' column.
      - 'sales' exists        → nothing to do
      - 'total_sales' exists  → copy to 'sales'
      - 'total_net_sales'     → copy to 'sales' as fallback
      - nothing found         → fill with zeros so charts never crash
    Always returns a copy.
    """
    df = df.copy()
    if "sales" not in df.columns:
        if "total_sales" in df.columns:
            df["sales"] = df["total_sales"]
        elif "total_net_sales" in df.columns:
            df["sales"] = df["total_net_sales"]
        else:
            df["sales"] = 0
    return df


def _ensure_col(df: pd.DataFrame, col: str, default=0) -> pd.DataFrame:
    """Add a column with a default value if it doesn't exist. Returns copy."""
    df = df.copy()
    if col not in df.columns:
        df[col] = default
    return df


def _get_season(month: int) -> str:
    if month in (12, 1, 2):    return "Winter"
    elif month in (3, 4, 5):   return "Summer"
    elif month in (6, 7, 8, 9): return "Monsoon"
    else:                       return "Autumn"


# ── EDA Charts ────────────────────────────────────────────────────────────────

def plot_sales_trend(df: pd.DataFrame) -> go.Figure:
    df = _resolve_sales(df)
    monthly = (
        df.groupby(df["order_date"].dt.to_period("M"))["sales"]
        .sum().reset_index()
    )
    monthly["order_date"] = monthly["order_date"].dt.to_timestamp()
    fig = px.line(
        monthly, x="order_date", y="sales",
        title="📈 Monthly Sales Trend",
        labels={"order_date": "Month", "sales": "Total Sales (₹)"},
        color_discrete_sequence=[ACCENT],
    )
    fig.update_traces(line_width=2.5, mode="lines+markers")
    fig.update_layout(hovermode="x unified", template="plotly_white")
    return fig


def plot_top_products(df: pd.DataFrame, top_n: int = 10) -> go.Figure:
    df = _resolve_sales(df)
    top = (
        df.groupby("sub_category")["sales"]
        .sum().nlargest(top_n).reset_index().sort_values("sales")
    )
    fig = px.bar(
        top, x="sales", y="sub_category", orientation="h",
        title=f"🏆 Top {top_n} Sub-Categories by Revenue",
        labels={"sales": "Total Sales (₹)", "sub_category": ""},
        color="sales", color_continuous_scale="Oranges",
    )
    fig.update_layout(template="plotly_white", showlegend=False)
    return fig


def plot_category_donut(df: pd.DataFrame) -> go.Figure:
    df = _resolve_sales(df)
    cat = df.groupby("category")["sales"].sum().reset_index()
    fig = px.pie(
        cat, names="category", values="sales",
        title="🛒 Sales Share by Category",
        hole=0.4, color_discrete_sequence=PALETTE,
    )
    fig.update_traces(textposition="inside", textinfo="percent+label")
    fig.update_layout(template="plotly_white")
    return fig


def plot_seasonal_demand(df: pd.DataFrame) -> go.Figure:
    df = _resolve_sales(df)
    if "season" not in df.columns:
        df["season"] = df["order_date"].dt.month.apply(_get_season)
    daily = df.groupby(["order_date", "season"])["sales"].sum().reset_index()
    season_order = ["Winter", "Summer", "Monsoon", "Autumn"]
    fig = px.box(
        daily, x="season", y="sales",
        category_orders={"season": season_order},
        title="🌦️ Seasonal Demand Distribution",
        labels={"sales": "Daily Sales (₹)", "season": "Season"},
        color="season", color_discrete_sequence=PALETTE,
    )
    fig.update_layout(template="plotly_white", showlegend=False)
    return fig


def plot_festival_impact(df: pd.DataFrame) -> go.Figure:
    df = _resolve_sales(df)
    df = _ensure_col(df, "is_festival", 0)
    daily = df.groupby(["order_date", "is_festival"])["sales"].sum().reset_index()
    daily["Festival"] = daily["is_festival"].map({0: "Non-Festival", 1: "Festival"})
    agg = daily.groupby("Festival")["sales"].mean().reset_index()
    fig = px.bar(
        agg, x="Festival", y="sales",
        title="🎉 Festival vs Non-Festival Average Daily Sales",
        labels={"sales": "Avg Daily Sales (₹)", "Festival": ""},
        color="Festival",
        color_discrete_map={"Festival": ACCENT, "Non-Festival": "#4A90D9"},
        text_auto=".2s",
    )
    fig.update_layout(template="plotly_white", showlegend=False)
    return fig


def plot_heatmap_dow_month(df: pd.DataFrame) -> go.Figure:
    df = _resolve_sales(df)
    df2 = df.copy()
    df2["dow"]   = df2["order_date"].dt.dayofweek
    df2["month"] = df2["order_date"].dt.month
    pivot = df2.pivot_table(
        values="sales", index="dow", columns="month", aggfunc="mean"
    ).fillna(0)
    dow_labels   = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
    month_labels = ["Jan","Feb","Mar","Apr","May","Jun",
                    "Jul","Aug","Sep","Oct","Nov","Dec"]
    fig = px.imshow(
        pivot,
        x=[month_labels[m-1] for m in pivot.columns],
        y=[dow_labels[d]      for d in pivot.index],
        color_continuous_scale="YlOrRd",
        title="🗓️ Avg Sales Heat-Map (Day × Month)",
        labels={"color": "Avg Sales (₹)"}, aspect="auto",
    )
    fig.update_layout(template="plotly_white")
    return fig


def plot_region_sales(df: pd.DataFrame) -> go.Figure:
    df = _resolve_sales(df)
    if "region" not in df.columns:
        df["region"] = "Unknown"
    reg = df.groupby(["region", "category"])["sales"].sum().reset_index()
    fig = px.treemap(
        reg, path=["region", "category"], values="sales",
        title="🗺️ Sales by Region & Category",
        color="sales", color_continuous_scale="Blues",
    )
    fig.update_layout(template="plotly_white")
    return fig


def plot_profit_vs_sales(df: pd.DataFrame) -> go.Figure:
    df = _resolve_sales(df)
    if "profit" not in df.columns:
        df["profit"] = df.get("total_profit", 0)
    sample = df.sample(min(1000, len(df)), random_state=42)
    fig = px.scatter(
        sample, x="sales", y="profit", color="category",
        title="💰 Profit vs Sales by Category",
        labels={"sales": "Sales (₹)", "profit": "Profit (₹)"},
        opacity=0.6, color_discrete_sequence=PALETTE,
    )
    fig.update_layout(template="plotly_white")
    return fig


# ── Prediction / Inventory Charts ─────────────────────────────────────────────

def plot_forecast(pred_df: pd.DataFrame, sub_cat: str = None) -> go.Figure:
    df = pred_df.copy()
    if sub_cat and sub_cat != "All Products":
        df = df[df["sub_category"] == sub_cat]
    fig = go.Figure()
    for sc in df["sub_category"].unique():
        sub = df[df["sub_category"] == sc]
        fig.add_trace(go.Scatter(
            x=sub["order_date"], y=sub["predicted_sales"],
            mode="lines+markers", name=sc, line=dict(width=2),
        ))
    if "is_festival" in df.columns:
        fest = df[df["is_festival"] == 1]
        if not fest.empty:
            fig.add_trace(go.Scatter(
                x=fest["order_date"], y=fest["predicted_sales"],
                mode="markers", name="Festival Day",
                marker=dict(size=10, color="red", symbol="star"),
            ))
    fig.update_layout(
        title="🔮 Demand Forecast", xaxis_title="Date",
        yaxis_title="Predicted Sales (₹)",
        template="plotly_white", hovermode="x unified",
    )
    return fig


def plot_model_comparison(lr_metrics: dict, rf_metrics: dict) -> go.Figure:
    metrics = ["MAE", "RMSE", "R2"]
    fig = go.Figure(data=[
        go.Bar(name="Linear Regression",
               x=metrics, y=[lr_metrics.get(m, 0) for m in metrics],
               marker_color="#4A90D9"),
        go.Bar(name="Random Forest",
               x=metrics, y=[rf_metrics.get(m, 0) for m in metrics],
               marker_color=ACCENT),
    ])
    fig.update_layout(
        barmode="group", title="📊 Model Performance Comparison",
        template="plotly_white", yaxis_title="Score",
    )
    return fig


def plot_feature_importance(importance_df: pd.DataFrame) -> go.Figure:
    df = importance_df.head(12).sort_values("importance")
    fig = px.bar(
        df, x="importance", y="feature", orientation="h",
        title="🔍 Feature Importance (Random Forest)",
        labels={"importance": "Importance Score", "feature": "Feature"},
        color="importance", color_continuous_scale="Teal",
    )
    fig.update_layout(template="plotly_white", showlegend=False)
    return fig


def plot_inventory_gauge(inv_df: pd.DataFrame) -> go.Figure:
    df = inv_df.head(6)
    fig = make_subplots(
        rows=2, cols=3,
        subplot_titles=df["Sub Category"].tolist(),
        specs=[[{"type": "indicator"}]*3]*2,
    )
    positions = [(1,1),(1,2),(1,3),(2,1),(2,2),(2,3)]
    for idx, (_, row) in enumerate(df.iterrows()):
        r, c = positions[idx]
        avg     = max(row["Avg Daily Demand (₹)"], 0)
        peak    = max(row["Peak Demand (₹)"], avg * 2, 1)
        reorder = row["Reorder Level (₹)"]
        gauge_max = peak * 1.2
        if reorder >= gauge_max:
            reorder = gauge_max * 0.75
        fig.add_trace(
            go.Indicator(
                mode="gauge+number", value=avg,
                gauge=dict(
                    axis=dict(range=[0, gauge_max]),
                    bar=dict(color=ACCENT),
                    steps=[
                        dict(range=[0, reorder],              color="#FFDDC1"),
                        dict(range=[reorder, gauge_max],      color="#FFA07A"),
                    ],
                    threshold=dict(
                        line=dict(color="red", width=3),
                        thickness=0.75, value=reorder,
                    ),
                ),
                title={"text": "Avg Daily (₹)"},
            ),
            row=r, col=c,
        )
    fig.update_layout(
        title_text="📦 Inventory Demand Gauges (vs Reorder Level)",
        template="plotly_white", height=450,
    )
    return fig


# ── Report Generation ─────────────────────────────────────────────────────────

def generate_csv_report(pred_df: pd.DataFrame, inv_df: pd.DataFrame) -> bytes:
    inv_renamed = inv_df.rename(columns={"Sub Category": "sub_category"})
    merged = pred_df.merge(inv_renamed, on="sub_category", how="left")
    merged["order_date"] = merged["order_date"].dt.strftime("%Y-%m-%d")
    buffer = io.BytesIO()
    merged.to_csv(buffer, index=False)
    return buffer.getvalue()