
import os
import sys
import pickle
import warnings
import numpy as np
import pandas as pd
import streamlit as st

warnings.filterwarnings("ignore")

#  Path setup 
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from src.data_processing import full_pipeline, clean_data, load_data
from src.model_training import train_models, predict_future_demand, load_models
from src.inventory_optimization import compute_inventory_recommendations, generate_alert_summary
from src.utils import (
    plot_sales_trend, plot_top_products, plot_category_donut,
    plot_seasonal_demand, plot_festival_impact, plot_heatmap_dow_month,
    plot_region_sales, plot_profit_vs_sales,
    plot_forecast, plot_model_comparison, plot_feature_importance,
    plot_inventory_gauge, generate_csv_report,
)

MODELS_DIR = os.path.join(BASE_DIR, "models")

# Page Config

st.set_page_config(
    page_title="Smart Kirana Store",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS 
st.markdown("""
<style>
    /* Header gradient */
    .main-header {
        background: linear-gradient(135deg, #FF6B35 0%, #F7C59F 100%);
        padding: 1.5rem 2rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
    }
    .main-header h1 { color: white; margin: 0; font-size: 2rem; }
    .main-header p  { color: #fff8f0; margin: 0.3rem 0 0; }

    /* KPI cards */
    .kpi-card {
        background: white;
        border: 1px solid #e8e8e8;
        border-radius: 10px;
        padding: 1rem 1.2rem;
        text-align: center;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    }
    .kpi-value { font-size: 1.8rem; font-weight: 700; color: #FF6B35; }
    .kpi-label { font-size: 0.82rem; color: #666; margin-top: 0.2rem; }

    /* Alert box */
    .alert-box {
        background: #FFF3CD;
        border-left: 4px solid #FF6B35;
        border-radius: 6px;
        padding: 0.6rem 1rem;
        margin: 0.4rem 0;
        font-size: 0.88rem;
    }

    /* Section headers */
    .section-title {
        font-size: 1.15rem;
        font-weight: 600;
        color: #333;
        border-bottom: 2px solid #FF6B35;
        padding-bottom: 0.3rem;
        margin: 1rem 0 0.8rem;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab"] { font-size: 0.95rem; }

    /* Sidebar */
    section[data-testid="stSidebar"] { background: #FFF8F0; }
</style>
""", unsafe_allow_html=True)



# Session State
if "df_processed" not in st.session_state:
    st.session_state["df_processed"] = None
if "df_clean" not in st.session_state:
    st.session_state["df_clean"] = None
if "train_results" not in st.session_state:
    st.session_state["train_results"] = None
if "predictions" not in st.session_state:
    st.session_state["predictions"] = None
if "inventory" not in st.session_state:
    st.session_state["inventory"] = None


# Sidebar

with st.sidebar:
    st.image("https://img.icons8.com/color/96/shop.png", width=60)
    st.markdown("## 🛒 Smart Kirana")
    st.markdown("*Demand Prediction & Inventory Optimization*")
    st.divider()

    st.markdown("### 📂 Dataset")
    uploaded_file = st.file_uploader(
        "Upload Sales CSV", type=["csv"],
        help="Upload the Supermart grocery sales dataset",
    )

    # Use bundled dataset if none uploaded
    default_path = os.path.join(BASE_DIR, "data", "supermart_sales.csv")
    if uploaded_file is None and os.path.exists(default_path):
        st.info("🗂️ Using bundled dataset")
        data_source = default_path
    elif uploaded_file:
        data_source = uploaded_file
        st.success("✅ File uploaded!")
    else:
        data_source = None
        st.warning("⚠️ No dataset found")

    st.divider()
    st.markdown("### ⚙️ Inventory Settings")
    lead_time   = st.slider("Lead Time (days)",      1, 14, 3)
    service_lvl = st.selectbox("Service Level",     [0.90, 0.95, 0.98, 0.99], index=1)
    order_cost  = st.number_input("Ordering Cost (₹)", 100, 5000, 500, step=50)
    hold_rate   = st.slider("Holding Cost Rate (%)", 5, 40, 20) / 100

    st.divider()
    st.markdown("### 🔮 Forecast Settings")
    forecast_days = st.slider("Forecast Horizon (days)", 7, 30, 14)
    model_choice  = st.radio("Prediction Model",
                             ["Random Forest", "Linear Regression"],
                             index=0)



# Header
st.markdown("""
<div class="main-header">
  <h1>🛒 Smart Kirana Store</h1>
  <p>Demand Prediction & Inventory Optimization System</p>
</div>
""", unsafe_allow_html=True)



# Data Loading

@st.cache_data(show_spinner="⏳ Processing dataset...")
def load_and_process(src):
    if isinstance(src, str):
        return full_pipeline(src)
    else:
        import tempfile, os
        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
            tmp.write(src.read())
            tmp_path = tmp.name
        result = full_pipeline(tmp_path)
        os.unlink(tmp_path)
        return result


if data_source is not None:
    try:
        df_proc, df_clean = load_and_process(data_source)
        st.session_state["df_processed"] = df_proc
        st.session_state["df_clean"]     = df_clean
    except Exception as e:
        st.error(f"❌ Error loading data: {e}")
        st.stop()
else:
    st.warning("👈 Please upload a CSV dataset or ensure the bundled file exists.")
    st.stop()

df_proc  = st.session_state["df_processed"]
df_clean = st.session_state["df_clean"]


# KPI Strip
total_sales    = df_clean["sales"].sum()
total_orders   = len(df_clean)
unique_prods   = df_clean["sub_category"].nunique()
avg_order_val  = df_clean["sales"].mean()
total_profit   = df_clean["profit"].sum()
festival_pct   = df_proc["is_festival"].mean() * 100

kpi_cols = st.columns(6)
kpis = [
    ("₹{:,.0f}".format(total_sales),   "Total Revenue"),
    ("{:,}".format(total_orders),       "Total Orders"),
    (str(unique_prods),                 "Product Categories"),
    ("₹{:,.0f}".format(avg_order_val),  "Avg Order Value"),
    ("₹{:,.0f}".format(total_profit),   "Total Profit"),
    ("{:.1f}%".format(festival_pct),    "Festival Day %"),
]
for col, (val, label) in zip(kpi_cols, kpis):
    with col:
        st.markdown(f"""
        <div class="kpi-card">
          <div class="kpi-value">{val}</div>
          <div class="kpi-label">{label}</div>
        </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)



# Tabs

tabs = st.tabs([
    "📊 EDA Dashboard",
    "🤖 Model Training",
    "🔮 Demand Forecast",
    "📦 Inventory Optimization",
    "⚠️ Alerts & Reports",
])


# TAB 1 – EDA
with tabs[0]:
    st.markdown('<div class="section-title">Exploratory Data Analysis</div>',
                unsafe_allow_html=True)

    # Row 1
    c1, c2 = st.columns([2, 1])
    with c1:
        st.plotly_chart(plot_sales_trend(df_clean), use_container_width=True)
    with c2:
        st.plotly_chart(plot_category_donut(df_clean), use_container_width=True)

    # Row 2
    c3, c4 = st.columns(2)
    with c3:
        st.plotly_chart(plot_top_products(df_clean, top_n=10), use_container_width=True)
    with c4:
        st.plotly_chart(plot_seasonal_demand(df_clean), use_container_width=True)

    # Row 3
    c5, c6 = st.columns(2)
    with c5:
        st.plotly_chart(plot_festival_impact(df_proc), use_container_width=True)
    with c6:
        st.plotly_chart(plot_heatmap_dow_month(df_clean), use_container_width=True)

    # Row 4
    c7, c8 = st.columns(2)
    with c7:
        st.plotly_chart(plot_region_sales(df_clean), use_container_width=True)
    with c8:
        st.plotly_chart(plot_profit_vs_sales(df_clean), use_container_width=True)

    st.divider()
    st.markdown("### 🔎 Raw Data Preview")
    st.dataframe(df_clean.head(500), use_container_width=True, height=300)



# TAB 2 – Model Training

with tabs[1]:
    st.markdown('<div class="section-title">Machine Learning Model Training</div>',
                unsafe_allow_html=True)

    col_info, col_btn = st.columns([3, 1])
    with col_info:
        st.markdown("""
        **Two models are trained and compared:**
        - 🔵 **Linear Regression** – Fast baseline, interpretable coefficients
        - 🟠 **Random Forest** – Ensemble model, captures non-linear patterns

        Training uses an 80/20 train-test split. 
        Metrics: MAE (lower=better), RMSE (lower=better), R² (higher=better).
        """)
    with col_btn:
        train_btn = st.button("🚀 Train Models", type="primary", use_container_width=True)

    # Check if pre-trained models exist
    models_exist = all(
        os.path.exists(os.path.join(MODELS_DIR, f))
        for f in ["random_forest.pkl", "linear_regression.pkl", "feature_cols.pkl"]
    )

    if models_exist and st.session_state["train_results"] is None:
        st.info("💾 Pre-trained models found. Click **Train Models** to retrain, or proceed to Forecast tab.")

    if train_btn:
        with st.spinner("🔄 Training models... this may take ~30 seconds"):
            try:
                results = train_models(df_proc)
                st.session_state["train_results"] = results
                st.success("✅ Models trained and saved successfully!")
            except Exception as e:
                st.error(f"Training failed: {e}")

    if st.session_state["train_results"]:
        results = st.session_state["train_results"]

        # Metrics table
        st.markdown("### 📈 Model Metrics")
        metrics_df = pd.DataFrame({
            "Model": ["Linear Regression", "Random Forest"],
            "MAE (₹)":  [results["lr_metrics"]["MAE"],  results["rf_metrics"]["MAE"]],
            "RMSE (₹)": [results["lr_metrics"]["RMSE"], results["rf_metrics"]["RMSE"]],
            "R²":        [results["lr_metrics"]["R2"],   results["rf_metrics"]["R2"]],
        })
        # Highlight best per metric
        def highlight_best(s):
            if s.name == "R²":
                best = s.max()
                return ["background-color: #d4edda" if v == best else "" for v in s]
            else:
                best = s.min()
                return ["background-color: #d4edda" if v == best else "" for v in s]

        st.dataframe(
            metrics_df.style.apply(highlight_best, subset=["MAE (₹)", "RMSE (₹)", "R²"]),
            use_container_width=True, hide_index=True,
        )

        # Comparison chart
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(
                plot_model_comparison(results["lr_metrics"], results["rf_metrics"]),
                use_container_width=True,
            )
        with c2:
            st.plotly_chart(
                plot_feature_importance(results["feature_importance"]),
                use_container_width=True,
            )

        # Winner callout
        if results["rf_metrics"]["RMSE"] < results["lr_metrics"]["RMSE"]:
            winner = "🌲 Random Forest"
            delta  = results["lr_metrics"]["RMSE"] - results["rf_metrics"]["RMSE"]
        else:
            winner = "📏 Linear Regression"
            delta  = results["rf_metrics"]["RMSE"] - results["lr_metrics"]["RMSE"]

        st.success(f"**Best model: {winner}** — RMSE is ₹{delta:,.0f} lower than the other model.")

# TAB 3 – Demand Forecast

with tabs[2]:
    st.markdown('<div class="section-title">Demand Forecasting</div>',
                unsafe_allow_html=True)

    # Load models
    @st.cache_resource
    def get_models():
        if all(os.path.exists(os.path.join(MODELS_DIR, f))
               for f in ["random_forest.pkl", "linear_regression.pkl", "feature_cols.pkl"]):
            return load_models()
        return None, None, None

    lr_model, rf_model, feature_cols = get_models()

    if lr_model is None:
        st.warning("⚠️ No trained models found. Go to **Model Training** tab and click Train.")
        st.stop()

    active_model = rf_model if model_choice == "Random Forest" else lr_model

    # Sub-category selector
    all_subcats = sorted(df_clean["sub_category"].unique().tolist())
    selected_subcats = st.multiselect(
        "Select Sub-Categories to Forecast",
        options=all_subcats,
        default=all_subcats[:5],
        help="Leave empty to forecast all",
    )
    if not selected_subcats:
        selected_subcats = all_subcats

    if st.button("🔮 Generate Forecast", type="primary"):
        with st.spinner(f"Forecasting {forecast_days} days for {len(selected_subcats)} categories..."):
            preds = predict_future_demand(
                df_proc, active_model, feature_cols,
                days=forecast_days,
                sub_categories=selected_subcats,
            )
            st.session_state["predictions"] = preds
        st.success(f"✅ Forecast generated for {forecast_days} days!")

    if st.session_state["predictions"] is not None:
        preds = st.session_state["predictions"]

        # Summary KPIs
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Total Predicted Revenue",
                  f"₹{preds['predicted_sales'].sum():,.0f}")
        k2.metric("Daily Average",
                  f"₹{preds['predicted_sales'].mean():,.0f}")
        k3.metric("Peak Day",
                  f"₹{preds['predicted_sales'].max():,.0f}")
        k4.metric("Festival Days",
                  f"{preds['is_festival'].sum()}")

        # Forecast chart
        view_cat = st.selectbox("View forecast for:", ["All"] + sorted(preds["sub_category"].unique().tolist()))
        if view_cat == "All":
            agg_preds = preds.groupby("order_date")["predicted_sales"].sum().reset_index()
            agg_preds["sub_category"] = "All Products"
            agg_preds["is_festival"] = preds.groupby("order_date")["is_festival"].max().values
            agg_preds["season"] = preds.groupby("order_date")["season"].first().values
            st.plotly_chart(plot_forecast(agg_preds), use_container_width=True)
        else:
            st.plotly_chart(plot_forecast(preds, view_cat), use_container_width=True)

        # Detailed forecast table
        st.markdown("### 📋 Forecast Table")
        display_df = preds.copy()
        display_df["order_date"] = display_df["order_date"].dt.strftime("%Y-%m-%d")
        display_df["predicted_sales"] = display_df["predicted_sales"].round(2)
        display_df["is_festival"] = display_df["is_festival"].map({0: "No", 1: "🎉 Yes"})
        st.dataframe(display_df, use_container_width=True, height=350)


# TAB 4 – Inventory Optimization

with tabs[3]:
    st.markdown('<div class="section-title">Inventory Optimization (EOQ Model)</div>',
                unsafe_allow_html=True)

    if st.session_state["predictions"] is None:
        st.warning("⚠️ Please generate a forecast first (Demand Forecast tab).")
    else:
        preds = st.session_state["predictions"]

        if st.button("📦 Compute Inventory Recommendations", type="primary"):
            with st.spinner("Running EOQ + Safety Stock calculations..."):
                inv = compute_inventory_recommendations(
                    preds, df_clean,
                    ordering_cost=order_cost,
                    holding_cost_rate=hold_rate,
                    lead_time_days=lead_time,
                    service_level=service_lvl,
                )
                st.session_state["inventory"] = inv
            st.success("✅ Inventory recommendations computed!")

        if st.session_state["inventory"] is not None:
            inv = st.session_state["inventory"]

            # EOQ explanation
            with st.expander("ℹ️ How EOQ is calculated"):
                st.latex(r"EOQ = \sqrt{\frac{2 \cdot D \cdot S}{H}}")
                st.markdown("""
                - **D** = Annual demand  
                - **S** = Ordering cost per order (₹)  
                - **H** = Holding cost per unit per year (₹)  
                
                **Safety Stock** = Z × √(Lead Time × σ²demand)  
                **Reorder Level** = Avg Daily Demand × Lead Time + Safety Stock  
                
                For **perishable items**, order quantity = Avg Daily Demand × Shelf Life × 0.85
                """)

            # Gauge chart
            st.plotly_chart(plot_inventory_gauge(inv), use_container_width=True)

            # Table
            st.markdown("### 📊 Full Inventory Recommendations")
            styled = inv.style.applymap(
                lambda v: "color: red; font-weight: bold"
                if isinstance(v, str) and ("⚡" in v or "⚠️" in v or "🎉" in v) else "",
                subset=["Alerts"],
            )
            st.dataframe(styled, use_container_width=True, height=420)



# TAB 5 – Alerts & Reports

with tabs[4]:
    st.markdown('<div class="section-title">⚠️ Alerts & Report Download</div>',
                unsafe_allow_html=True)

    if st.session_state["inventory"] is None or st.session_state["predictions"] is None:
        st.warning("Please complete Forecasting and Inventory steps first.")
    else:
        inv   = st.session_state["inventory"]
        preds = st.session_state["predictions"]
        alerts = generate_alert_summary(inv)

        # Alerts panel
        st.markdown("### 🔔 Active Alerts")
        if alerts:
            for alert in alerts:
                st.markdown(f'<div class="alert-box">{alert}</div>',
                            unsafe_allow_html=True)
        else:
            st.success("✅ No critical alerts – all inventory levels are normal.")

        st.divider()

        # Festival demand table
        st.markdown("### 🎉 Festival Demand Summary")
        fest_items = inv[inv["Festival Boost (%)"] > 0].sort_values(
            "Festival Boost (%)", ascending=False
        )
        if not fest_items.empty:
            st.dataframe(
                fest_items[["Sub Category", "Avg Daily Demand (₹)",
                            "Festival Boost (%)", "Recommended Order Qty (₹)", "Alerts"]],
                use_container_width=True,
            )
        else:
            st.info("No festival-period demand spikes detected in forecast window.")

        st.divider()

        # Perishables
        st.markdown("### 🥦 Perishable Items")
        perishables = inv[inv["Is Perishable"] == "Yes"]
        if not perishables.empty:
            st.dataframe(
                perishables[["Sub Category","Avg Daily Demand (₹)",
                             "Recommended Order Qty (₹)","EOQ Type","Alerts"]],
                use_container_width=True,
            )
            st.info("💡 Perishable items are ordered based on shelf-life, not classical EOQ, to minimise waste.")

        st.divider()

        # Download report
        st.markdown("### 📥 Download Full Report")
        csv_bytes = generate_csv_report(preds, inv)
        st.download_button(
            label="⬇️ Download Prediction + Inventory Report (CSV)",
            data=csv_bytes,
            file_name="smart_kirana_report.csv",
            mime="text/csv",
            type="primary",
        )

        st.caption(f"Report generated with {len(preds)} forecast rows × {len(inv)} product categories.")


# Footer 
st.divider()
st.markdown("""
<div style="text-align:center; color:#999; font-size:0.8rem; padding-bottom:1rem;">
    🛒 <strong>Smart Kirana Store</strong> &nbsp;|&nbsp;
    Built with Python · Streamlit · scikit-learn · Plotly &nbsp;|&nbsp;
    Data Science Portfolio Project
</div>
""", unsafe_allow_html=True)
