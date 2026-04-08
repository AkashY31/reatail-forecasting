"""
=============================================================
 Phase 7 — Streamlit Dashboard
 Cloud: Deploy as Docker container on
   AWS ECS Fargate / Azure Container Apps / GCP Cloud Run
 Run locally: streamlit run src/serving/dashboard.py
=============================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from configs.config import DATA_DIR, OUTPUT_DIR

# ── Page config ───────────────────────────────────────────────
st.set_page_config(
    page_title="Retail Demand Forecasting",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────
st.markdown("""
<style>
    .metric-card {
        background: #f8f9fa;
        border-radius: 10px;
        padding: 1rem 1.5rem;
        border-left: 4px solid #185FA5;
    }
    .alert-red  { border-left-color: #993C1D; }
    .alert-green{ border-left-color: #0F6E56; }
    h1 { font-size: 1.8rem !important; }
</style>
""", unsafe_allow_html=True)


# ── Data loader (cached) ──────────────────────────────────────
@st.cache_data(ttl=300)
def load_data():
    path = DATA_DIR / "raw" / "retail_sales.parquet"
    if path.exists():
        return pd.read_parquet(path)
    # Fallback: generate on the fly
    from src.data.data_generator import generate_retail_dataset
    return generate_retail_dataset(save=False)


@st.cache_data(ttl=300)
def load_processed():
    path = DATA_DIR / "processed" / "features.parquet"
    if path.exists():
        return pd.read_parquet(path)
    return None


# ── Sidebar ───────────────────────────────────────────────────
def render_sidebar(df: pd.DataFrame):
    st.sidebar.image("https://img.icons8.com/fluency/96/warehouse.png", width=60)
    st.sidebar.title("Filters")

    stores = sorted(df["store_id"].unique())
    skus   = sorted(df["sku_id"].unique())
    cats   = sorted(df["category"].unique())

    sel_store = st.sidebar.multiselect("Store", stores, default=stores[:3])
    sel_sku   = st.sidebar.multiselect("SKU",   skus,   default=skus[:5])
    sel_cat   = st.sidebar.multiselect("Category", cats, default=cats)

    date_min  = df["date"].min().date()
    date_max  = df["date"].max().date()
    date_range = st.sidebar.date_input(
        "Date range",
        value=[date_min, date_max],
        min_value=date_min,
        max_value=date_max,
    )

    return sel_store, sel_sku, sel_cat, date_range


# ── KPI cards ─────────────────────────────────────────────────
def render_kpis(df: pd.DataFrame):
    col1, col2, col3, col4, col5 = st.columns(5)
    total_demand  = int(df["demand"].sum())
    avg_daily     = round(df.groupby("date")["demand"].sum().mean(), 1)
    promo_lift    = round(
        df[df["promotion"]==1]["demand"].mean() /
        (df[df["promotion"]==0]["demand"].mean() + 1e-8) - 1, 3) * 100
    missing_pct   = round(df["demand"].isna().mean() * 100, 2)
    stockout_pct  = round(df["stockout"].mean() * 100, 2)

    col1.metric("Total demand", f"{total_demand:,}", help="Sum of all units")
    col2.metric("Avg daily demand", f"{avg_daily:,}")
    col3.metric("Promo lift", f"+{promo_lift:.1f}%",
                delta=f"{promo_lift:.1f}%")
    col4.metric("Missing values", f"{missing_pct}%",
                delta=f"-{missing_pct}%" if missing_pct > 2 else None,
                delta_color="inverse")
    col5.metric("Stockout rate", f"{stockout_pct}%")


# ── Time-series chart ─────────────────────────────────────────
def render_demand_chart(df: pd.DataFrame, freq: str = "W"):
    agg = (df.groupby("date")["demand"]
             .sum()
             .resample(freq)
             .sum()
             .reset_index())
    agg.columns = ["date", "demand"]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=agg["date"], y=agg["demand"],
        fill="tozeroy", mode="lines",
        line=dict(color="#185FA5", width=2),
        fillcolor="rgba(24,95,165,0.12)",
        name="Total demand"
    ))
    fig.update_layout(
        title=f"Demand trend ({freq} aggregation)",
        xaxis_title="Date", yaxis_title="Units",
        height=350, margin=dict(l=20, r=20, t=40, b=20),
        hovermode="x unified",
    )
    st.plotly_chart(fig, use_container_width=True)


# ── Category breakdown ────────────────────────────────────────
def render_category_breakdown(df: pd.DataFrame):
    col1, col2 = st.columns(2)

    cat_agg = df.groupby("category")["demand"].sum().reset_index()
    fig1 = px.pie(cat_agg, values="demand", names="category",
                   title="Demand by category",
                   color_discrete_sequence=["#185FA5","#0F6E56","#854F0B","#993C1D","#534AB7"])
    fig1.update_traces(textposition="inside", textinfo="percent+label")
    col1.plotly_chart(fig1, use_container_width=True)

    store_agg = df.groupby("store_id")["demand"].mean().reset_index()
    fig2 = px.bar(store_agg, x="store_id", y="demand",
                   title="Avg daily demand per store",
                   color="demand",
                   color_continuous_scale="Blues")
    fig2.update_layout(height=350, margin=dict(l=10, r=10, t=40, b=10))
    col2.plotly_chart(fig2, use_container_width=True)


# ── Seasonality heatmap ───────────────────────────────────────
def render_seasonality(df: pd.DataFrame):
    df2 = df.copy()
    df2["dow"]   = pd.to_datetime(df2["date"]).dt.day_name()
    df2["month"] = pd.to_datetime(df2["date"]).dt.strftime("%b")
    df2["month_num"] = pd.to_datetime(df2["date"]).dt.month

    pivot = (df2.groupby(["month_num","month","dow"])["demand"]
               .mean()
               .reset_index()
               .groupby(["month","dow"])["demand"].mean()
               .unstack("dow"))

    day_order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
    pivot = pivot[[d for d in day_order if d in pivot.columns]]

    fig = px.imshow(
        pivot,
        title="Avg demand — month × day of week",
        color_continuous_scale="YlOrRd",
        aspect="auto",
        labels=dict(color="Avg demand"),
    )
    fig.update_layout(height=350, margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(fig, use_container_width=True)


# ── Drift monitoring section ──────────────────────────────────
def render_drift_section(df: pd.DataFrame):
    st.subheader("Data Drift Monitor")
    col1, col2 = st.columns([1, 2])

    split = "2023-07-01"
    ref = df[df["date"] < split]["demand"].dropna()
    cur = df[df["date"] >= split]["demand"].dropna()

    from scipy import stats
    ks_stat, p_val = stats.ks_2samp(ref, cur)

    # PSI quick calc
    bins = np.percentile(ref, np.linspace(0, 100, 11))
    bins[0] -= 1; bins[-1] += 1
    rp = np.histogram(ref, bins)[0] / len(ref)
    cp = np.histogram(cur, bins)[0] / len(cur)
    rp = np.where(rp==0, 1e-4, rp)
    cp = np.where(cp==0, 1e-4, cp)
    psi = float(((rp - cp) * np.log(rp / cp)).sum())

    status_color = "🔴" if psi >= 0.2 else "🟡" if psi >= 0.1 else "🟢"
    col1.metric("PSI Score", f"{psi:.4f}", help="< 0.1 OK | 0.1-0.2 Warn | >0.2 Alert")
    col1.metric("KS p-value", f"{p_val:.4f}",
                delta="Drift detected" if p_val < 0.05 else "No drift",
                delta_color="inverse" if p_val < 0.05 else "normal")
    col1.markdown(f"### Status: {status_color}")

    fig = go.Figure()
    fig.add_trace(go.Histogram(x=ref.clip(upper=ref.quantile(0.99)),
                                name="Reference", opacity=0.6,
                                marker_color="#185FA5", histnorm="probability density"))
    fig.add_trace(go.Histogram(x=cur.clip(upper=cur.quantile(0.99)),
                                name="Current", opacity=0.6,
                                marker_color="#993C1D", histnorm="probability density"))
    fig.update_layout(barmode="overlay", title="Demand distribution shift",
                       height=300, margin=dict(l=10,r=10,t=40,b=10))
    col2.plotly_chart(fig, use_container_width=True)


# ── Main app ──────────────────────────────────────────────────
def main():
    st.title("📦 Retail Demand Forecasting Dashboard")
    st.caption("End-to-end ML system — Phase 7 · Production-ready")

    df = load_data()
    df["date"] = pd.to_datetime(df["date"])

    sel_store, sel_sku, sel_cat, date_range = render_sidebar(df)

    # Apply filters
    mask = (
        df["store_id"].isin(sel_store) &
        df["sku_id"].isin(sel_sku) &
        df["category"].isin(sel_cat)
    )
    if len(date_range) == 2:
        mask &= (df["date"].dt.date >= date_range[0]) & \
                (df["date"].dt.date <= date_range[1])
    df_filtered = df[mask]

    if df_filtered.empty:
        st.warning("No data matches the selected filters.")
        return

    # ── KPIs ──
    st.subheader("Key Performance Indicators")
    render_kpis(df_filtered)
    st.divider()

    # ── Demand trend ──
    freq = st.radio("Aggregation", ["D","W","ME"], horizontal=True,
                     format_func=lambda x: {"D":"Daily","W":"Weekly","ME":"Monthly"}[x])
    render_demand_chart(df_filtered, freq)
    st.divider()

    # ── Breakdown ──
    st.subheader("Category & Store Breakdown")
    render_category_breakdown(df_filtered)
    st.divider()

    # ── Seasonality ──
    st.subheader("Seasonality Patterns")
    render_seasonality(df_filtered)
    st.divider()

    # ── Drift ──
    render_drift_section(df)

    # ── Model comparison table (from saved CSV) ──
    st.divider()
    st.subheader("Model Comparison")
    comp_path = OUTPUT_DIR / "evaluation" / "model_comparison.csv"
    if comp_path.exists():
        comp = pd.read_csv(comp_path, index_col=0)
        st.dataframe(comp.style.highlight_min(color="#d4edda", axis=0)
                               .format("{:.3f}"), use_container_width=True)
    else:
        st.info("Run the full pipeline first to generate model comparison results.")

    st.caption("Built with Python · scikit-learn · XGBoost · TensorFlow · Streamlit")


if __name__ == "__main__":
    main()
