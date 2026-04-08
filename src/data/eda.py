"""
=============================================================
 Phase 1 — Exploratory Data Analysis (EDA)
 Cloud mapping: This notebook runs as an AWS Glue / Azure
   Synapse / Databricks notebook in production, outputting
   HTML reports to S3 / ADLS / GCS for stakeholder review.
=============================================================
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # non-interactive backend for server/cloud
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import FuncFormatter
import warnings, logging, sys
from pathlib import Path
from scipy import stats
from statsmodels.tsa.seasonal import STL

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from configs.config import OUTPUT_DIR, FEATURE_CONFIG

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ── Style ─────────────────────────────────────────────────────
PALETTE  = ["#185FA5", "#0F6E56", "#854F0B", "#993C1D", "#534AB7"]
plt.rcParams.update({
    "figure.facecolor" : "white",
    "axes.facecolor"   : "white",
    "axes.grid"        : True,
    "grid.alpha"       : 0.3,
    "font.family"      : "sans-serif",
    "axes.spines.top"  : False,
    "axes.spines.right": False,
})


# ─────────────────────────────────────────────────────────────
# 1. Dataset overview
# ─────────────────────────────────────────────────────────────
def compute_data_overview(df: pd.DataFrame) -> dict:
    """Return a structured summary of the dataset."""
    num_df = df.select_dtypes(include="number")
    overview = {
        "shape"             : df.shape,
        "date_range"        : (df["date"].min(), df["date"].max()),
        "n_stores"          : df["store_id"].nunique(),
        "n_skus"            : df["sku_id"].nunique(),
        "n_categories"      : df["category"].nunique(),
        "missing_pct"       : (df.isna().sum() / len(df) * 100).round(2).to_dict(),
        "demand_stats"      : df["demand"].describe().round(2).to_dict(),
        "outlier_pct"       : _count_outliers(df["demand"].dropna()),
        "stockout_pct"      : (df["stockout"].sum() / len(df) * 100).round(2),
        "promo_pct"         : (df["promotion"].sum() / len(df) * 100).round(2),
        "numeric_corr"      : num_df.corr()["demand"].drop("demand").round(3).to_dict(),
    }
    log.info("=== DATA OVERVIEW ===")
    for k, v in overview.items():
        log.info(f"  {k}: {v}")
    return overview


def _count_outliers(series: pd.Series) -> float:
    """IQR-based outlier percentage."""
    q1, q3 = series.quantile([0.25, 0.75])
    iqr = q3 - q1
    mask = (series < q1 - 1.5 * iqr) | (series > q3 + 1.5 * iqr)
    return round(mask.mean() * 100, 2)


# ─────────────────────────────────────────────────────────────
# 2. Time-series decomposition (STL)
# ─────────────────────────────────────────────────────────────
def plot_stl_decomposition(df: pd.DataFrame, store_id: int = 1,
                            sku_id: int = 1) -> plt.Figure:
    """
    STL decomposition: separates demand into Trend + Season + Residual.
    STL (Seasonal-Trend decomposition using LOESS) is more robust
    than classical decomposition — handles non-constant seasonality.
    """
    series = (
        df[(df["store_id"] == store_id) & (df["sku_id"] == sku_id)]
        .set_index("date")["demand"]
        .interpolate()        # fill missing before decomposition
    )

    stl    = STL(series, period=7, robust=True)
    result = stl.fit()

    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
    fig.suptitle(f"STL Decomposition — Store {store_id} · SKU {sku_id}",
                 fontsize=14, fontweight="bold", y=1.01)

    components = [
        (series,         "Observed demand",  PALETTE[0]),
        (result.trend,   "Trend",            PALETTE[1]),
        (result.seasonal,"Weekly seasonality",PALETTE[2]),
        (result.resid,   "Residual",         PALETTE[3]),
    ]
    for ax, (data, title, color) in zip(axes, components):
        ax.plot(data, color=color, linewidth=1.0, alpha=0.85)
        ax.set_ylabel(title, fontsize=10)
        ax.fill_between(data.index, data, alpha=0.08, color=color)

    axes[-1].set_xlabel("Date", fontsize=10)
    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────
# 3. Demand distribution analysis
# ─────────────────────────────────────────────────────────────
def plot_demand_distributions(df: pd.DataFrame) -> plt.Figure:
    """
    Distribution plots per category:
      - Histogram + KDE
      - Box plots by category
      - Demand vs price scatter
      - Promo vs non-promo comparison
    """
    fig = plt.figure(figsize=(16, 12))
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.3)
    fig.suptitle("Demand Distribution Analysis", fontsize=15, fontweight="bold")

    clean = df.dropna(subset=["demand"])

    # Panel 1: Overall demand histogram + KDE
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.hist(clean["demand"], bins=60, color=PALETTE[0], alpha=0.65, edgecolor="white")
    ax2 = ax1.twinx()
    kde_x = np.linspace(clean["demand"].min(), clean["demand"].quantile(0.99), 300)
    kde   = stats.gaussian_kde(clean["demand"].clip(upper=clean["demand"].quantile(0.99)))
    ax2.plot(kde_x, kde(kde_x), color=PALETTE[3], linewidth=2)
    ax2.set_ylabel("Density", fontsize=9)
    ax1.set_xlabel("Units demand", fontsize=9)
    ax1.set_title("Overall demand distribution", fontsize=11)
    ax1.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:,.0f}"))

    # Panel 2: Box plot by category
    ax3 = fig.add_subplot(gs[0, 1])
    categories = clean["category"].unique()
    data_by_cat = [
        clean[clean["category"] == c]["demand"].clip(upper=clean["demand"].quantile(0.99))
        for c in categories
    ]
    bp = ax3.boxplot(data_by_cat, patch_artist=True, notch=True,
                     medianprops={"color": "white", "linewidth": 2})
    for patch, color in zip(bp["boxes"], PALETTE):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax3.set_xticklabels(categories, rotation=20, ha="right", fontsize=9)
    ax3.set_title("Demand by product category", fontsize=11)
    ax3.set_ylabel("Units", fontsize=9)

    # Panel 3: Demand vs price scatter
    ax4 = fig.add_subplot(gs[1, 0])
    sample = clean.sample(min(5000, len(clean)), random_state=42)
    ax4.scatter(sample["price"], sample["demand"], alpha=0.2, s=8,
                color=PALETTE[0])
    # Trend line
    m, b, r, p, _ = stats.linregress(sample["price"], sample["demand"])
    x_line = np.linspace(sample["price"].min(), sample["price"].max(), 100)
    ax4.plot(x_line, m * x_line + b, color=PALETTE[3], linewidth=2,
             label=f"r = {r:.2f}")
    ax4.set_xlabel("Price (₹)", fontsize=9)
    ax4.set_ylabel("Demand", fontsize=9)
    ax4.set_title("Demand vs price (price elasticity)", fontsize=11)
    ax4.legend(fontsize=9)

    # Panel 4: Promo vs non-promo
    ax5 = fig.add_subplot(gs[1, 1])
    promo_demand     = clean[clean["promotion"] == 1]["demand"]
    non_promo_demand = clean[clean["promotion"] == 0]["demand"]
    ax5.hist(non_promo_demand.clip(upper=clean["demand"].quantile(0.99)),
             bins=50, alpha=0.6, color=PALETTE[0], label="No promotion",
             density=True)
    ax5.hist(promo_demand.clip(upper=clean["demand"].quantile(0.99)),
             bins=50, alpha=0.6, color=PALETTE[2], label="Promotion",
             density=True)
    ax5.axvline(non_promo_demand.median(), color=PALETTE[0], linestyle="--", linewidth=1.5)
    ax5.axvline(promo_demand.median(),     color=PALETTE[2], linestyle="--", linewidth=1.5)
    ax5.set_title("Promotion effect on demand", fontsize=11)
    ax5.set_xlabel("Demand", fontsize=9)
    ax5.legend(fontsize=9)

    return fig


# ─────────────────────────────────────────────────────────────
# 4. Seasonality patterns
# ─────────────────────────────────────────────────────────────
def plot_seasonality_patterns(df: pd.DataFrame) -> plt.Figure:
    """
    Multi-panel seasonality analysis:
      - Average demand by day of week
      - Average demand by month
      - Average demand by week of year
      - Year-over-year comparison
    """
    clean = df.dropna(subset=["demand"])
    clean = clean[clean["stockout"] == 0]

    daily_agg = (
        clean.groupby("date")["demand"]
        .mean()
        .reset_index()
        .rename(columns={"demand": "avg_demand"})
    )
    daily_agg["dayofweek"] = daily_agg["date"].dt.day_name()
    daily_agg["month"]     = daily_agg["date"].dt.month
    daily_agg["week"]      = daily_agg["date"].dt.isocalendar().week.astype(int)
    daily_agg["year"]      = daily_agg["date"].dt.year
    daily_agg["monthname"] = daily_agg["date"].dt.strftime("%b")

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle("Seasonality Pattern Analysis", fontsize=15, fontweight="bold")

    day_order   = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
    month_order = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]

    # Weekly seasonality
    ax = axes[0, 0]
    dow = (daily_agg.groupby("dayofweek")["avg_demand"].mean()
           .reindex(day_order))
    bars = ax.bar(range(7), dow.values, color=PALETTE, alpha=0.8)
    ax.set_xticks(range(7))
    ax.set_xticklabels([d[:3] for d in day_order], fontsize=9)
    ax.set_title("Avg demand by day of week", fontsize=11)
    ax.set_ylabel("Average units", fontsize=9)
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f"{bar.get_height():.1f}", ha="center", va="bottom", fontsize=8)

    # Monthly seasonality
    ax = axes[0, 1]
    mom = daily_agg.groupby(["month","monthname"])["avg_demand"].mean().reset_index()
    mom = mom.sort_values("month")
    ax.plot(mom["month"], mom["avg_demand"], marker="o", color=PALETTE[0],
            linewidth=2, markersize=7)
    ax.fill_between(mom["month"], mom["avg_demand"], alpha=0.15, color=PALETTE[0])
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(month_order, fontsize=9)
    ax.set_title("Avg demand by month", fontsize=11)
    ax.set_ylabel("Average units", fontsize=9)

    # Week of year
    ax = axes[1, 0]
    wow = daily_agg.groupby("week")["avg_demand"].mean()
    ax.plot(wow.index, wow.values, color=PALETTE[2], linewidth=1.5, alpha=0.8)
    ax.fill_between(wow.index, wow.values, alpha=0.12, color=PALETTE[2])
    ax.axvspan(48, 52, alpha=0.15, color=PALETTE[3], label="Holiday season")
    ax.legend(fontsize=9)
    ax.set_xlabel("Week of year", fontsize=9)
    ax.set_title("Avg demand by week", fontsize=11)
    ax.set_ylabel("Average units", fontsize=9)

    # Year-over-year
    ax = axes[1, 1]
    for i, year in enumerate(sorted(daily_agg["year"].unique())):
        ydata = (daily_agg[daily_agg["year"] == year]
                 .groupby("week")["avg_demand"].mean())
        ax.plot(ydata.index, ydata.values, label=str(year),
                color=PALETTE[i % len(PALETTE)], linewidth=1.8, alpha=0.85)
    ax.legend(fontsize=9, title="Year")
    ax.set_xlabel("Week of year", fontsize=9)
    ax.set_title("Year-over-year demand comparison", fontsize=11)
    ax.set_ylabel("Average units", fontsize=9)

    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────
# 5. Store & SKU level heatmap
# ─────────────────────────────────────────────────────────────
def plot_store_sku_heatmap(df: pd.DataFrame) -> plt.Figure:
    """
    Heatmap of average demand per store-SKU pair.
    In production this becomes a Quicksight / Power BI drill-down.
    """
    pivot = (
        df.dropna(subset=["demand"])
        .groupby(["store_id", "sku_id"])["demand"]
        .mean()
        .unstack("sku_id")
        .round(1)
    )

    fig, ax = plt.subplots(figsize=(16, 6))
    im = ax.imshow(pivot.values, aspect="auto", cmap="YlOrRd")
    plt.colorbar(im, ax=ax, label="Avg daily demand (units)")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_yticks(range(len(pivot.index)))
    ax.set_xticklabels([f"SKU {c}" for c in pivot.columns], rotation=45, ha="right", fontsize=7)
    ax.set_yticklabels([f"Store {r}" for r in pivot.index], fontsize=9)
    ax.set_title("Average demand heatmap — Store × SKU", fontsize=13, fontweight="bold")
    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────
# 6. Missing value & outlier report
# ─────────────────────────────────────────────────────────────
def plot_data_quality(df: pd.DataFrame) -> plt.Figure:
    """Visualise missing values, outliers, and stockouts."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Data Quality Report", fontsize=14, fontweight="bold")

    # Missing per column
    ax = axes[0]
    miss = df.isna().sum() / len(df) * 100
    miss = miss[miss > 0]
    ax.barh(miss.index, miss.values, color=PALETTE[3], alpha=0.8)
    ax.set_xlabel("Missing (%)", fontsize=9)
    ax.set_title("Missing values per column", fontsize=11)
    for i, v in enumerate(miss.values):
        ax.text(v + 0.01, i, f"{v:.2f}%", va="center", fontsize=9)

    # Outlier detection (IQR)
    ax = axes[1]
    clean = df.dropna(subset=["demand"])
    q1, q3 = clean["demand"].quantile([0.25, 0.75])
    iqr    = q3 - q1
    normal   = clean[(clean["demand"] >= q1 - 1.5*iqr) & (clean["demand"] <= q3 + 1.5*iqr)]["demand"]
    outliers = clean[(clean["demand"] <  q1 - 1.5*iqr) | (clean["demand"] >  q3 + 1.5*iqr)]["demand"]
    ax.hist(normal,   bins=60, color=PALETTE[0], alpha=0.7, label=f"Normal ({len(normal):,})",   density=True)
    ax.hist(outliers, bins=30, color=PALETTE[3], alpha=0.8, label=f"Outliers ({len(outliers):,})", density=True)
    ax.legend(fontsize=9)
    ax.set_title(f"Outlier detection (IQR)\n{_count_outliers(clean['demand']):.2f}% outliers", fontsize=11)
    ax.set_xlabel("Demand", fontsize=9)

    # Stockout & promo breakdown
    ax = axes[2]
    labels  = ["Normal", "Stockout", "Promotion"]
    n_stock = df["stockout"].sum()
    n_promo = df["promotion"].sum()
    n_norm  = len(df) - n_stock - n_promo
    sizes   = [n_norm, n_stock, n_promo]
    wedges, texts, autotexts = ax.pie(
        sizes, labels=labels, autopct="%1.1f%%",
        colors=[PALETTE[0], PALETTE[3], PALETTE[2]],
        startangle=140, wedgeprops={"linewidth": 1, "edgecolor": "white"}
    )
    for t in autotexts:
        t.set_fontsize(9)
    ax.set_title("Row classification", fontsize=11)

    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────
# 7. Correlation & feature importance preview
# ─────────────────────────────────────────────────────────────
def plot_correlation_matrix(df: pd.DataFrame) -> plt.Figure:
    """Correlation heatmap of numeric features vs demand."""
    num_cols = ["demand", "price", "promotion", "stockout"]
    corr     = df[num_cols].dropna().corr()

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(corr.values, vmin=-1, vmax=1, cmap="RdBu_r")
    plt.colorbar(im, ax=ax)
    ax.set_xticks(range(len(num_cols)))
    ax.set_yticks(range(len(num_cols)))
    ax.set_xticklabels(num_cols, rotation=30, ha="right", fontsize=10)
    ax.set_yticklabels(num_cols, fontsize=10)
    for i in range(len(num_cols)):
        for j in range(len(num_cols)):
            ax.text(j, i, f"{corr.values[i,j]:.2f}",
                    ha="center", va="center", fontsize=11,
                    color="white" if abs(corr.values[i,j]) > 0.5 else "black")
    ax.set_title("Correlation matrix", fontsize=13, fontweight="bold")
    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────
# Master EDA runner
# ─────────────────────────────────────────────────────────────
def run_full_eda(df: pd.DataFrame, save_plots: bool = True) -> dict:
    """
    Run complete EDA pipeline and optionally save all plots.
    Returns dict of matplotlib figures for further use.
    """
    log.info("=" * 60)
    log.info("STARTING FULL EDA PIPELINE")
    log.info("=" * 60)

    overview = compute_data_overview(df)

    figures = {}

    log.info("Generating STL decomposition...")
    figures["stl"]          = plot_stl_decomposition(df, store_id=1, sku_id=1)

    log.info("Generating demand distributions...")
    figures["distributions"] = plot_demand_distributions(df)

    log.info("Generating seasonality patterns...")
    figures["seasonality"]  = plot_seasonality_patterns(df)

    log.info("Generating store-SKU heatmap...")
    figures["heatmap"]      = plot_store_sku_heatmap(df)

    log.info("Generating data quality report...")
    figures["data_quality"] = plot_data_quality(df)

    log.info("Generating correlation matrix...")
    figures["correlation"]  = plot_correlation_matrix(df)

    if save_plots:
        eda_out = OUTPUT_DIR / "eda"
        eda_out.mkdir(exist_ok=True)
        for name, fig in figures.items():
            path = eda_out / f"{name}.png"
            fig.savefig(path, dpi=150, bbox_inches="tight")
            log.info(f"  Saved → {path}")
            plt.close(fig)

    log.info("EDA complete.")
    return {"overview": overview, "figures": figures}


if __name__ == "__main__":
    from src.data.data_generator import generate_retail_dataset
    df = generate_retail_dataset(save=False)
    run_full_eda(df, save_plots=True)
