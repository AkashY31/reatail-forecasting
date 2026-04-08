"""
=============================================================
Phase 1 — Synthetic Retail Data Generator
Cloud mapping: In production this module is replaced by
AWS Glue / Azure Data Factory / GCP Dataflow
reading from your actual data warehouse.
=============================================================
"""

import numpy as np
import pandas as pd
from pathlib import Path
import sys, logging

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from configs.config import DATA_CONFIG, DATA_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# Helper: realistic demand signal
# ─────────────────────────────────────────────────────────────
def _build_demand_signal(dates: pd.DatetimeIndex, rng: np.random.Generator,
                          base: float, sku_id: int, store_id: int) -> np.ndarray:
    """
    Compose a realistic demand signal from 5 components:
      1. Base level (per SKU)
      2. Long-term trend (slow growth or decline)
      3. Weekly seasonality (weekends spike)
      4. Annual seasonality (holiday peaks)
      5. Promotional spikes + noise
    """
    n = len(dates)
    t = np.arange(n)

    # 1. Base — each SKU has its own popularity
    sku_factor   = rng.uniform(0.3, 3.0)
    store_factor = rng.uniform(0.5, 2.0)
    base_level   = base * sku_factor * store_factor

    # 2. Trend — slight upward or downward drift over years
    trend_slope = rng.uniform(-0.005, 0.015)
    trend       = trend_slope * t

    # 3. Weekly seasonality — weekday index 0=Mon…6=Sun
    weekday     = dates.dayofweek.values
    weekly_amp  = rng.uniform(0.05, 0.25) * base_level
    weekly      = weekly_amp * np.where(weekday >= 5, 1.5, -0.5)

    # 4. Annual seasonality — Fourier series
    day_of_year = dates.dayofyear.values / 365.25
    annual      = np.zeros(n)
    for k in [1, 2, 3]:
        amp_sin = rng.uniform(-0.15, 0.25) * base_level
        amp_cos = rng.uniform(-0.10, 0.20) * base_level
        annual += amp_sin * np.sin(2 * np.pi * k * day_of_year)
        annual += amp_cos * np.cos(2 * np.pi * k * day_of_year)

    # Holiday boosts: Christmas, New Year, Diwali-period, Summer
    holiday_mask = (
        ((dates.month == 12) & (dates.day >= 20)) |
        ((dates.month ==  1) & (dates.day <=  5)) |
        ((dates.month == 10) & (dates.day >= 15) & (dates.day <= 31)) |
        ((dates.month ==  7))
    )
    holiday_boost = np.where(holiday_mask, rng.uniform(0.3, 0.8) * base_level, 0)

    # 5. Random promotions — 8% of days, demand jumps 40–120%
    promo_days  = rng.random(n) < 0.08
    promo_boost = np.where(promo_days, rng.uniform(0.4, 1.2, n) * base_level, 0)

    # Noise — heteroscedastic (more noise when demand is high)
    noise_scale = rng.uniform(0.05, 0.15)
    noise       = rng.normal(0, noise_scale * base_level, n)

    # Compose and clip to non-negative integers
    demand = base_level + trend + weekly + annual + holiday_boost + promo_boost + noise
    demand = np.clip(demand, 0, None).round().astype(int)
    return demand, promo_days.astype(int)


# ─────────────────────────────────────────────────────────────
# Helper: inject real-world data quality issues
# ─────────────────────────────────────────────────────────────
def _inject_data_quality_issues(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """
    Simulate real retail data issues:
      - Missing values (system outages, ETL failures)
      - Outliers (data entry errors, flash sales)
      - Inventory stockouts (demand = 0 even though real demand existed)
    """
    df = df.copy()
    n  = len(df)

    # 1. Random missing (~2% of rows)
    missing_mask = rng.random(n) < 0.02
    df.loc[missing_mask, "demand"] = np.nan

    # 2. Outliers (~0.5%) — extreme spikes or zeroes
    outlier_mask = rng.random(n) < 0.005
    df.loc[outlier_mask, "demand"] = df.loc[outlier_mask, "demand"] * rng.uniform(5, 15, outlier_mask.sum())

    # 3. Stockout periods — random stretches of forced 0 demand
    store_sku_groups = df.groupby(["store_id", "sku_id"])
    for (sid, kid), grp_idx in store_sku_groups.groups.items():
        if rng.random() < 0.15:   # 15% of store-SKU combos have at least one stockout
            stockout_start = rng.integers(0, len(grp_idx) - 15)
            stockout_len   = rng.integers(3, 14)
            s_idx = grp_idx[stockout_start : stockout_start + stockout_len]
            df.loc[s_idx, "demand"]   = 0
            df.loc[s_idx, "stockout"] = 1

    return df


# ─────────────────────────────────────────────────────────────
# Main: generate full dataset
# ─────────────────────────────────────────────────────────────
def generate_retail_dataset(config: dict = DATA_CONFIG,
                             save: bool = True) -> pd.DataFrame:
    """
    Generate a multi-year, multi-store, multi-SKU retail
    transactional dataset with realistic demand patterns.

    Returns
    -------
    pd.DataFrame with columns:
        date, store_id, sku_id, demand, price, promotion,
        stockout, category, region
    """
    rng   = np.random.default_rng(config["random_seed"])
    dates = pd.date_range(config["start_date"], config["end_date"], freq="D")

    categories = ["Electronics", "Apparel", "Grocery", "Home", "Beauty"]
    regions    = ["North", "South", "East", "West", "Central"]

    # Assign static attributes
    sku_meta = {
        kid: {
            "category": categories[kid % len(categories)],
            "base_price": rng.uniform(5, 500),
        }
        for kid in range(1, config["n_skus"] + 1)
    }
    store_meta = {
        sid: {"region": regions[sid % len(regions)]}
        for sid in range(1, config["n_stores"] + 1)
    }

    records = []
    total   = config["n_stores"] * config["n_skus"]
    done    = 0

    for store_id in range(1, config["n_stores"] + 1):
        for sku_id in range(1, config["n_skus"] + 1):
            demand, promo = _build_demand_signal(
                dates, rng, config["base_demand"], sku_id, store_id
            )
            price = sku_meta[sku_id]["base_price"] * rng.uniform(0.8, 1.2, len(dates))

            chunk = pd.DataFrame({
                "date"      : dates,
                "store_id"  : store_id,
                "sku_id"    : sku_id,
                "demand"    : demand,
                "price"     : price.round(2),
                "promotion" : promo,
                "stockout"  : 0,
                "category"  : sku_meta[sku_id]["category"],
                "region"    : store_meta[store_id]["region"],
            })
            records.append(chunk)
            done += 1
            if done % 100 == 0:
                log.info(f"  Generated {done}/{total} store-SKU series")

    df = pd.concat(records, ignore_index=True)
    df = _inject_data_quality_issues(df, rng)
    df["date"] = pd.to_datetime(df["date"])
    df.sort_values(["store_id", "sku_id", "date"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    if save:
        out_path = DATA_DIR / "raw" / "retail_sales.parquet"
        df.to_parquet(out_path, index=False)
        log.info(f"Saved {len(df):,} rows → {out_path}")

    log.info(f"Dataset shape: {df.shape}")
    log.info(f"Date range  : {df['date'].min().date()} → {df['date'].max().date()}")
    log.info(f"Missing pct : {df['demand'].isna().mean():.2%}")
    return df


if __name__ == "__main__":
    df = generate_retail_dataset()
    print(df.head(10).to_string())
    print("\nDtypes:\n", df.dtypes)
    print("\nMissing:\n", df.isna().sum())
