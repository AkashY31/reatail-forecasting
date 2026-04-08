"""
=============================================================
 Phase 2 — Preprocessing Pipeline + Feature Engineering
 Cloud mapping: AWS Glue / Azure Databricks / GCP Dataflow
=============================================================
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
import logging, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from configs.config import FEATURE_CONFIG, DATA_DIR

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# Step 1 — Missing value imputer
# ─────────────────────────────────────────────────────────────
class DemandImputer(BaseEstimator, TransformerMixin):
    """
    Strategy:
      - Stockout rows → keep 0 (real signal, not missing)
      - Remaining NaN → interpolate within each store-SKU group
        using time-aware linear interpolation, then forward/back fill
    Cloud note: in Spark, use Window functions instead of groupby.
    """
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()
        def _impute_group(g):
            g = g.sort_values("date")
            # Don't impute stockout rows
            mask = g["stockout"] == 0
            g.loc[mask, "demand"] = (
                g.loc[mask, "demand"]
                .interpolate(method="linear", limit_direction="both")
            )
            g["demand"] = g["demand"].ffill().bfill().fillna(0)
            return g

        df = (df.groupby(["store_id", "sku_id"], group_keys=False)
                .apply(_impute_group))
        log.info(f"After imputation — missing: {df['demand'].isna().sum()}")
        return df


# ─────────────────────────────────────────────────────────────
# Step 2 — Outlier handler
# ─────────────────────────────────────────────────────────────
class OutlierHandler(BaseEstimator, TransformerMixin):
    """
    Winsorize demand at (1%, 99%) per store-SKU.
    Preserves signal shape while removing extreme data-entry errors.
    Alternative: replace with rolling median (more conservative).
    """
    def __init__(self, lower=0.01, upper=0.99):
        self.lower = lower
        self.upper = upper
        self.bounds_ = {}

    def fit(self, X, y=None):
        for (sid, kid), grp in X.groupby(["store_id", "sku_id"]):
            d = grp["demand"].dropna()
            self.bounds_[(sid, kid)] = (d.quantile(self.lower),
                                         d.quantile(self.upper))
        return self

    def transform(self, X):
        df = X.copy()
        for (sid, kid), (lo, hi) in self.bounds_.items():
            mask = (df["store_id"] == sid) & (df["sku_id"] == kid)
            df.loc[mask, "demand"] = df.loc[mask, "demand"].clip(lo, hi)
        return df


# ─────────────────────────────────────────────────────────────
# Step 3 — Feature engineering
# ─────────────────────────────────────────────────────────────
class FeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Builds the full feature matrix from cleaned demand signal.

    Features created:
      Calendar   : dow, month, quarter, is_weekend, is_month_start/end
      Lag        : demand_lag_1/7/14/28/56
      Rolling    : rolling_mean/std_7/14/28
      Fourier    : sin/cos pairs for weekly + annual seasonality
      Exogenous  : price, promotion, stockout, category_encoded
      Interaction: promo × lag7, weekend × rolling7
    """
    def __init__(self, config: dict = None):
        self.config = config or FEATURE_CONFIG
        self._le    = LabelEncoder()

    def fit(self, X, y=None):
        self._le.fit(X["category"])
        return self

    def transform(self, X):
        df = X.copy().sort_values(["store_id", "sku_id", "date"])

        # ── Calendar features ────────────────────────────────
        df["dow"]            = df["date"].dt.dayofweek
        df["month"]          = df["date"].dt.month
        df["quarter"]        = df["date"].dt.quarter
        df["day_of_year"]    = df["date"].dt.dayofyear
        df["week_of_year"]   = df["date"].dt.isocalendar().week.astype(int)
        df["is_weekend"]     = (df["dow"] >= 5).astype(int)
        df["is_month_start"] = df["date"].dt.is_month_start.astype(int)
        df["is_month_end"]   = df["date"].dt.is_month_end.astype(int)
        df["year"]           = df["date"].dt.year

        # ── Fourier terms (weekly + annual seasonality) ──────
        for k in range(1, self.config["fourier_terms"] + 1):
            df[f"sin_week_{k}"] = np.sin(2 * np.pi * k * df["dow"] / 7)
            df[f"cos_week_{k}"] = np.cos(2 * np.pi * k * df["dow"] / 7)
            df[f"sin_year_{k}"] = np.sin(2 * np.pi * k * df["day_of_year"] / 365.25)
            df[f"cos_year_{k}"] = np.cos(2 * np.pi * k * df["day_of_year"] / 365.25)

        # ── Lag + rolling features (per store-SKU group) ─────
        def _group_features(g):
            g = g.sort_values("date")
            for lag in self.config["lag_days"]:
                g[f"demand_lag_{lag}"] = g["demand"].shift(lag)
            for w in self.config["rolling_windows"]:
                g[f"rolling_mean_{w}"] = g["demand"].shift(1).rolling(w).mean()
                g[f"rolling_std_{w}"]  = g["demand"].shift(1).rolling(w).std()
                g[f"rolling_max_{w}"]  = g["demand"].shift(1).rolling(w).max()
            # Trend: demand growth rate over 28 days
            g["demand_trend_28"] = (
                g["demand"].shift(1).rolling(28).mean() -
                g["demand"].shift(29).rolling(28).mean()
            )
            return g

        df = (df.groupby(["store_id", "sku_id"], group_keys=False)
                .apply(_group_features))

        # ── Encode categoricals ───────────────────────────────
        df["category_enc"] = self._le.transform(df["category"])

        # ── Interaction features ──────────────────────────────
        df["promo_x_lag7"]     = df["promotion"] * df.get("demand_lag_7", 0)
        df["weekend_x_roll7"]  = df["is_weekend"] * df.get("rolling_mean_7", 0)

        log.info(f"Feature matrix: {df.shape[1]} columns, {len(df):,} rows")
        return df


# ─────────────────────────────────────────────────────────────
# Full preprocessing pipeline
# ─────────────────────────────────────────────────────────────
def build_preprocessing_pipeline() -> Pipeline:
    return Pipeline([
        ("imputer",  DemandImputer()),
        ("outlier",  OutlierHandler(lower=0.01, upper=0.99)),
        ("features", FeatureEngineer()),
    ])


def get_feature_columns(df: pd.DataFrame) -> list:
    """Return all ML-ready feature columns (exclude target + identifiers)."""
    exclude = {"date", "demand", "store_id", "sku_id", "category", "region"}
    return [c for c in df.columns if c not in exclude]


def temporal_train_test_split(df: pd.DataFrame,
                               test_date: str,
                               val_date:  str,
                               feature_cols: list,
                               target: str = "demand"):
    """
    Temporal split — NO data leakage.
    Always split by date, never random shuffle on time-series data.
    """
    df = df.dropna(subset=feature_cols + [target])
    train = df[df["date"] <  val_date]
    val   = df[(df["date"] >= val_date) & (df["date"] < test_date)]
    test  = df[df["date"] >= test_date]

    X_tr, y_tr = train[feature_cols], train[target]
    X_va, y_va = val[feature_cols],   val[target]
    X_te, y_te = test[feature_cols],  test[target]

    log.info(f"Train: {len(X_tr):,}  Val: {len(X_va):,}  Test: {len(X_te):,}")
    return (X_tr, y_tr), (X_va, y_va), (X_te, y_te)


if __name__ == "__main__":
    from src.data.data_generator import generate_retail_dataset
    df  = generate_retail_dataset(save=False)
    pipe = build_preprocessing_pipeline()
    pipe.fit(df)
    df_proc = pipe.transform(df)
    feat_cols = get_feature_columns(df_proc)
    print(f"Features ({len(feat_cols)}): {feat_cols[:10]} ...")
    df_proc.to_parquet(DATA_DIR / "processed" / "features.parquet", index=False)
    print("Saved processed features.")
