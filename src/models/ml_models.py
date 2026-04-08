"""
=============================================================
 Phase 4 — ML & Deep Learning Models
 Models: Linear Regression · Random Forest · XGBoost · LSTM
 HPT   : Optuna (replaces GridSearchCV for efficiency)
 Cloud : SageMaker Automatic Model Tuning / Vertex AI Vizier
=============================================================
"""

import numpy as np
import pandas as pd
import logging, sys, warnings, joblib
from pathlib import Path

from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from configs.config import MODEL_CONFIG, OUTPUT_DIR
from src.evaluation.metrics import compute_metrics

warnings.filterwarnings("ignore")
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# 1. Linear Regression (Ridge)
# ─────────────────────────────────────────────────────────────
def fit_linear(X_train, y_train, X_val, y_val) -> dict:
    """
    Ridge regression as baseline.
    Simple, interpretable, fast to retrain.
    Expected to underfit complex retail patterns.
    """
    scaler  = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_train)
    X_va_sc = scaler.transform(X_val)

    model   = Ridge(alpha=1.0)
    model.fit(X_tr_sc, y_train)
    preds   = model.predict(X_va_sc).clip(0)
    metrics = compute_metrics(y_val, preds)
    log.info(f"Ridge | {metrics}")
    return {"model": model, "scaler": scaler, "metrics": metrics,
            "preds": preds, "name": "Linear Regression"}


# ─────────────────────────────────────────────────────────────
# 2. Random Forest
# ─────────────────────────────────────────────────────────────
def fit_random_forest(X_train, y_train, X_val, y_val,
                      n_trials: int = 15) -> dict:
    """
    Random Forest with Optuna HPT.
    Optuna uses Tree-structured Parzen Estimator (TPE) —
    smarter than grid search, finds good params in fewer trials.
    """
    def objective(trial):
        params = {
            "n_estimators"      : trial.suggest_int("n_estimators", 50, 300),
            "max_depth"         : trial.suggest_int("max_depth", 3, 20),
            "min_samples_split" : trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf"  : trial.suggest_int("min_samples_leaf", 1, 10),
            "max_features"      : trial.suggest_categorical("max_features", ["sqrt","log2",0.5]),
            "n_jobs"            : -1,
            "random_state"      : 42,
        }
        m = RandomForestRegressor(**params)
        m.fit(X_train, y_train)
        return compute_metrics(y_val, m.predict(X_val).clip(0))["rmse"]

    study = optuna.create_study(direction="minimize",
                                 sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best_params = study.best_params
    best_params.update({"n_jobs": -1, "random_state": 42})
    model  = RandomForestRegressor(**best_params)
    model.fit(X_train, y_train)
    preds  = model.predict(X_val).clip(0)
    metrics = compute_metrics(y_val, preds)
    log.info(f"RandomForest best_params={best_params} | {metrics}")

    feat_imp = pd.Series(model.feature_importances_,
                          index=X_train.columns).sort_values(ascending=False)

    return {"model": model, "best_params": best_params, "study": study,
            "metrics": metrics, "preds": preds,
            "feature_importance": feat_imp, "name": "Random Forest"}


# ─────────────────────────────────────────────────────────────
# 3. XGBoost
# ─────────────────────────────────────────────────────────────
def fit_xgboost(X_train, y_train, X_val, y_val,
                n_trials: int = 20) -> dict:
    """
    XGBoost with Optuna HPT + early stopping.
    XGBoost typically dominates on tabular retail data due to:
      - Gradient boosting captures non-linear interactions
      - Built-in regularisation prevents overfitting
      - Native handling of missing values
    """
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval   = xgb.DMatrix(X_val,   label=y_val)

    def objective(trial):
        params = {
            "objective"        : "reg:squarederror",
            "eval_metric"      : "rmse",
            "max_depth"        : trial.suggest_int("max_depth", 3, 10),
            "learning_rate"    : trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "n_estimators"     : trial.suggest_int("n_estimators", 100, 600),
            "subsample"        : trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree" : trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha"        : trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
            "reg_lambda"       : trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
            "min_child_weight" : trial.suggest_int("min_child_weight", 1, 10),
            "seed"             : 42,
            "verbosity"        : 0,
        }
        n_est = params.pop("n_estimators")
        model  = xgb.train(
            params, dtrain,
            num_boost_round=n_est,
            evals=[(dval, "val")],
            early_stopping_rounds=30,
            verbose_eval=False,
        )
        preds = model.predict(dval)
        return compute_metrics(y_val, preds.clip(0))["rmse"]

    study = optuna.create_study(direction="minimize",
                                 sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best = study.best_params
    n_est = best.pop("n_estimators", 200)
    best.update({"objective": "reg:squarederror", "eval_metric": "rmse",
                  "seed": 42, "verbosity": 0})
    model = xgb.train(best, dtrain, num_boost_round=n_est,
                       evals=[(dval, "val")], early_stopping_rounds=30,
                       verbose_eval=False)

    preds   = model.predict(dval).clip(0)
    metrics = compute_metrics(y_val, preds)
    log.info(f"XGBoost best_params={best} | {metrics}")

    feat_imp = pd.Series(model.get_score(importance_type="gain")).sort_values(ascending=False)

    return {"model": model, "best_params": best, "study": study,
            "metrics": metrics, "preds": preds,
            "feature_importance": feat_imp, "name": "XGBoost"}


# ─────────────────────────────────────────────────────────────
# 4. LSTM
# ─────────────────────────────────────────────────────────────
def build_lstm_sequences(series: np.ndarray, window: int = 28):
    """Create (X, y) pairs for LSTM from a 1D demand series."""
    X, y = [], []
    for i in range(window, len(series)):
        X.append(series[i - window : i])
        y.append(series[i])
    return np.array(X)[..., np.newaxis], np.array(y)


def fit_lstm(train_series: np.ndarray, val_series: np.ndarray,
             window: int = 28, epochs: int = 30) -> dict:
    """
    Univariate LSTM for sequence-to-one demand forecasting.
    Architecture: LSTM(64) → Dropout → LSTM(32) → Dense(1)
    In production use TF-Serving / SageMaker TF endpoint.
    """
    try:
        import tensorflow as tf
        from tensorflow import keras
        tf.get_logger().setLevel("ERROR")

        scaler = StandardScaler()
        full   = np.concatenate([train_series, val_series])
        full_s = scaler.fit_transform(full.reshape(-1, 1)).flatten()

        tr_s = full_s[:len(train_series)]
        va_s = full_s[len(train_series):]

        X_tr, y_tr = build_lstm_sequences(tr_s, window)
        X_va, y_va = build_lstm_sequences(va_s, window)

        if len(X_tr) == 0 or len(X_va) == 0:
            raise ValueError("Not enough data for LSTM sequences.")

        model = keras.Sequential([
            keras.layers.LSTM(64, return_sequences=True,
                               input_shape=(window, 1)),
            keras.layers.Dropout(0.2),
            keras.layers.LSTM(32),
            keras.layers.Dropout(0.1),
            keras.layers.Dense(1),
        ])
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3),
                       loss="mse")

        cb = [
            keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(patience=3, factor=0.5),
        ]
        history = model.fit(
            X_tr, y_tr,
            validation_data=(X_va, y_va),
            epochs=epochs, batch_size=64,
            callbacks=cb, verbose=0,
        )

        preds_s = model.predict(X_va, verbose=0).flatten()
        preds   = scaler.inverse_transform(preds_s.reshape(-1, 1)).flatten().clip(0)
        y_true  = scaler.inverse_transform(y_va.reshape(-1, 1)).flatten()
        metrics = compute_metrics(y_true, preds)
        log.info(f"LSTM epochs={len(history.history['loss'])} | {metrics}")

        return {"model": model, "scaler": scaler, "window": window,
                "metrics": metrics, "preds": preds,
                "history": history.history, "name": "LSTM"}

    except ImportError:
        log.warning("TensorFlow not available — skipping LSTM.")
        return {"model": None, "metrics": {"rmse": 9999, "mae": 9999, "mape": 9999},
                "preds": np.zeros(len(val_series)), "name": "LSTM (unavailable)"}


# ─────────────────────────────────────────────────────────────
# Run all ML models
# ─────────────────────────────────────────────────────────────
def run_ml_models(X_train, y_train, X_val, y_val,
                  train_series=None, val_series=None,
                  n_trials: int = 15) -> dict:
    results = {}
    log.info("=== Fitting Linear Regression ===")
    results["Linear Regression"] = fit_linear(X_train, y_train, X_val, y_val)
    log.info("=== Fitting Random Forest ===")
    results["Random Forest"]     = fit_random_forest(X_train, y_train, X_val, y_val, n_trials)
    log.info("=== Fitting XGBoost ===")
    results["XGBoost"]           = fit_xgboost(X_train, y_train, X_val, y_val, n_trials)
    if train_series is not None:
        log.info("=== Fitting LSTM ===")
        results["LSTM"]          = fit_lstm(train_series, val_series)
    return results


def save_models(results: dict, path: Path):
    path.mkdir(parents=True, exist_ok=True)
    for name, r in results.items():
        if r.get("model") is None:
            continue
        fname = name.lower().replace(" ", "_")
        try:
            joblib.dump(r["model"], path / f"{fname}.pkl")
        except Exception:
            try:
                r["model"].save(str(path / f"{fname}.keras"))
            except Exception:
                pass
    log.info(f"Models saved to {path}")
