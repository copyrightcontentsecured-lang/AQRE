# src/models/loo_eval.py
# -*- coding: utf-8 -*-
"""
Leave-One-Out (LOO) evaluation that reads central config.
Writes:
  - reports/loo_predictions.csv
  - reports/loo_summary.json
"""

from __future__ import annotations

import json
import os
import sys
import warnings
from pathlib import Path
from typing import List

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import LeaveOneOut

# --- Config (centralized) ----------------------------------------------------
try:
    from src import config as _cfg  # use central config if available
except Exception:  # safe fallback if import path differs in user env
    _cfg = None  # type: ignore

def _get_float_env(name: str, default: str) -> float:
    try:
        return float(os.getenv(name, default))
    except Exception:
        return float(default)

def _get_int_env(name: str, default: str) -> int:
    try:
        return int(os.getenv(name, default))
    except Exception:
        return int(default)

# Pull from config if present; otherwise fall back to env/defaults
EPS        = getattr(_cfg, "EPS",        _get_float_env("EPS", "1e-12"))
ECE_BINS   = getattr(_cfg, "ECE_BINS",   _get_int_env("ECE_BINS", "15"))
CAL        = getattr(_cfg, "LOO_CALIBRATE",
                     os.getenv("LOO_CALIBRATE", "1") not in ("0", "false", "False"))
CAL_METHOD = getattr(_cfg, "CAL_METHOD", os.getenv("CAL_METHOD", "sigmoid"))
CAL_CV     = getattr(_cfg, "CAL_CV",     _get_int_env("CAL_CV", "5"))

# --- Paths -------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]  # repo root
DATA_PROCESSED = ROOT / "data" / "processed" / "features.parquet"
REPORTS_DIR    = ROOT / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# --- Helpers -----------------------------------------------------------------
CLASSES: List[str] = ["A", "D", "H"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}

def brier_multiclass(y_true_idx: np.ndarray, proba: np.ndarray) -> float:
    """
    Multiclass Brier score (mean squared error to one-hot).
    """
    n, k = proba.shape
    y_onehot = np.zeros((n, k), dtype=float)
    y_onehot[np.arange(n), y_true_idx] = 1.0
    return float(np.mean(np.sum((proba - y_onehot) ** 2, axis=1)))

def ece_maxprob(y_true_idx: np.ndarray, proba: np.ndarray, n_bins: int = 15) -> float:
    """
    Expected Calibration Error using max probability per sample.
    """
    conf = proba.max(axis=1)
    pred = proba.argmax(axis=1)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(conf)
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (conf > lo) & (conf <= hi) if i > 0 else (conf >= lo) & (conf <= hi)
        if not np.any(mask):
            continue
        bin_conf = float(np.mean(conf[mask]))
        bin_acc  = float(np.mean((pred[mask] == y_true_idx[mask]).astype(float)))
        ece += (np.sum(mask) / n) * abs(bin_acc - bin_conf)
    return float(ece)

def select_numeric_features(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if target_col in num_cols:
        num_cols.remove(target_col)
    return df[num_cols]

# --- Main LOO ----------------------------------------------------------------
def main() -> int:
    warnings.filterwarnings("ignore")

    if not DATA_PROCESSED.exists():
        print(f"[ERROR] Processed features not found: {DATA_PROCESSED}", file=sys.stderr)
        return 2

    df = pd.read_parquet(DATA_PROCESSED)
    target_col = "match_outcome"
    if target_col not in df.columns:
        print(f"[ERROR] '{target_col}' not in features. Available: {list(df.columns)}", file=sys.stderr)
        return 3

    X = select_numeric_features(df, target_col)
    y = df[target_col].astype(str).values
    y_idx = np.array([CLASS_TO_IDX.get(lbl, -1) for lbl in y], dtype=int)
    if (y_idx < 0).any():
        bad = df.loc[y_idx < 0, target_col].unique().tolist()
        print(f"[ERROR] Unknown labels in data: {bad}", file=sys.stderr)
        return 4

    print(f"[RUN] Kalan sayısal özellik: {X.shape[1]}")
    print(f"[RUN] LOO (calibrate={bool(CAL)}:{CAL_METHOD}, cv={CAL_CV})")

    loo = LeaveOneOut()
    n = len(df)
    proba_out = np.zeros((n, len(CLASSES)), dtype=float)

    # We use the same base model family as training (GBM).
    base_params = dict(
        random_state=42,
        n_estimators=300,
        learning_rate=0.05,
        max_depth=3,
        subsample=1.0
    )

    # LOO loop
    for i, (train_idx, test_idx) in enumerate(loo.split(X), 1):
        X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
        y_tr = y[train_idx]

        model = GradientBoostingClassifier(**base_params)

        if CAL:
            # Calibrate on training fold with inner CV
            clf = CalibratedClassifierCV(estimator=model, method=CAL_METHOD, cv=CAL_CV)
        else:
            clf = model

        clf.fit(X_tr, y_tr)
        p = clf.predict_proba(X_te)[0]  # (k,)
        # Ensure order of classes matches CLASSES
        # Scikit may order classes_ alphabetically; align to our CLASSES.
        cls_order = list(clf.classes_)
        aligned = np.zeros(len(CLASSES), dtype=float)
        for j, c in enumerate(CLASSES):
            try:
                idx_c = cls_order.index(c)
                aligned[j] = p[idx_c]
            except ValueError:
                aligned[j] = 0.0
        # numerical safety
        aligned = np.clip(aligned, EPS, 1.0)
        aligned /= aligned.sum()
        proba_out[test_idx[0], :] = aligned

        if i % 25 == 0 or i == n:
            print(f"  - progress: {i}/{n}", flush=True)

    # Metrics
    y_pred_idx = proba_out.argmax(axis=1)
    y_pred = np.array([CLASSES[i] for i in y_pred_idx])
    acc = float(accuracy_score(y, y_pred))
    ll = float(log_loss(\1))
    brier = brier_multiclass(y_idx, proba_out)
    ece = ece_maxprob(y_idx, proba_out, n_bins=ECE_BINS)

    # Save predictions
    pred_df = pd.DataFrame({
        "index": np.arange(n, dtype=int),
        "y_true_label": y,
        "proba_A": proba_out[:, 0],
        "proba_D": proba_out[:, 1],
        "proba_H": proba_out[:, 2],
    })
    pred_path = REPORTS_DIR / "loo_predictions.csv"
    pred_df.to_csv(pred_path, index=False)

    # Save summary
    summary = {
        "n_samples": int(n),
        "classes": CLASSES,
        "metrics": {
            "log_loss": ll,
            "brier_multiclass": brier,
            "ece_15bins": ece,  # kept key name for backwards-compat with plots
            "accuracy": acc,
        },
        "calibration": {
            "enabled": bool(CAL),
            "method": CAL_METHOD,
            "cv": int(CAL_CV),
            "folds_failed": 0,  # placeholder (no inner fold failures tracked here)
        },
    }
    summ_path = REPORTS_DIR / "loo_summary.json"
    with open(summ_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"[DONE] LOO written: {pred_path.resolve()} {summ_path.resolve()}")
    print(
        "Classes:", CLASSES,
        "\nMetrics:",
        json.dumps(summary["metrics"], indent=2)
    )
    return 0

if __name__ == "__main__":
    raise SystemExit(main())


