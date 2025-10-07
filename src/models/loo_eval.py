# C:\Users\melik\AQRE\src\models\loo_eval.py
# Leave-One-Out with class-weight, epsilon smoothing and optional per-fold calibration

import json, os, warnings
from pathlib import Path
from typing import Tuple, List
import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import log_loss
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.utils.class_weight import compute_class_weight

from src.config import PROCESSED_DATA_DIR, REPORTS_DIR

def log(*a, p="[RUN]"):
    print(p, *a)

TARGET_COL = "match_outcome"
EPS = float(os.getenv("EPS", "1e-12"))
ECE_BINS = int(os.getenv("ECE_BINS", "15"))
CAL = os.getenv("LOO_CALIBRATE", "1") not in ("0","false","False")
CAL_METHOD = os.getenv("CAL_METHOD", "sigmoid")
CAL_CV = int(os.getenv("CAL_CV", "5"))

def robust_numeric(df: pd.DataFrame, ycol: str) -> pd.DataFrame:
    X = df.drop(columns=[ycol]).select_dtypes(include=[np.number]).copy()
    keep = []
    for c in X.columns:
        s = X[c]
        if s.isna().mean() > 0.20: 
            continue
        if not np.isfinite(s.to_numpy(dtype=float, copy=False)).all():
            continue
        if s.nunique(dropna=True) <= 1:
            continue
        keep.append(c)
    return pd.concat([X[keep], df[[ycol]]], axis=1)

def eps_smooth(P, eps=1e-12):
    P = P + eps
    P /= np.clip(P.sum(axis=1, keepdims=True), 1e-32, None)
    return P

def brier_multiclass(y_true_int, proba, K):
    Y = np.eye(K)[y_true_int]
    return float(np.mean(np.sum((Y - proba)**2, axis=1)))

def ece(maxp, correct, n_bins=15):
    bins = np.linspace(0.0, 1.0, n_bins+1)
    idx = np.digitize(maxp, bins) - 1
    e, n = 0.0, len(maxp)
    for b in range(n_bins):
        m = (idx == b)
        if not np.any(m): 
            continue
        conf, acc = float(np.mean(maxp[m])), float(np.mean(correct[m]))
        w = float(np.sum(m)/n)
        e += abs(acc-conf)*w
    return float(e)

def base():
    return GradientBoostingClassifier(
        n_estimators=400, learning_rate=0.05, max_depth=3, subsample=0.9, random_state=42
    )

def loo_eval(df: pd.DataFrame):
    df = robust_numeric(df, TARGET_COL)
    y_raw = df[TARGET_COL].astype(str).to_numpy()
    X = df.drop(columns=[TARGET_COL]).to_numpy(dtype=float, copy=False)

    le = LabelEncoder().fit(y_raw)
    classes = list(le.classes_)
    y = le.transform(y_raw)
    K, n = len(classes), len(y)

    # class-weight -> örnek ağırlıkları
    cw = compute_class_weight(class_weight="balanced", classes=np.arange(K), y=y)
    cw_map = dict(zip(range(K), cw))

    rows, cal_fail = [], 0
    log(f"Kalan sayısal özellik: {X.shape[1]}")
    log(f"LOO (calibrate={CAL}:{CAL_METHOD}, cv={CAL_CV})")

    for i in range(n):
        mask = np.ones(n, dtype=bool); mask[i] = False
        X_tr, y_tr = X[mask], y[mask]
        X_te = X[~mask].reshape(1, -1)
        sw_tr = np.array([cw_map[k] for k in y_tr], dtype=float)

        clf = base()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            clf.fit(X_tr, y_tr, sample_weight=sw_tr)

        if CAL:
            try:
                try:
                    cal = CalibratedClassifierCV(estimator=clf, method=CAL_METHOD, cv=CAL_CV)
                except TypeError:
                    cal = CalibratedClassifierCV(base_estimator=clf, method=CAL_METHOD, cv=CAL_CV)
                cal.fit(X_tr, y_tr, sample_weight=sw_tr)
                p = cal.predict_proba(X_te)[0]
            except Exception:
                cal_fail += 1
                p = clf.predict_proba(X_te)[0]
        else:
            p = clf.predict_proba(X_te)[0]

        p = eps_smooth(p.reshape(1,-1), EPS)[0]
        row = {"index": i, "y_true_label": classes[y[i]]}
        for j, c in enumerate(classes):
            row[f"proba_{c}"] = float(p[j])
        rows.append(row)

    pred_df = pd.DataFrame(rows)
    P = pred_df[[f"proba_{c}" for c in classes]].to_numpy()
    y_true = le.transform(pred_df["y_true_label"].to_numpy())
    y_hat = np.argmax(P, axis=1)

    summ = {
        "n_samples": int(n),
        "classes": classes,
        "metrics": {
            "log_loss": float(log_loss(y_true, P, labels=list(range(K)))),
            "brier_multiclass": brier_multiclass(y_true, P, K),
            "ece_15bins": ece(P.max(axis=1), (y_hat==y_true).astype(int), n_bins=ECE_BINS),
            "accuracy": float(np.mean(y_hat==y_true)),
        },
        "calibration": {
            "enabled": bool(CAL), "method": CAL_METHOD if CAL else None,
            "cv": CAL_CV if CAL else None, "folds_failed": int(cal_fail)
        },
    }
    return pred_df, summ

def main():
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    fp = PROCESSED_DATA_DIR / "features.parquet"
    df = pd.read_parquet(fp)
    pred_df, summary = loo_eval(df)
    pred_df.to_csv(REPORTS_DIR / "loo_predictions.csv", index=False)
    (REPORTS_DIR / "loo_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    log("LOO written:", str(REPORTS_DIR / "loo_predictions.csv"), str(REPORTS_DIR / "loo_summary.json"), p="[DONE]")

if __name__ == "__main__":
    os.environ.setdefault("PYTHONUTF8", "1"); main()
