# C:\Users\melik\AQRE\src\models\train_models.py
import os, json, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import log_loss

# local utils
def brier_multiclass(y_true_int: np.ndarray, proba: np.ndarray, n_classes: int) -> float:
    Y = np.eye(n_classes)[y_true_int]
    return float(np.mean(np.sum((Y - proba) ** 2, axis=1)))

def expected_calibration_error(maxp: np.ndarray, correct: np.ndarray, n_bins: int = 15) -> float:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.digitize(maxp, bins) - 1
    ece = 0.0
    n = len(maxp)
    for b in range(n_bins):
        m = (idx == b)
        if not np.any(m):
            continue
        conf = float(np.mean(maxp[m]))
        acc  = float(np.mean(correct[m]))
        w    = float(np.sum(m) / n)
        ece += abs(acc - conf) * w
    return float(ece)

def robust_feature_drop(df: pd.DataFrame, y_name: str) -> pd.DataFrame:
    X = df.drop(columns=[y_name])
    keep = []
    for c in X.columns:
        s = X[c]
        # tarih/kimlik olabilecek kolonları direkt at
        if c.lower() in {"fixture_date_utc", "match_id", "fixture_id", "id"}:
            continue
        # çok eksik / sonsuz / sabit
        try:
            arr = s.to_numpy(dtype=float, copy=False)
        except Exception:
            # sayısal olmayan kolonları ele
            continue
        if np.mean(pd.isna(arr)) > 0.20:      # >%20 NaN
            continue
        if not np.isfinite(arr).all():
            continue
        if pd.Series(arr).nunique(dropna=True) <= 1:
            continue
        keep.append(c)
    cols = keep + [y_name]
    return df[cols]

# config (portable)
from src.config import PROCESSED_DATA_DIR, MODELS_DIR, REPORTS_DIR
DATA_PQ = PROCESSED_DATA_DIR / "features.parquet"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

def main():
    print(f"[RUN] Using file: {__file__}")

    if not DATA_PQ.exists():
        raise FileNotFoundError(f"Features parquet yok: {DATA_PQ}")

    df = pd.read_parquet(DATA_PQ)
    target = "match_outcome"
    if target not in df.columns:
        raise ValueError(f"'{target}' kolonu yok.")

    df = robust_feature_drop(df, target)
    y = df[target].astype(str).to_numpy()
    X = df.drop(columns=[target])

    le = LabelEncoder().fit(y)
    y_int = le.transform(y)
    classes = list(le.classes_)
    K = len(classes)

    X_np = X.to_numpy(dtype=float, copy=False)

    # aynı oran: 258 → 172/86
    X_tr, X_te, y_tr, y_te = train_test_split(
        X_np, y_int, test_size=86, random_state=42, stratify=y_int
    )

    clf = GradientBoostingClassifier(random_state=42)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        clf.fit(X_tr, y_tr)

    proba = clf.predict_proba(X_te)
    ll = float(log_loss(y_te, proba, labels=range(K)))
    brier = brier_multiclass(y_te, proba, K)
    y_pred = np.argmax(proba, axis=1)
    acc = float(np.mean(y_pred == y_te))
    ece = expected_calibration_error(proba.max(axis=1), (y_pred == y_te).astype(int), 15)

    # kayıtlar
    import joblib
    model_path = MODELS_DIR / "gbm_raw.joblib"
    joblib.dump(clf, model_path)

    metrics = {
        "classes": classes,
        "log_loss": ll,
        "brier_multiclass": brier,
        "ece_15bins": ece,
        "accuracy": acc,
    }
    (REPORTS_DIR / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # kısa özet
    print("\n=== SUMMARY ===")
    print(f"Total/Train/Test: {len(y_int)}/{len(y_tr)}/{len(y_te)}")
    print(f"Classes (global): {classes}")
    print(f"\nMetrics:\n  LogLoss           : {ll}\n  Brier (multiclass): {brier}\n  ECE (15 bins)     : {ece}\n  Accuracy          : {acc}")
    print(f"\nModel  : {model_path}")
    print(f"Metrics: {REPORTS_DIR / 'metrics.json'}")

if __name__ == "__main__":
    os.environ.setdefault("PYTHONUTF8", "1")
    main()
