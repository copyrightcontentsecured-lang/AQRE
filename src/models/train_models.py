# C:\Users\melik\AQRE\src\models\train_models.py
import json, os, warnings
from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import log_loss
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.utils.class_weight import compute_class_weight

from src.config import PROCESSED_DATA_DIR, MODELS_DIR, REPORTS_DIR

def log(*a): print("[RUN]", *a)
def ensure_dirs():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

def robust_numeric(df: pd.DataFrame, ycol: str) -> pd.DataFrame:
    X = df.drop(columns=[ycol]).select_dtypes(include=[np.number]).copy()
    keep = []
    for c in X.columns:
        s = X[c]
        if s.isna().mean() > 0.20:  # çok eksik at
            continue
        if not np.isfinite(s.to_numpy(dtype=float, copy=False)).all():
            continue
        if s.nunique(dropna=True) <= 1:  # sabit kolon
            continue
        keep.append(c)
    return pd.concat([X[keep], df[[ycol]]], axis=1)

def brier_multiclass(y_true_int, proba, K):
    Y = np.eye(K)[y_true_int]
    return float(np.mean(np.sum((Y - proba)**2, axis=1)))

def expected_calibration_error(maxp, correct, n_bins=15):
    bins = np.linspace(0.0, 1.0, n_bins+1)
    idx = np.digitize(maxp, bins) - 1
    ece, n = 0.0, len(maxp)
    for b in range(n_bins):
        m = (idx == b)
        if not np.any(m): 
            continue
        conf, acc = float(np.mean(maxp[m])), float(np.mean(correct[m]))
        w = float(np.sum(m)/n)
        ece += abs(acc-conf)*w
    return float(ece)

def build_model():
    # Güvenli, hızlı ve predict_proba destekli
    return GradientBoostingClassifier(
        n_estimators=400, learning_rate=0.05,
        max_depth=3, subsample=0.9, random_state=42
    )

def main():
    ensure_dirs()
    ycol = "match_outcome"
    feats_path = PROCESSED_DATA_DIR / "features.parquet"
    df = pd.read_parquet(feats_path)
    df = robust_numeric(df, ycol)
    log("numeric feature count:", df.shape[1]-1)

    y_raw = df[ycol].astype(str).to_numpy()
    X = df.drop(columns=[ycol]).to_numpy(dtype=float, copy=False)

    le = LabelEncoder().fit(y_raw)
    classes = list(le.classes_)
    y = le.transform(y_raw)
    K = len(classes)

    # Class weight → örnek ağırlığı
    cw = compute_class_weight(class_weight="balanced", classes=np.arange(K), y=y)
    per_class = dict(zip(range(K), cw))
    sw = np.array([per_class[i] for i in y], dtype=float)

    # Split
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.333, random_state=42)
    tr_idx, te_idx = next(sss.split(X, y))
    X_tr, X_te = X[tr_idx], X[te_idx]
    y_tr, y_te = y[tr_idx], y[te_idx]
    sw_tr = sw[tr_idx]

    clf = build_model()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        clf.fit(X_tr, y_tr, sample_weight=sw_tr)

    # Opsiyonel kalibrasyon
    CALIBRATE = os.getenv("TRAIN_CALIBRATE", "1") not in ("0", "false", "False")
    CAL_METHOD = os.getenv("CAL_METHOD", "sigmoid")
    CAL_CV = int(os.getenv("CAL_CV", "5"))
    if CALIBRATE:
        try:
            try:
                cal = CalibratedClassifierCV(estimator=clf, method=CAL_METHOD, cv=CAL_CV)
            except TypeError:
                cal = CalibratedClassifierCV(base_estimator=clf, method=CAL_METHOD, cv=CAL_CV)
            cal.fit(X_tr, y_tr, sample_weight=sw_tr)
            model_to_eval = cal
        except Exception:
            model_to_eval = clf
    else:
        model_to_eval = clf

    # Değerlendirme
    P = model_to_eval.predict_proba(X_te)
    y_hat = np.argmax(P, axis=1)
    acc = float(np.mean(y_hat == y_te))
    ll = float(log_loss(y_te, P, labels=list(range(K))))
    brier = brier_multiclass(y_te, P, K)
    ece = expected_calibration_error(P.max(axis=1), (y_hat==y_te).astype(int), n_bins=15)

    # Kaydet
    import joblib
    raw_path = MODELS_DIR / "gbm_raw.joblib"
    joblib.dump(clf, raw_path)
    out = {
        "classes": classes,
        "metrics": {
            "accuracy": acc, "log_loss": ll,
            "brier_multiclass": brier, "ece_15bins": ece
        },
        "train": {"calibrated": bool(CALIBRATE), "method": CAL_METHOD, "cv": CAL_CV}
    }
    if CALIBRATE and "cal" in locals():
        cal_path = MODELS_DIR / "gbm_calibrated.joblib"
        joblib.dump(cal, cal_path)
        out["model_files"] = {"raw": str(raw_path), "calibrated": str(cal_path)}
    else:
        out["model_files"] = {"raw": str(raw_path)}

    (REPORTS_DIR / "metrics.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("=== SUMMARY ===")
    print("Classes:", classes)
    print("Metrics:", json.dumps(out["metrics"], indent=2))
    print("Models :", json.dumps(out["model_files"], indent=2))

if __name__ == "__main__":
    os.environ.setdefault("PYTHONUTF8", "1")
    main()
