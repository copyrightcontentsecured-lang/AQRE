# C:\Users\melik\AQRE\src\models\loo_eval.py
# LOO evaluation with epsilon-smoothing, robust feature drop, and optional per-fold probability calibration
# Writes:
#   - reports/loo_predictions.csv
#   - reports/loo_summary.json

import json
import os
import sys
import warnings
from typing import Tuple

import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import log_loss
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier

# --- Pretty logging ---------------------------------------------------------
def log(msg, color=None, prefix="[RUN] "):
    try:
        from colorama import Fore, Style  # optional
        COLORS = {
            "cyan": Fore.CYAN,
            "green": Fore.GREEN,
            "yellow": Fore.YELLOW,
            "magenta": Fore.MAGENTA,
            "red": Fore.RED,
        }
        c = COLORS.get(color, "")
        r = Style.RESET_ALL if c else ""
        print(f"{prefix}{c}{msg}{r}")
    except Exception:
        print(f"{prefix}{msg}")

# --- Config ----------------------------------------------------------------
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_PQ = os.path.join(ROOT, "data", "processed", "features.parquet")
REPORTS = os.path.join(ROOT, "reports")
os.makedirs(REPORTS, exist_ok=True)

TARGET_COL = "match_outcome"

# Kalibrasyon bayrakları
CALIBRATE = True           # True → her fold’da train üzerinde kalibrasyon
CAL_METHOD = "sigmoid"     # "sigmoid" genelde stabil; "isotonic" da deneyebilirsin
CAL_CV = 5                 # kalibrasyon için iç-CV

# Değerler
EPS = 1e-12                # epsilon smoothing
ECE_BINS = 15              # reliability hesaplaması için bin sayısı

# --- Helper metrics ---------------------------------------------------------
def brier_multiclass(y_true_int: np.ndarray, proba: np.ndarray, n_classes: int) -> float:
    """Multiclass Brier score."""
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
        acc = float(np.mean(correct[m]))
        w = float(np.sum(m) / n)
        ece += abs(acc - conf) * w
    return float(ece)

def eps_smooth(P: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """p_i := (p_i + eps) / (sum_j p_j + K*eps)  (satır bazında)."""
    K = P.shape[1]
    P = P + eps
    row_sum = P.sum(axis=1, keepdims=True)
    P = P / np.clip(row_sum, 1e-32, None)
    return P

# --- Robust feature filtering ----------------------------------------------
def robust_feature_drop(df: pd.DataFrame, y_name: str) -> pd.DataFrame:
    """
    Aşırı eksik/sonsuz/sabit kolonları düşür.
    + her zaman çıkarılacak kolonlar (tarih/ID)
    + sadece sayısal sütunları tut
    """
    # 1) her zaman çıkarılacak kolonlar
    EXCLUDE_ALWAYS = {
        "fixture_date_utc", "fixture_datetime_utc", "kickoff_utc",
        "match_id", "fixture_id", "id"
    }
    drop_list = [c for c in EXCLUDE_ALWAYS if c in df.columns]

    # hedef dışı ham X
    X = df.drop(columns=[y_name] + drop_list, errors="ignore")

    # 2) sadece sayısal sütunları tut
    X = X.select_dtypes(include=[np.number]).copy()

    # 3) kalite filtreleri
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

    return pd.concat([X[keep], df[[y_name]]], axis=1)

# --- Model factory ----------------------------------------------------------
def build_base_model():
    """
    Güvenli varsayılan: sklearn GradientBoostingClassifier (predict_proba destekli).
    İstersen kendi GBM/LGBM/XGB parametrelerinle değiştir.
    """
    return GradientBoostingClassifier(random_state=42)

# --- Load features ----------------------------------------------------------
def load_features(path: str) -> pd.DataFrame:
    try:
        return pd.read_parquet(path)
    except Exception as e:
        raise RuntimeError(f"Parquet okunamadı: {path}\n{e}")

# --- Leave-One-Out core -----------------------------------------------------
def loo_eval(df: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
    # Hedef & özellikler
    if TARGET_COL not in df.columns:
        raise KeyError(f"TARGET_COL='{TARGET_COL}' dataframe içinde yok.")

    df = robust_feature_drop(df, TARGET_COL)
    y = df[TARGET_COL].astype(str).to_numpy()
    X = df.drop(columns=[TARGET_COL])

    # Label encoder (sınıf sırasını sabit tut)
    le = LabelEncoder()
    le.fit(y)
    classes = list(le.classes_)
    y_int = le.transform(y)
    n = len(y_int)
    K = len(classes)

    log(f"Kalan sayısal özellik sayısı: {X.shape[1]}", "yellow")

    # Çıktılar
    rows = []
    cal_fail_count = 0

    log(f"LOO eval (ε-smoothing + robust feature drop + calibration={CALIBRATE}:{CAL_METHOD}, cv={CAL_CV})", "cyan")

    X_np = X.to_numpy(dtype=float, copy=False)

    for i in range(n):
        # Train/test böl
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        X_tr, y_tr = X_np[mask], y_int[mask]
        X_te = X_np[~mask].reshape(1, -1)
        y_true_i = y_int[i]

        # Model
        clf = build_base_model()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            clf.fit(X_tr, y_tr)

        # Kalibrasyon (train üzerinde)
        if CALIBRATE:
            try:
                try:
                    cal = CalibratedClassifierCV(estimator=clf, method=CAL_METHOD, cv=CAL_CV)
                except TypeError:
                    # eski sklearn imzası
                    cal = CalibratedClassifierCV(base_estimator=clf, method=CAL_METHOD, cv=CAL_CV)

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    cal.fit(X_tr, y_tr)

                proba_i = cal.predict_proba(X_te)[0]
            except Exception:
                cal_fail_count += 1
                proba_i = clf.predict_proba(X_te)[0]
        else:
            proba_i = clf.predict_proba(X_te)[0]

        # epsilon smoothing + normalize
        proba_i = eps_smooth(proba_i.reshape(1, -1), EPS)[0]

        # satır yaz
        row = {"index": i, "y_true_label": classes[y_true_i]}
        for c_idx, c in enumerate(classes):
            row[f"proba_{c}"] = float(proba_i[c_idx])
        rows.append(row)

    # DataFrame
    pred_df = pd.DataFrame(rows)

    # Metrics
    P = pred_df[[f"proba_{c}" for c in classes]].to_numpy()
    y_true_int = le.transform(pred_df["y_true_label"].to_numpy())
    y_pred_int = np.argmax(P, axis=1)

    ll = float(log_loss(y_true_int, P, labels=range(K)))
    brier = float(brier_multiclass(y_true_int, P, K))
    ece = float(expected_calibration_error(P.max(axis=1), (y_pred_int == y_true_int).astype(int), ECE_BINS))
    acc = float(np.mean(y_pred_int == y_true_int))

    summary = {
        "n_samples": int(n),
        "classes": classes,
        "metrics": {
            "log_loss": ll,
            "brier_multiclass": brier,
            "ece_15bins": ece,
            "accuracy": acc,
        },
        "calibration": {
            "enabled": CALIBRATE,
            "method": CAL_METHOD if CALIBRATE else None,
            "cv": CAL_CV if CALIBRATE else None,
            "folds_failed": int(cal_fail_count),
        },
    }
    return pred_df, summary

# --- Main -------------------------------------------------------------------
def main():
    log(f"Using file: {__file__}", "cyan")
    if not os.path.exists(DATA_PQ):
        raise FileNotFoundError(f"Features parquet yok: {DATA_PQ}. Önce build_features.py çalıştırın.")

    df = load_features(DATA_PQ)
    pred_df, summary = loo_eval(df)

    # Yaz
    csv_path = os.path.join(REPORTS, "loo_predictions.csv")
    json_path = os.path.join(REPORTS, "loo_summary.json")

    pred_df.to_csv(csv_path, index=False)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    log(f"LOO written: {csv_path} {json_path}", "green", prefix="[DONE] ")

    # Konsola kısa özet
    print("\n[REPORT] loo_summary.json\n")
    print(f"n_samples : {summary['n_samples']}")
    print(f"classes   : {{{', '.join(summary['classes'])}}}")
    m = summary["metrics"]
    print(
        "metrics   : "
        f"log_loss={m['log_loss']:.6f}  "
        f"brier_multiclass={m['brier_multiclass']:.6f}  "
        f"ece_15bins={m['ece_15bins']:.6f}  "
        f"accuracy={m['accuracy']:.6f}"
    )
    if summary["calibration"]["enabled"]:
        c = summary["calibration"]
        print(f"calib     : method={c['method']} cv={c['cv']} folds_failed={c['folds_failed']}")

if __name__ == "__main__":
    os.environ.setdefault("PYTHONUTF8", "1")
    try:
        main()
    except Exception as e:
        log(str(e), "red", prefix="[FAIL] ")
        sys.exit(2)
