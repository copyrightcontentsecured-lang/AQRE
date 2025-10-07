# C:\Users\melik\AQRE\src\models\loo_eval.py
# LOO evaluation with epsilon-smoothing, robust feature drop, and optional per-fold calibration
# Writes:
#   - reports/loo_predictions.csv
#   - reports/loo_summary.json

import json, os, sys, warnings
from typing import Tuple, List
import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import log_loss
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier

# ---- logging ---------------------------------------------------------------
def log(msg, color=None, prefix="[RUN] "):
    try:
        from colorama import Fore, Style
        COLORS = {"cyan": Fore.CYAN, "green": Fore.GREEN, "yellow": Fore.YELLOW,
                  "magenta": Fore.MAGENTA, "red": Fore.RED}
        c = COLORS.get(color, "")
        r = Style.RESET_ALL if c else ""
        print(f"{prefix}{c}{msg}{r}")
    except Exception:
        print(f"{prefix}{msg}")

# ---- config (portable) -----------------------------------------------------
from src.config import PROCESSED_DATA_DIR, REPORTS_DIR
DATA_PQ = (PROCESSED_DATA_DIR / "features.parquet").as_posix()
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

TARGET_COL = "match_outcome"
EPS = 1e-12
ECE_BINS = 15

# CI’da kalibrasyonu kapat (lokalde açık kalsın). İstersen override et:
#   - AQRE_LOO_CALIBRATE=1  -> aç
#   - AQRE_LOO_CALIBRATE=0  -> kapat
def _env_bool(name: str, default: bool) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return v.strip() not in {"0", "false", "False", ""}

RUNNING_CI = bool(os.getenv("GITHUB_ACTIONS") or os.getenv("CI"))
_cal_default = False if RUNNING_CI else True
CALIBRATE = _env_bool("AQRE_LOO_CALIBRATE", _cal_default)
CAL_METHOD = os.getenv("AQRE_CAL_METHOD", "sigmoid")
try:
    CAL_CV = int(os.getenv("AQRE_CAL_CV", "5"))
except ValueError:
    CAL_CV = 5

# ---- utils -----------------------------------------------------------------
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

def eps_smooth(P: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    K = P.shape[1]
    P = P + eps
    row_sum = P.sum(axis=1, keepdims=True)
    return P / np.clip(row_sum, 1e-32, None)

def robust_feature_drop(df: pd.DataFrame, y_name: str) -> pd.DataFrame:
    """Aşırı eksik/sonsuz/sabit + kimlik/tarih kolonlarını at."""
    ALWAYS_DROP = {"fixture_date_utc", "match_id", "fixture_id", "id"}
    X = df.drop(columns=[y_name])
    keep: List[str] = []
    for c in X.columns:
        if c in ALWAYS_DROP or c.lower() in ALWAYS_DROP:
            continue
        s = X[c]
        # sayısal dönüştür; olmazsa at
        try:
            arr = s.to_numpy(dtype=float, copy=False)
        except Exception:
            continue
        if np.mean(pd.isna(arr)) > 0.20:
            continue
        if not np.isfinite(arr).all():
            continue
        if pd.Series(arr).nunique(dropna=True) <= 1:
            continue
        keep.append(c)
    return df[keep + [y_name]]

def build_base_model():
    return GradientBoostingClassifier(random_state=42)

def load_features(path: str) -> pd.DataFrame:
    try:
        return pd.read_parquet(path)
    except Exception as e:
        raise RuntimeError(f"Parquet okunamadı: {path}\n{e}")

# ---- LOO core --------------------------------------------------------------
def loo_eval(df: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
    df = robust_feature_drop(df, TARGET_COL)
    y = df[TARGET_COL].astype(str).to_numpy()
    X = df.drop(columns=[TARGET_COL])

    le = LabelEncoder().fit(y)
    classes = list(le.classes_)
    y_int = le.transform(y)
    n = len(y_int)
    K = len(classes)

    log(f"Kalan sayısal özellik sayısı: {X.shape[1]}", "cyan")
    log(f"LOO eval (ε-smoothing + robust feature drop + calibration={CALIBRATE}:{CAL_METHOD}, cv={CAL_CV})", "cyan")

    X_np = X.to_numpy(dtype=float, copy=False)
    rows = []
    cal_fail = 0

    for i in range(n):
        mask = np.ones(n, dtype=bool); mask[i] = False
        X_tr, y_tr = X_np[mask], y_int[mask]
        X_te = X_np[~mask].reshape(1, -1)
        y_true_i = y_int[i]

        clf = build_base_model()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            clf.fit(X_tr, y_tr)

        if CALIBRATE:
            try:
                try:
                    cal = CalibratedClassifierCV(estimator=clf, method=CAL_METHOD, cv=CAL_CV)
                except TypeError:
                    cal = CalibratedClassifierCV(base_estimator=clf, method=CAL_METHOD, cv=CAL_CV)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    cal.fit(X_tr, y_tr)
                proba_i = cal.predict_proba(X_te)[0]
            except Exception:
                cal_fail += 1
                proba_i = clf.predict_proba(X_te)[0]
        else:
            proba_i = clf.predict_proba(X_te)[0]

        proba_i = eps_smooth(proba_i.reshape(1, -1), EPS)[0]
        row = {"index": i, "y_true_label": classes[y_true_i]}
        for c_idx, c in enumerate(classes):
            row[f"proba_{c}"] = float(proba_i[c_idx])
        rows.append(row)

    pred_df = pd.DataFrame(rows)
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
        "metrics": {"log_loss": ll, "brier_multiclass": brier, "ece_15bins": ece, "accuracy": acc},
        "calibration": {"enabled": bool(CALIBRATE), "method": CAL_METHOD if CALIBRATE else None,
                        "cv": CAL_CV if CALIBRATE else None, "folds_failed": int(cal_fail)},
    }
    return pred_df, summary

# ---- main ------------------------------------------------------------------
def main():
    log(f"Using file: {__file__}", "cyan")
    if not os.path.exists(DATA_PQ):
        raise FileNotFoundError(f"Features parquet yok: {DATA_PQ}. Önce build_features.py çalıştırın.")

    df = load_features(DATA_PQ)
    pred_df, summary = loo_eval(df)

    csv_path = (REPORTS_DIR / "loo_predictions.csv").as_posix()
    json_path = (REPORTS_DIR / "loo_summary.json").as_posix()
    pred_df.to_csv(csv_path, index=False)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    log(f"LOO written: {csv_path} {json_path}", "green", prefix="[DONE] ")
    print("\n[REPORT] loo_summary.json\n")
    print(f"n_samples : {summary['n_samples']}")
    print(f"classes   : {{{', '.join(summary['classes'])}}}")
    m = summary["metrics"]
    print(f"metrics   : log_loss={m['log_loss']:.6f}  brier_multiclass={m['brier_multiclass']:.6f}  "
          f"ece_15bins={m['ece_15bins']:.6f}  accuracy={m['accuracy']:.6f}")
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
