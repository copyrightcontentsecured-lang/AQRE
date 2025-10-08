# C:\Users\melik\AQRE\src\models\loo_eval.py
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, accuracy_score, confusion_matrix
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import check_random_state

# ---- Proje config ----
from src.config import (
    REPORTS_DIR,
    PROCESSED_DATA_DIR,
    LOO_CALIBRATE,
    CAL_METHOD,
    CAL_CV,
    ECE_BINS,
    SEED,
)

# =========================================================
# Yardımcılar
# =========================================================
def _ece_maxproba(y_true: np.ndarray, proba: np.ndarray, n_bins: int = 15) -> float:
    """
    Multiclass ECE (Expected Calibration Error).
    'max-proba' tanımıyla: her örnek için tahmin edilen sınıfın olasılığı kullanılır.
    """
    y_true = np.asarray(y_true)
    proba = np.asarray(proba, dtype=float)

    pred = proba.argmax(axis=1)
    conf = proba.max(axis=1)

    # Bin aralıkları [0,1]
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true)

    for b0, b1 in zip(bins[:-1], bins[1:]):
        mask = (conf >= b0) & (conf < b1) if b1 < 1.0 else (conf >= b0) & (conf <= b1)
        if not np.any(mask):
            continue

        acc_bin = (pred[mask] == y_true[mask]).mean()
        conf_bin = conf[mask].mean()
        ece += (mask.sum() / n) * abs(acc_bin - conf_bin)

    return float(ece)


def _brier_multiclass(y_true: np.ndarray, proba: np.ndarray, classes_: list[str]) -> float:
    """
    Çok sınıflı Brier skorunun (mean squared error) genişletilmiş hali.
    """
    y_true = np.asarray(y_true)
    proba = np.asarray(proba, dtype=float)
    K = len(classes_)
    y_onehot = np.eye(K, dtype=float)[y_true]
    return float(np.mean(np.sum((proba - y_onehot) ** 2, axis=1)))


def _ensure_dirs():
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (PROCESSED_DATA_DIR).mkdir(parents=True, exist_ok=True)


def _load_dataset() -> tuple[pd.DataFrame, np.ndarray, list[str]]:
    """
    Beklenen dosya: PROCESSED_DATA_DIR / 'features.parquet'
    Beklenen hedef kolon: 'match_outcome' (değerleri 'A','D','H')
    """
    p = PROCESSED_DATA_DIR / "features.parquet"
    if not p.exists():
        raise FileNotFoundError(
            f"Beklenen veri yok: {p}\nÖnce features üret: python -m src.data.build_features"
        )

    df = pd.read_parquet(p)
    if "match_outcome" not in df.columns:
        raise ValueError("features.parquet içinde 'match_outcome' kolonu yok!")

    y_raw = df["match_outcome"].astype(str).values
    X = df.drop(columns=["match_outcome"])

    # Sadece sayısal kolonlar
    X = X.select_dtypes(include=["number"]).copy()
    print(f"[RUN] Kalan sayısal özellik: {X.shape[1]}", flush=True)

    # Etiketleri 0..K-1'e encode et (sıra: alfabetik—A,D,H beklenir)
    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    classes_ = list(le.classes_)  # örn. ['A','D','H']

    return X, y, classes_


def _build_model(random_state: int = 42):
    """
    Basit ama sağlam bir temel model: multinomial LogisticRegression.
    (Projenin kendi modelini kullanmak istersen burayı değiştirebilirsin.)
    """
    return LogisticRegression(
        penalty="l2",
        solver="lbfgs",
        multi_class="multinomial",
        max_iter=200,
        random_state=random_state,
        n_jobs=None,
    )


def _maybe_calibrated(clf, method: str, cv: int, enable: bool):
    if not enable:
        return clf
    # sklearn>=1.1: base_estimator -> estimator
    return CalibratedClassifierCV(estimator=clf, method=method, cv=cv)


# =========================================================
# Ana akış
# =========================================================
def main() -> int:
    _ensure_dirs()
    rng = check_random_state(SEED if isinstance(SEED, int) else 42)

    X, y, classes_ = _load_dataset()
    n, k = X.shape[0], X.shape[1]

    # LOO
    print(f"[RUN] LOO (calibrate={bool(LOO_CALIBRATE)}:{CAL_METHOD}, cv={int(CAL_CV)})", flush=True)
    loo = LeaveOneOut()

    # Çıktılar
    proba_out = np.zeros((n, len(classes_)), dtype=float)
    pred_out = np.zeros(n, dtype=int)

    # LOO döngüsü
    for i, (train_idx, test_idx) in enumerate(loo.split(X), start=1):
        if i % 25 == 0 or i == n:
            print(f"  - progress: {i}/{n}", flush=True)

        X_tr, y_tr = X.iloc[train_idx].values, y[train_idx]
        X_te = X.iloc[test_idx].values

        base = _build_model(random_state=rng.randint(0, 10_000_000))
        clf = _maybe_calibrated(base, method=CAL_METHOD, cv=int(CAL_CV), enable=bool(LOO_CALIBRATE))

        clf.fit(X_tr, y_tr)
        p = clf.predict_proba(X_te)  # (1, K)

        # güvenli aralık + satır-normalizasyon
        p = np.clip(p, 1e-6, 1.0 - 1e-6)
        p = p / p.sum(axis=1, keepdims=True)

        proba_out[test_idx[0], :] = p[0]
        pred_out[test_idx[0]] = int(np.argmax(p[0]))

    # Metrikler
    y_true = y
    y_pred = pred_out
    proba = proba_out

    # log_loss (eps yok!)
    ll = float(log_loss(y_true, proba, labels=np.arange(len(classes_))))
    acc = float(accuracy_score(y_true, y_pred))
    brier = _brier_multiclass(y_true, proba, classes_)
    ece = _ece_maxproba(y_true, proba, n_bins=int(ECE_BINS))

    # Raporları yaz
    reports_dir = REPORTS_DIR
    pred_path = reports_dir / "loo_predictions.csv"
    sum_path = reports_dir / "loo_summary.json"

    df_out = pd.DataFrame(
        {
            "y_true": y_true,
            "y_pred": y_pred,
        }
    )
    # Sınıf olasılık kolonlarını ekle
    for j, c in enumerate(classes_):
        df_out[f"proba_{c}"] = proba[:, j]

    df_out.to_csv(pred_path, index=False)

    summary = {
        "n_samples": int(n),
        "n_features": int(k),
        "classes": classes_,
        "metrics": {
            "log_loss": ll,
            "brier_multiclass": brier,
            "ece_15bins": ece,
            "accuracy": acc,
        },
        "calibration": {
            "enabled": bool(LOO_CALIBRATE),
            "method": str(CAL_METHOD),
            "cv": int(CAL_CV),
        },
    }
    with open(sum_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(
        f"[DONE] LOO written: {pred_path} {sum_path}\n"
        f"Classes: {classes_}\n"
        f"Metrics: {json.dumps(summary['metrics'], indent=2)}",
        flush=True,
    )

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        sys.exit(1)
