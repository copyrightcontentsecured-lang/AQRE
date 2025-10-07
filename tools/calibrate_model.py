import os, json, numpy as np, pandas as pd
from pathlib import Path
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import log_loss
import joblib

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed" / "features.parquet"
RAW  = ROOT / "models" / "gbm_raw.joblib"
OUTM = ROOT / "models" / "gbm_calibrated.joblib"
REPORT = ROOT / "reports" / "calibration_summary.json"
os.makedirs(REPORT.parent, exist_ok=True)

def ece_maxprob(proba, y_true, classes, n_bins=15):
    proba = np.asarray(proba); y_true = np.asarray(y_true)
    idx_of = {c:i for i,c in enumerate(classes)}
    y_idx = np.array([idx_of[v] for v in y_true])
    maxp = proba.max(1); y_pred = proba.argmax(1)
    correct = (y_pred == y_idx).astype(float)
    bins = np.linspace(0.0, 1.0, n_bins+1)
    which = np.digitize(maxp, bins) - 1
    ece = 0.0
    for b in range(n_bins):
        m = which == b
        if m.any():
            conf = float(maxp[m].mean())
            acc  = float(correct[m].mean())
            w = float(m.mean())
            ece += abs(acc - conf) * w
    return float(ece)

def brier_multiclass(proba, y_true, classes):
    proba = np.asarray(proba)
    idx_of = {c:i for i,c in enumerate(classes)}
    y_idx = np.array([idx_of[v] for v in y_true])
    onehot = np.zeros_like(proba)
    onehot[np.arange(len(y_idx)), y_idx] = 1.0
    return float(np.mean((proba - onehot)**2))

def extract_estimator(obj):
    # Doğrudan estimator mı?
    if hasattr(obj, "predict_proba") and hasattr(obj, "get_params"):
        return obj
    # Dict içinden bir estimator bul
    if isinstance(obj, dict):
        # En yaygın anahtarlar:
        for k in ["model","estimator","clf","classifier","gbm","pipeline"]:
            if k in obj and hasattr(obj[k], "predict_proba") and hasattr(obj[k], "get_params"):
                return obj[k]
        # Olmadıysa tüm değerleri tara
        for v in obj.values():
            if hasattr(v, "predict_proba") and hasattr(v, "get_params"):
                return v
    raise ValueError("Kaynak dosyada uygun bir sınıflandırıcı bulunamadı.")

# --- veri
df = pd.read_parquet(DATA)
y  = df["match_outcome"].astype(str).values
X  = df.drop(columns=["match_outcome"])

# --- ham modeli yükle ve estimator'ı çıkar
loaded = joblib.load(RAW)
est = extract_estimator(loaded)

# --- kalibrasyon (yeni ve eski API ile uyumlu)
try:
    cal = CalibratedClassifierCV(estimator=est, method="isotonic", cv=5)
except TypeError:
    cal = CalibratedClassifierCV(base_estimator=est, method="isotonic", cv=5)

cal.fit(X, y)

# --- önce/sonra karşılaştırma (in-sample, iç CV kalibrasyonuna göre)
proba_raw = est.predict_proba(X)
proba_cal = cal.predict_proba(X)
classes = [str(c) for c in cal.classes_]

summary = {
    "classes": classes,
    "log_loss": {
        "raw":  float(log_loss(y, proba_raw, labels=classes)),
        "cal":  float(log_loss(y, proba_cal, labels=classes)),
    },
    "brier_multiclass": {
        "raw":  brier_multiclass(proba_raw, y, classes),
        "cal":  brier_multiclass(proba_cal, y, classes),
    },
    "ece_15bins": {
        "raw":  ece_maxprob(proba_raw, y, classes, n_bins=15),
        "cal":  ece_maxprob(proba_cal, y, classes, n_bins=15),
    }
}

joblib.dump(cal, OUTM)
with open(REPORT, "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)
print("[SAVED]", OUTM)
print(json.dumps(summary, indent=2))
