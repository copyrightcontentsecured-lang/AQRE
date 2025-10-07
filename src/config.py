from pathlib import Path
import os

# Proje kökü (.../AQRE)
ROOT = Path(__file__).resolve().parents[1]

# Dizinler
DATA_DIR    = ROOT / "data"
REPORTS_DIR = ROOT / "reports"
MODELS_DIR  = ROOT / "models"
for d in (DATA_DIR, REPORTS_DIR, MODELS_DIR):
    d.mkdir(parents=True, exist_ok=True)


RAW_DATA_DIR       = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

for d in (RAW_DATA_DIR, PROCESSED_DATA_DIR):
    d.mkdir(parents=True, exist_ok=True)
# Sayısal sabitler / env
EPS       = float(os.getenv("EPS", "1e-12"))
ECE_BINS  = int(os.getenv("ECE_BINS", "15"))

# LOO kalibrasyon (env ile override edilebilir)
LOO_CALIBRATE = os.getenv("LOO_CALIBRATE", "1").lower() not in ("0", "false")
CAL_METHOD    = os.getenv("CAL_METHOD", "sigmoid")
CAL_CV        = int(os.getenv("CAL_CV", "5"))

