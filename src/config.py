# C:\Users\melik\AQRE\src\config.py
from __future__ import annotations
from pathlib import Path
import os, json
from typing import Optional, Dict, Any

# ---------- Kök & dizinler ----------
ROOT = Path(__file__).resolve().parents[1]

DATA_DIR    = ROOT / "data"
REPORTS_DIR = ROOT / "reports"
MODELS_DIR  = ROOT / "models"
for d in (DATA_DIR, REPORTS_DIR, MODELS_DIR):
    d.mkdir(parents=True, exist_ok=True)

# Alt veri dizinleri
RAW_DATA_DIR       = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
for d in (RAW_DATA_DIR, PROCESSED_DATA_DIR):
    d.mkdir(parents=True, exist_ok=True)

# ---------- Güvenli env okuyucular ----------
def _get_bool(name: str, default: bool = True) -> bool:
    s = os.getenv(name)
    if s is None:
        return default
    return s.strip().lower() not in ("0", "false", "no", "off", "n", "")

def _get_int(name: str, default: int, lo: Optional[int] = None, hi: Optional[int] = None) -> int:
    try:
        v = int(os.getenv(name, str(default)))
    except Exception:
        v = default
    if lo is not None: v = max(lo, v)
    if hi is not None: v = min(hi, v)
    return v

def _get_float(name: str, default: float, lo: Optional[float] = None, hi: Optional[float] = None) -> float:
    try:
        v = float(os.getenv(name, str(default)))
    except Exception:
        v = default
    if lo is not None: v = max(lo, v)
    if hi is not None: v = min(hi, v)
    return v

# ---------- Genel sabitler ----------
EPS       = _get_float("EPS", 1e-12, 0.0, 1.0)
ECE_BINS  = _get_int("ECE_BINS", 15, 2, 100)
SEED      = _get_int("SEED", 42, 0, 10_000_000)

# ---------- LOO kalibrasyon ----------
LOO_CALIBRATE = _get_bool("LOO_CALIBRATE", True)
CAL_METHOD    = os.getenv("CAL_METHOD", "sigmoid")  # "sigmoid" | "isotonic"
CAL_CV        = _get_int("CAL_CV", 5, 2, 20)

# ---------- Picks stratejisi ----------
MIN_CONF    = _get_float("MIN_CONF", 0.47, 0.0, 1.0)
DRAW_MARGIN = _get_float("DRAW_MARGIN", 0.10, 0.0, 0.5)

# ---------- Görüntüleme ----------
def to_dict() -> Dict[str, Any]:
    return {
        "paths": {
            "ROOT": str(ROOT),
            "DATA_DIR": str(DATA_DIR),
            "RAW_DATA_DIR": str(RAW_DATA_DIR),
            "PROCESSED_DATA_DIR": str(PROCESSED_DATA_DIR),
            "REPORTS_DIR": str(REPORTS_DIR),
            "MODELS_DIR": str(MODELS_DIR),
        },
        "constants": {
            "EPS": EPS,
            "ECE_BINS": ECE_BINS,
            "SEED": SEED,
        },
        "loo_calibration": {
            "LOO_CALIBRATE": LOO_CALIBRATE,
            "CAL_METHOD": CAL_METHOD,
            "CAL_CV": CAL_CV,
        },
        "picks": {
            "MIN_CONF": MIN_CONF,
            "DRAW_MARGIN": DRAW_MARGIN,
        },
    }

def dump_config(pretty: bool = True) -> str:
    return json.dumps(to_dict(), indent=2 if pretty else None, ensure_ascii=False)
