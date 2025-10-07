# C:\Users\melik\AQRE\src\config.py
from pathlib import Path

# Proje kökü: bu dosyanın (src/config.py) iki üstü
ROOT_DIR = Path(__file__).resolve().parents[1]

RAW_DATA_DIR       = ROOT_DIR / "data" / "raw"
PROCESSED_DATA_DIR = ROOT_DIR / "data" / "processed"
MODELS_DIR         = ROOT_DIR / "models"
REPORTS_DIR        = ROOT_DIR / "reports"
TOOLS_DIR          = ROOT_DIR / "tools"
