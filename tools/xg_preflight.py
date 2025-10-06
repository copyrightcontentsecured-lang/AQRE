# C:\Users\melik\AQRE\tools\xg_preflight.py
# -*- coding: utf-8 -*-
import sys
sys.path.append(r"C:\Users\melik\AQRE")
from src.config import PROCESSED_DATA_DIR
from pathlib import Path
import pandas as pd

PREFIXES = (
    "home_xg","away_xg",
    "home_shots","away_shots",
    "home_shots_on_target","away_shots_on_target",
)

def main() -> int:
    feat_path = PROCESSED_DATA_DIR / "features.parquet"
    if not feat_path.exists():
        print(f"[FAIL] features.parquet bulunamadı: {feat_path}")
        return 2

    try:
        df = pd.read_parquet(feat_path)
    except Exception as e:
        print(f"[FAIL] features.parquet okunamadı: {e}")
        return 2

    xg_cols = [c for c in df.columns if c.startswith(PREFIXES)]
    if not xg_cols:
        print("[WARN] xg-türevi kolon bulunamadı (prefix eşleşmedi). Eğitime devam edilebilir ama bu beklenmiyor.")
        return 0

    bad = [c for c in xg_cols if df[c].isna().all()]
    if bad:
        print("[FAIL] xg türevi kolonlar tamamen NaN! Bu genelde xg merge/okuma sorunudur.")
        print("       Sorunlu kolonlar:", bad)
        return 2

    rates = df[xg_cols].isna().mean().round(3).to_dict()
    max_rate = max(rates.values()) if rates else 0.0

    print("[OK]  xg kolonları:", xg_cols)
    print("[OK]  xg NaN oranları:", rates)
    if max_rate > 0.20:
        print("[WARN] xg NaN rates yüksek görünüyor (>%20). Yine de eğitim devam edebilir.")

    return 0

if __name__ == "__main__":
    raise SystemExit(main())
