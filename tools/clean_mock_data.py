# C:\Users\melik\AQRE\tools\clean_mock_data.py
# -*- coding: utf-8 -*-
import pandas as pd
from pathlib import Path

CUTOFF_ID = 1055
ROOT_DIR = Path(r"C:\Users\melik\AQRE")
RAW_DIR = ROOT_DIR / "data" / "raw"
FILES = [RAW_DIR/"fixtures.csv", RAW_DIR/"odds.csv"]

def clean_one(p: Path):
    if not p.exists():
        print(f"[WARN] Missing:", p.name); return
    try:
        df = pd.read_csv(p)
        if "match_id" not in df.columns:
            print(f"[WARN] No 'match_id' in", p.name); return
        before = len(df)
        df = df[df["match_id"] <= CUTOFF_ID].copy()
        df.to_csv(p, index=False)
        print(f"[OK] {p.name}: {before} -> {len(df)}")
    except Exception as e:
        print(f"[ERROR] {p.name}: {e}")

def main():
    print(f"--- Clean start (keep match_id <= {CUTOFF_ID}) ---")
    for p in FILES:
        clean_one(p)
    print("--- Clean done ---")

if __name__ == "__main__":
    main()
