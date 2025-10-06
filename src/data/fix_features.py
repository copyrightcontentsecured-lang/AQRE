# -*- coding: utf-8 -*-
print("[RUN] fix_features:", __file__)

from pathlib import Path
import numpy as np
import pandas as pd

IN_PATH  = Path(r"C:\Users\melik\AQRE\data\processed\features.parquet")
OUT_PATH = IN_PATH  # aynısının üstüne yazacağız (istersen değiştir)

def to_num(s):
    """Sayısal kolonlar için güvenli dönüştürme (string/obj → float)."""
    return pd.to_numeric(s, errors="coerce")

def safe_inverse(x: pd.Series):
    """x>0 ise 1/x, değilse NaN."""
    x = to_num(x)
    return x.rdiv(1.0).where(x > 0)

def pct_nan(s: pd.Series) -> float:
    return float(s.isna().mean())

# --- load ---
df = pd.read_parquet(IN_PATH)
n0 = len(df)
print(f"[INFO] Loaded features: rows={n0}, cols={df.shape[1]}")

# --- target normalize (yalnızca trim/upper; asıl mapping train tarafında) ---
tgt = "match_outcome" if "match_outcome" in df.columns else None
if tgt:
    df[tgt] = (
        df[tgt].astype(str).str.strip()
          .replace({"": np.nan, "nan": np.nan, "NaN": np.nan, "None": np.nan})
          .str.upper()
    )
    dropped = df[tgt].isna().sum()
    print(f"[INFO] target NaN count: {dropped} / {n0}")

# --- odds kolonları: inverse üret ve tüm-NaN sütunları düzelt ---
odds_bases = ["odds_1_last", "odds_x_last", "odds_2_last"]
inv_map = {
    "odds_1_last": "odds_1_last_inv",
    "odds_x_last": "odds_x_last_inv",
    "odds_2_last": "odds_2_last_inv",
}

for base in odds_bases:
    inv = inv_map[base]
    if base in df.columns:
        df[base] = to_num(df[base])
        if pct_nan(df[base]) < 1.0:
            need_make = (inv not in df.columns) or (pct_nan(df[inv]) == 1.0)
            if need_make:
                df[inv] = safe_inverse(df[base])
                print(f"[FIX] created/overwrote {inv} from {base}")
        else:
            if inv in df.columns:
                df[inv] = np.nan
            print(f"[WARN] {base} is 100% NaN; {inv} set to NaN.")
    else:
        print(f"[INFO] {base} not found (skipping inverse).")

# --- tümü NaN sütunları düşür ---
all_nan_cols = [c for c in df.columns if df[c].isna().all()]
if all_nan_cols:
    print(f"[CLEAN] dropping all-NaN columns: {all_nan_cols}")
    df.drop(columns=all_nan_cols, inplace=True)

# --- hafif doldurma (opsiyonel): yalnızca sayısal ve %95'ten az NaN olanlar ---
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
fillable = [c for c in numeric_cols if 0.0 < pct_nan(df[c]) < 0.95]
for c in fillable:
    med = df[c].median(skipna=True)
    df[c] = df[c].fillna(med)

print(f"[INFO] final shape: rows={len(df)}, cols={df.shape[1]}")
df.to_parquet(OUT_PATH, index=False)
print(f"[DONE] saved → {OUT_PATH}")
