# -*- coding: utf-8 -*-
# C:\Users\melik\AQRE\tools\generate_mock_data.py
import numpy as np
import pandas as pd
from pathlib import Path

N_MATCHES_TO_ADD = 200
ROOT_DIR = Path(r"C:\Users\melik\AQRE")
RAW_DIR = ROOT_DIR / "data" / "raw"
FIXTURES_PATH = RAW_DIR / "fixtures.csv"
ODDS_PATH     = RAW_DIR / "odds.csv"

def append_aligned(df_new: pd.DataFrame, path: Path):
    """Var olan başlıklarla hizalayarak güvenli append."""
    if path.exists():
        try:
            df_old = pd.read_csv(path, nrows=1)
            cols = list(df_old.columns)
            for c in cols:
                if c not in df_new.columns:
                    df_new[c] = pd.NA
            df_new = df_new[cols]
        except Exception:
            pass
    df_new.to_csv(path, mode="a", header=not path.exists(), index=False)

def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)

    # Son match_id
    try:
        existing = pd.read_csv(FIXTURES_PATH)
        last_id = int(existing["match_id"].max()) if not existing.empty else 0
    except Exception:
        last_id = 0

    print(f"[INFO] Found last match_id: {last_id}. Adding {N_MATCHES_TO_ADD} new matches.")

    match_ids = np.arange(last_id + 1, last_id + 1 + N_MATCHES_TO_ADD)

    # H/D/A için ham skorlar (H biraz avantajlı)
    raw_H = rng.normal(loc=0.60, scale=0.25, size=N_MATCHES_TO_ADD)
    raw_D = rng.normal(loc=0.28, scale=0.12, size=N_MATCHES_TO_ADD)
    raw_A = rng.normal(loc=0.42, scale=0.25, size=N_MATCHES_TO_ADD)

    raw = np.clip(np.c_[raw_H, raw_D, raw_A], 1e-3, None)
    prob = raw / raw.sum(axis=1, keepdims=True)

    labels   = np.array(["H","D","A"])
    outcomes = np.array([rng.choice(labels, p=p) for p in prob])  # OLASILIKSAL ✅

    margin = 1.06
    odds   = margin / prob

    odds_df = pd.DataFrame({
        "match_id"  : match_ids,
        "odds_1_last": odds[:,0],
        "odds_x_last": odds[:,1],
        "odds_2_last": odds[:,2],
    })

    fix_df = pd.DataFrame({
        "match_id"       : match_ids,
        "home_team_id"   : rng.integers(100, 200, size=N_MATCHES_TO_ADD),
        "away_team_id"   : rng.integers(200, 300, size=N_MATCHES_TO_ADD),
        "fixture_date_utc": (
            pd.Timestamp("2024-09-01") +
            pd.to_timedelta(rng.integers(0, 365, size=N_MATCHES_TO_ADD), unit="D")
        ).strftime("%Y-%m-%d %H:%M:%S"),
        "match_outcome"  : outcomes,
    })

    append_aligned(fix_df,  FIXTURES_PATH)
    append_aligned(odds_df, ODDS_PATH)

    print(f"[OK] Appended: {N_MATCHES_TO_ADD} matches to fixtures and odds files.")
    print("[INFO] New match_outcome counts (batch):", pd.Series(outcomes).value_counts().to_dict())

if __name__ == "__main__":
    main()
