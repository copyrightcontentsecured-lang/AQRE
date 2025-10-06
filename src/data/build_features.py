# C:\Users\melik\AQRE\src\data\build_features.py
# -*- coding: utf-8 -*-
"""
Ham csv'leri (fixtures/odds/weather/xg/referees/squads) birleştirip
features.parquet üretir. Yollar config.py'den gelir.
"""
import sys; sys.path.append(r"C:\Users\melik\AQRE")
from src.config import RAW_DATA_DIR, PROCESSED_DATA_DIR
import pandas as pd
from pathlib import Path

def run():
    print(f"[RUN] build_features (auto from fixtures/odds/etc): {__file__}")
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    paths = {
        "fixtures":  RAW_DATA_DIR / "fixtures.csv",
        "odds":      RAW_DATA_DIR / "odds.csv",
        "weather":   RAW_DATA_DIR / "weather.csv",
        "xg":        RAW_DATA_DIR / "xg.csv",
        "referees":  RAW_DATA_DIR / "referees.csv",
        "squads":    RAW_DATA_DIR / "squads.csv",
    }

    dfs, shapes = {}, {}
    for name, path in paths.items():
        try:
            dfs[name] = pd.read_csv(path)
            shapes[name] = dfs[name].shape
        except FileNotFoundError:
            print(f"[WARN] missing: {path}")
            dfs[name] = pd.DataFrame()
            shapes[name] = (0,0)

    print("[INFO] shapes:", " ".join([f"{k} {v}" for k, v in shapes.items()]))

    # ---- CRITICAL: xg tam mı? (truncation guard + görünür yol) ----
    if not dfs["xg"].empty:
        print("[DEBUG] xg path:", paths["xg"])
        print("[DEBUG] xg shape (pre):", dfs["xg"].shape)
        if len(dfs["xg"]) < 200:
            # herhangi bir truncation ihtimaline karşı yeniden oku (ABSOLUTE path ile)
            dfs["xg"] = pd.read_csv(paths["xg"])
            print("[FIX] Re-read xg full:", dfs["xg"].shape)
        print("[DEBUG] xg columns:", list(dfs["xg"].columns))

    # ---- Ana tablo ve merge'ler ----
    if dfs["fixtures"].empty:
        raise ValueError("fixtures.csv boş veya yok.")
    out = dfs["fixtures"].copy()

    # tip hizalama (match_id Int64)
    for nm in ["fixtures","xg","odds","weather","referees","squads"]:
        if not dfs[nm].empty and "match_id" in dfs[nm].columns:
            dfs[nm]["match_id"] = pd.to_numeric(dfs[nm]["match_id"], errors="coerce").astype("Int64")
    out["match_id"] = pd.to_numeric(out["match_id"], errors="coerce").astype("Int64")

    for name in ["odds","weather","xg","referees","squads"]:
        if not dfs[name].empty:
            if "match_id" not in dfs[name].columns:
                print(f"[WARN] {name} has no match_id; skipped")
                continue
            out = out.merge(dfs[name], on="match_id", how="left")
            if name == "xg":
                xg_cols=[c for c in dfs["xg"].columns if c.startswith(("home_xg","away_xg","home_shots","away_shots","home_shots_on_target","away_shots_on_target"))]
                if xg_cols:
                    print("[DEBUG] xg NaN rates:", out[xg_cols].isna().mean().round(3).to_dict())

    # odds türevleri (varsa)
    odds_cols = ["odds_1_last","odds_x_last","odds_2_last"]
    if all(c in out.columns for c in odds_cols):
        for c in odds_cols: out[f"{c}_inv"] = 1.0 / out[c]
        print("[INFO] odds columns present:", odds_cols + [f"{c}_inv" for c in odds_cols])
        print("[INFO] odds NaN rates:", out[[f"{c}_inv" for c in odds_cols]].isna().mean().to_dict())

    if "match_outcome" in out.columns:
        print("[INFO] match_outcome counts:", out["match_outcome"].value_counts().to_dict())

    out_path = PROCESSED_DATA_DIR / "features.parquet"
    out.to_parquet(out_path, index=False)
    print(f"[DONE] saved -> {out_path} | shape: {out.shape}")

if __name__ == "__main__":
    run()
