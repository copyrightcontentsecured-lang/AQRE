# C:\Users\melik\AQRE\tools\make_picks.py
import os, json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

from src.config import REPORTS_DIR

DRAW_MARGIN = float(os.getenv("DRAW_MARGIN", "0.06"))   # top1-top2 < margin => D
MIN_CONF    = float(os.getenv("MIN_CONF", "0.45"))      # max_proba < MIN_CONF => D

def decide_row(row, cls=("A","D","H")):
    pA, pD, pH = row[f"proba_{cls[0]}"], row[f"proba_{cls[1]}"], row[f"proba_{cls[2]}"]
    probs = np.array([pA,pD,pH])
    order = probs.argsort()[::-1]
    top, second = order[0], order[1]
    maxp, secondp = probs[top], probs[second]
    # zaten D ise bırak
    if top == 1:
        return "D"
    # düşük güven → D
    if maxp < MIN_CONF:
        return "D"
    # A ve H yakınsa → D
    if {top, second} == {0,2} and (maxp - secondp) < DRAW_MARGIN:
        return "D"
    # aksi halde klasik tahmin
    return ("A","D","H")[top]

def main():
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    csv = REPORTS_DIR / "loo_predictions.csv"
    summ_path = REPORTS_DIR / "loo_summary.json"
    pred = pd.read_csv(csv)
    classes = ["A","D","H"]
    pred["pick_rule"] = pred.apply(decide_row, axis=1, cls=classes)
    # rapor
    y_true = pred["y_true_label"].values
    y_pred = pred["pick_rule"].values
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    rep = classification_report(y_true, y_pred, labels=classes, output_dict=True)
    out = {
        "rule": {"DRAW_MARGIN": DRAW_MARGIN, "MIN_CONF": MIN_CONF},
        "confusion": cm.tolist(),
        "report": rep
    }
    (REPORTS_DIR / "picks_rule.csv").write_text(pred.to_csv(index=False), encoding="utf-8")
    (REPORTS_DIR / "picks_rule_summary.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("Confusion (rows=true, cols=pred A/D/H):")
    print(cm)
    print("\nmacro F1:", rep["macro avg"]["f1-score"])
    if summ_path.exists():
        print("\n[base summary]", summ_path.read_text())

if __name__ == "__main__":
    main()
