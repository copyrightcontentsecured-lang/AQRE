# tools/make_picks.py
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score

from src.config import REPORTS_DIR, MIN_CONF, DRAW_MARGIN


CLASSES_DEFAULT: List[str] = ["A", "D", "H"]  # yedek (loo_summary yoksa)


def _load_summary(report_dir: Path) -> dict:
    summ_path = report_dir / "loo_summary.json"
    if summ_path.exists():
        with open(summ_path, "r", encoding="utf-8") as f:
            return json.load(f)
    # yedek
    return {
        "n_samples": None,
        "n_features": None,
        "classes": CLASSES_DEFAULT,
        "metrics": {},
        "calibration": {},
    }


def _ensure_labels(df: pd.DataFrame, classes: List[str]) -> pd.DataFrame:
    """CSV’de y_true_label / y_pred_label eksikse y_true / y_pred üzerinden türetir."""
    out = df.copy()
    if "y_true_label" not in out.columns:
        if "y_true" in out.columns:
            out["y_true_label"] = out["y_true"].astype(int).map(lambda i: classes[i])
        else:
            raise KeyError("CSV lacks both 'y_true_label' and 'y_true'.")
    if "y_pred_label" not in out.columns:
        if "y_pred" in out.columns:
            out["y_pred_label"] = out["y_pred"].astype(int).map(lambda i: classes[i])
        else:
            # y_pred yoksa, proba argmax'tan üret
            proba_cols = [c for c in out.columns if c.startswith("proba_")]
            if len(proba_cols) != 3:
                raise KeyError("CSV lacks 'y_pred_label' and cannot infer from proba_*.")
            order = [f"proba_{c}" for c in classes]  # "proba_A","proba_D","proba_H"
            p = out[order].to_numpy()
            out["y_pred_label"] = p.argmax(axis=1)
            out["y_pred_label"] = out["y_pred_label"].map(lambda i: classes[i])
    return out


def _apply_draw_rule_row(row, classes: List[str], min_conf: float, draw_margin: float) -> Tuple[str, float]:
    """
    Basit kural:
      - Varsayılan: argmax
      - Eğer A ve H olasılıkları birbirine yakınsa (|pH - pA| <= draw_margin),
        ve beraberlik (D) olasılığı da ‘çok zayıf’ değilse => D seç.
      - Ek olarak: argmax olasılığı çok düşükse (< min_conf) ve top-2 farkı da küçükse,
        temkinli davranıp D seç.
    """
    pA = float(row["proba_A"])
    pD = float(row["proba_D"])
    pH = float(row["proba_H"])

    probs = np.array([pA, pD, pH], dtype=float)
    argmax_idx = int(probs.argmax())
    argmax_lbl = classes[argmax_idx]
    argmax_p = float(probs[argmax_idx])

    # "A vs H yakın" ölçütü
    close_AH = abs(pH - pA) <= draw_margin

    # "D çok zayıf değil" ölçütü: D, A/H'nin çok gerisinde olmasın
    not_weak_D = (pD >= max(pA, pH) - draw_margin) or (pD >= 0.33 - draw_margin / 2)

    # "top-2 farkı küçükse ve güven düşükse"
    top2_sorted = np.sort(probs)[::-1][:2]
    top_gap_small = (top2_sorted[0] - top2_sorted[1]) <= (draw_margin * 0.75)
    low_conf = argmax_p < min_conf

    choose_draw = (close_AH and not_weak_D) or (low_conf and top_gap_small)

    if choose_draw:
        return "D", pD
    return argmax_lbl, argmax_p


def _make_picks(df: pd.DataFrame, classes: List[str], min_conf: float, draw_margin: float) -> pd.DataFrame:
    """Eşiklere göre 'pick_label' üretir, ayrıca 'pick_conf' ekler."""
    fn = lambda r: _apply_draw_rule_row(r, classes, min_conf, draw_margin)
    picks = df.apply(fn, axis=1, result_type="expand")
    picks.columns = ["pick_label", "pick_conf"]
    return pd.concat([df.reset_index(drop=True), picks], axis=1)


def main() -> None:
    # ---- İstenen LOG satırı (kalıcı) ----
    print(f"[picks] thresholds -> MIN_CONF={MIN_CONF}  DRAW_MARGIN={DRAW_MARGIN}")

    reports = Path(REPORTS_DIR)
    csv_path = reports / "loo_predictions.csv"

    if not csv_path.exists():
        raise FileNotFoundError(f"Predictions not found: {csv_path}")

    # Özet ve sınıflar
    summary = _load_summary(reports)
    classes = summary.get("classes") or CLASSES_DEFAULT
    classes = list(classes)  # ['A','D','H'] bekleriz

    # CSV'yi yükle ve label kolonlarını güvenceye al
    pred = pd.read_csv(csv_path)
    pred = _ensure_labels(pred, classes)

    # picks_rule üret
    pred_rule = _make_picks(pred, classes, MIN_CONF, DRAW_MARGIN)

    # Confusion ve F1 (picks'e göre)
    y_true = pred_rule["y_true_label"].astype(str).to_numpy()
    y_pick = pred_rule["pick_label"].astype(str).to_numpy()

    cm = confusion_matrix(y_true, y_pick, labels=classes)
    f1 = f1_score(y_true, y_pick, labels=classes, average="macro")

    # Yazdır
    print("Confusion (rows=true, cols=pred A/D/H):")
    print(cm)
    print("\nmacro F1:", f1, "\n")

    # Base summary’yi da göster (ham model metrikleri)
    base_summary = {
        "n_samples": summary.get("n_samples"),
        "n_features": summary.get("n_features"),
        "classes": classes,
        "metrics": summary.get("metrics", {}),
        "calibration": summary.get("calibration", {}),
    }
    print("[base summary]", json.dumps(base_summary, indent=2, ensure_ascii=False))

    # picks çıktısını kaydet
    out_path = reports / "picks_rule.csv"
    pred_rule.to_csv(out_path, index=False)
    # sessizce de bilgi verelim
    # (Grafik üretimi ayrı modülde)
    # print(f"[picks] written -> {out_path}")

if __name__ == "__main__":
    main()
