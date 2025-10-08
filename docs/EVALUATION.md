

---



\# tools/make\_picks.py (tam dosya)



```python

\# -\*- coding: utf-8 -\*-

"""

tools.make\_picks

\- LOO CSV'den etiketleri okuyup karıştırmadan karışıklık matrisi ve metrikleri basar.

\- Sınıf bazlı precision/recall/F1 ve toplam 'D' (beraberlik) sayısını raporlar.

\- Eşikleri (MIN\_CONF, DRAW\_MARGIN) terminalde gösterir.



Gerekenler:

&nbsp; - reports/loo\_predictions.csv  (kolonlar: y\_true\_label, y\_pred\_label, proba\_A, proba\_D, proba\_H)

&nbsp; - reports/loo\_summary.json     (n\_samples, classes, metrics, calibration vs.)

"""



from \_\_future\_\_ import annotations

import json

from pathlib import Path



import numpy as np

import pandas as pd

from sklearn.metrics import (

&nbsp;   confusion\_matrix,

&nbsp;   classification\_report,

&nbsp;   accuracy\_score,

&nbsp;   f1\_score,

)



\# ----- yollar

ROOT = Path(\_\_file\_\_).resolve().parents\[1]

REPORTS = ROOT / "reports"

CSV = REPORTS / "loo\_predictions.csv"

SUM = REPORTS / "loo\_summary.json"



\# ----- okunabilir sınıf sırası

DEFAULT\_ORDER = \["A", "D", "H"]





def load\_summary():

&nbsp;   if SUM.exists():

&nbsp;       with open(SUM, "r", encoding="utf-8") as f:

&nbsp;           js = json.load(f)

&nbsp;       classes = js.get("classes") or DEFAULT\_ORDER

&nbsp;       # güvenlik: beklenen sırayı koru

&nbsp;       classes = \[c for c in DEFAULT\_ORDER if c in set(classes)]

&nbsp;       base = js.get("metrics", {})

&nbsp;       n\_samples = js.get("n\_samples", None)

&nbsp;       n\_features = js.get("n\_features", None)

&nbsp;       cal = js.get("calibration", {})

&nbsp;       return classes, base, n\_samples, n\_features, cal

&nbsp;   else:

&nbsp;       return DEFAULT\_ORDER, {}, None, None, {}





def main():

&nbsp;   # Eşikleri env'den oku (yoksa config içindeki default'lar terminalden hızlı bakış için)

&nbsp;   try:

&nbsp;       from src.config import MIN\_CONF, DRAW\_MARGIN

&nbsp;       print(f"\[picks] thresholds -> MIN\_CONF={MIN\_CONF}  DRAW\_MARGIN={DRAW\_MARGIN}")

&nbsp;   except Exception:

&nbsp;       print("\[picks] thresholds -> (env/config okunamadı)")



&nbsp;   if not CSV.exists():

&nbsp;       raise FileNotFoundError(f"CSV bulunamadı: {CSV}")



&nbsp;   df = pd.read\_csv(CSV)

&nbsp;   # Etiket kolon adlarını garanti et

&nbsp;   for col in \["y\_true\_label", "y\_pred\_label"]:

&nbsp;       if col not in df.columns:

&nbsp;           raise KeyError(

&nbsp;               f"Beklenen kolon yok: '{col}'. "

&nbsp;               f"Önce `python -m tools.ensure\_labels` çalıştırın."

&nbsp;           )



&nbsp;   y\_true = df\["y\_true\_label"].astype(str).to\_numpy()

&nbsp;   y\_pred = df\["y\_pred\_label"].astype(str).to\_numpy()



&nbsp;   classes, base\_metrics, n\_samples, n\_features, cal = load\_summary()

&nbsp;   labels = classes if classes else DEFAULT\_ORDER



&nbsp;   # ----- metrikler

&nbsp;   acc = accuracy\_score(y\_true, y\_pred)

&nbsp;   macro\_f1 = f1\_score(y\_true, y\_pred, average="macro")



&nbsp;   # confusion

&nbsp;   cm = confusion\_matrix(y\_true, y\_pred, labels=labels)



&nbsp;   # sınıf raporu

&nbsp;   cls\_rep = classification\_report(

&nbsp;       y\_true, y\_pred, labels=labels, target\_names=labels, digits=3

&nbsp;   )



&nbsp;   # toplam 'D' (beraberlik) tahmini

&nbsp;   n\_draw\_preds = int(np.sum(y\_pred == "D"))



&nbsp;   # ----- yazdır

&nbsp;   print("Confusion (rows=true, cols=pred A/D/H):")

&nbsp;   # ekranda sabit sırayla gösterelim

&nbsp;   order = \["A", "D", "H"]

&nbsp;   # cm çıktısı seçilen 'labels' sırasına göre; görünüm için A/D/H'yi yeniden dizelim

&nbsp;   idx\_map = \[labels.index(c) for c in order if c in labels]

&nbsp;   cm\_view = cm\[np.ix\_(idx\_map, idx\_map)]

&nbsp;   print(cm\_view.tolist())

&nbsp;   print()

&nbsp;   print(f"macro F1: {macro\_f1}")

&nbsp;   print()



&nbsp;   # base summary (loo\_summary.json) + picks özet

&nbsp;   merged\_summary = {

&nbsp;       "n\_samples": n\_samples if n\_samples is not None else int(len(df)),

&nbsp;       "n\_features": n\_features,

&nbsp;       "classes": order,

&nbsp;       "metrics": {

&nbsp;           # base\_metrics (varsa) aynen geçir

&nbsp;           \*\*base\_metrics,

&nbsp;           # picks sonrası güncel (hesaplanan) metrikler:

&nbsp;           "accuracy": acc,

&nbsp;           "macro\_f1": macro\_f1,

&nbsp;       },

&nbsp;       "picks": {

&nbsp;           "n\_draw\_preds": n\_draw\_preds,

&nbsp;       },

&nbsp;       "calibration": cal,

&nbsp;   }

&nbsp;   print("\[base summary]", json.dumps(merged\_summary, indent=2))



&nbsp;   # sınıf bazlı tablo

&nbsp;   print("\\n\[classification report]\\n")

&nbsp;   print(cls\_rep)





if \_\_name\_\_ == "\_\_main\_\_":

&nbsp;   main()



