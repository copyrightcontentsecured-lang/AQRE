<!-- DEFAULTS_BANNER_BEGIN -->
> **Calibration:** isotonic (CV=10)  
> **Picks defaults:** MIN_CONF=0.46, DRAW_MARGIN=0.09
<!-- DEFAULTS_BANNER_END -->


---



\# tools/make\_picks.py (tam dosya)



```python

\# -\*- coding: utf-8 -\*-

"""

tools.make\_picks

\- LOO CSV'den etiketleri okuyup karÄ±ÅŸtÄ±rmadan karÄ±ÅŸÄ±klÄ±k matrisi ve metrikleri basar.

\- SÄ±nÄ±f bazlÄ± precision/recall/F1 ve toplam 'D' (beraberlik) sayÄ±sÄ±nÄ± raporlar.

\- EÅŸikleri (MIN\_CONF, DRAW\_MARGIN) terminalde gÃ¶sterir.



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



\# ----- okunabilir sÄ±nÄ±f sÄ±rasÄ±

DEFAULT\_ORDER = \["A", "D", "H"]





def load\_summary():

&nbsp;   if SUM.exists():

&nbsp;       with open(SUM, "r", encoding="utf-8") as f:

&nbsp;           js = json.load(f)

&nbsp;       classes = js.get("classes") or DEFAULT\_ORDER

&nbsp;       # gÃ¼venlik: beklenen sÄ±rayÄ± koru

&nbsp;       classes = \[c for c in DEFAULT\_ORDER if c in set(classes)]

&nbsp;       base = js.get("metrics", {})

&nbsp;       n\_samples = js.get("n\_samples", None)

&nbsp;       n\_features = js.get("n\_features", None)

&nbsp;       cal = js.get("calibration", {})

&nbsp;       return classes, base, n\_samples, n\_features, cal

&nbsp;   else:

&nbsp;       return DEFAULT\_ORDER, {}, None, None, {}





def main():

&nbsp;   # EÅŸikleri env'den oku (yoksa config iÃ§indeki default'lar terminalden hÄ±zlÄ± bakÄ±ÅŸ iÃ§in)

&nbsp;   try:

&nbsp;       from src.config import MIN\_CONF, DRAW\_MARGIN

&nbsp;       print(f"\[picks] thresholds -> MIN\_CONF={MIN\_CONF}  DRAW\_MARGIN={DRAW\_MARGIN}")

&nbsp;   except Exception:

&nbsp;       print("\[picks] thresholds -> (env/config okunamadÄ±)")



&nbsp;   if not CSV.exists():

&nbsp;       raise FileNotFoundError(f"CSV bulunamadÄ±: {CSV}")



&nbsp;   df = pd.read\_csv(CSV)

&nbsp;   # Etiket kolon adlarÄ±nÄ± garanti et

&nbsp;   for col in \["y\_true\_label", "y\_pred\_label"]:

&nbsp;       if col not in df.columns:

&nbsp;           raise KeyError(

&nbsp;               f"Beklenen kolon yok: '{col}'. "

&nbsp;               f"Ã–nce `python -m tools.ensure\_labels` Ã§alÄ±ÅŸtÄ±rÄ±n."

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



&nbsp;   # sÄ±nÄ±f raporu

&nbsp;   cls\_rep = classification\_report(

&nbsp;       y\_true, y\_pred, labels=labels, target\_names=labels, digits=3

&nbsp;   )



&nbsp;   # toplam 'D' (beraberlik) tahmini

&nbsp;   n\_draw\_preds = int(np.sum(y\_pred == "D"))



&nbsp;   # ----- yazdÄ±r

&nbsp;   print("Confusion (rows=true, cols=pred A/D/H):")

&nbsp;   # ekranda sabit sÄ±rayla gÃ¶sterelim

&nbsp;   order = \["A", "D", "H"]

&nbsp;   # cm Ã§Ä±ktÄ±sÄ± seÃ§ilen 'labels' sÄ±rasÄ±na gÃ¶re; gÃ¶rÃ¼nÃ¼m iÃ§in A/D/H'yi yeniden dizelim

&nbsp;   idx\_map = \[labels.index(c) for c in order if c in labels]

&nbsp;   cm\_view = cm\[np.ix\_(idx\_map, idx\_map)]

&nbsp;   print(cm\_view.tolist())

&nbsp;   print()

&nbsp;   print(f"macro F1: {macro\_f1}")

&nbsp;   print()



&nbsp;   # base summary (loo\_summary.json) + picks Ã¶zet

&nbsp;   merged\_summary = {

&nbsp;       "n\_samples": n\_samples if n\_samples is not None else int(len(df)),

&nbsp;       "n\_features": n\_features,

&nbsp;       "classes": order,

&nbsp;       "metrics": {

&nbsp;           # base\_metrics (varsa) aynen geÃ§ir

&nbsp;           \*\*base\_metrics,

&nbsp;           # picks sonrasÄ± gÃ¼ncel (hesaplanan) metrikler:

&nbsp;           "accuracy": acc,

&nbsp;           "macro\_f1": macro\_f1,

&nbsp;       },

&nbsp;       "picks": {

&nbsp;           "n\_draw\_preds": n\_draw\_preds,

&nbsp;       },

&nbsp;       "calibration": cal,

&nbsp;   }

&nbsp;   print("\[base summary]", json.dumps(merged\_summary, indent=2))



&nbsp;   # sÄ±nÄ±f bazlÄ± tablo

&nbsp;   print("\\n\[classification report]\\n")

&nbsp;   print(cls\_rep)





if \_\_name\_\_ == "\_\_main\_\_":

&nbsp;   main()





<!-- ONE_LINERS_BEGIN -->
### Tek Satırlık Komut Örnekleri

**Tüm akış (tek satır):**
\\\powershell
python -m src.models.loo_eval; python -m tools.ensure_labels; python -m tools.make_picks; python -m tools.plot_eval; Start-Process .\reports\confusion_matrix.png; Start-Process .\reports\reliability_maxproba.png; Start-Process .\reports\loo_summary.json
\\\

**Sadece değerlendirme (LOO + metrikler):**
\\\powershell
python -m src.models.loo_eval; Start-Process .\reports\loo_summary.json
\\\

**Etiket sütunlarını garanti et:**
\\\powershell
python -m tools.ensure_labels
\\\

**Picks üret (default eşikler):**
\\\powershell
python -m tools.make_picks
\\\

**Picks üret (seanslık agresif eşikler):**
\\\powershell
="0.46"; ="0.09"; python -m tools.make_picks; Remove-Item Env:MIN_CONF,Env:DRAW_MARGIN -ErrorAction SilentlyContinue
\\\

**Grafikleri çiz:**
\\\powershell
python -m tools.plot_eval; Start-Process .\reports\confusion_matrix.png; Start-Process .\reports\reliability_maxproba.png
\\\
<!-- ONE_LINERS_END -->
