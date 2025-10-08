import json
from pathlib import Path
import pandas as pd
import numpy as np

rep = Path(r'.\reports')
csv_path = rep / 'loo_predictions.csv'
sum_path = rep / 'loo_summary.json'

if not csv_path.exists():
    raise SystemExit(f'Not found: {csv_path}')

df = pd.read_csv(csv_path)

# Sınıf sırası: summary varsa ordan, yoksa proba_ kolonlarından
classes = None
if sum_path.exists():
    try:
        with open(sum_path, 'r', encoding='utf-8') as f:
            sm = json.load(f)
            classes = sm.get('classes')
    except Exception:
        classes = None

if not classes:
    classes = [c.replace('proba_', '') for c in df.columns if c.startswith('proba_')]

# y_true_label
if 'y_true_label' not in df.columns:
    if 'y_true' in df.columns:
        inv = {i: c for i, c in enumerate(classes)}
        df['y_true_label'] = df['y_true'].map(inv)
    else:
        raise SystemExit("CSV has no 'y_true' to derive 'y_true_label'.")

# y_pred_label
if 'y_pred_label' not in df.columns:
    if 'y_pred' in df.columns:
        inv = {i: c for i, c in enumerate(classes)}
        df['y_pred_label'] = df['y_pred'].map(inv)
    else:
        proba = df[[f'proba_{c}' for c in classes]].to_numpy()
        idx = np.argmax(proba, axis=1)
        df['y_pred_label'] = [classes[i] for i in idx]

df.to_csv(csv_path, index=False)
print("OK: ensured labels in", csv_path)
