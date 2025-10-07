import os, json, numpy as np, pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report

os.makedirs("reports/figs", exist_ok=True)
df = pd.read_csv(r"reports/loo_predictions.csv")

labels = ["A","D","H"]
y_true = df["y_true_label"].values
P = df[[f"proba_{l}" for l in labels]].values
y_pred = np.array([labels[i] for i in P.argmax(1)])

# === 1) Confusion matrix + classification report ===
cm = confusion_matrix(y_true, y_pred, labels=labels)
print("Confusion matrix (rows=true, cols=pred):")
print(pd.DataFrame(cm, index=labels, columns=labels), "\n")
print(classification_report(y_true, y_pred, labels=labels, digits=3))

# Görsel: confusion matrix
plt.figure()
plt.imshow(cm, interpolation='nearest')
plt.xticks(range(len(labels)), labels)
plt.yticks(range(len(labels)), labels)
plt.title("Confusion matrix")
for i in range(len(labels)):
    for j in range(len(labels)):
        plt.text(j, i, cm[i, j], ha='center', va='center')
plt.tight_layout()
plt.savefig("reports/figs/confusion_matrix.png", dpi=160)

# === 2) Reliability (max-proba) + ECE(15) ===
maxp = P.max(1)
correct = (y_true == y_pred).astype(float)

edges = np.linspace(0.0, 1.0, 16)
idx = np.digitize(maxp, edges) - 1

rows = []
ece = 0.0
n = len(y_true)
for b in range(15):
    m = (idx == b)
    if m.any():
        conf = float(maxp[m].mean())
        acc  = float(correct[m].mean())
        w    = m.sum() / n
        ece += abs(acc - conf) * w
        rows.append((f"{edges[b]:.2f}-{edges[b+1]:.2f}", int(m.sum()), conf, acc))

rep = pd.DataFrame(rows, columns=["bin","count","confidence","accuracy"])
print("\nReliability by bins:")
print(rep.to_string(index=False))
print(f"\nECE (15 bins): {ece:.6f}")

# Görsel: reliability
plt.figure()
plt.plot(rep["confidence"], rep["accuracy"], marker='o')
plt.plot([0,1],[0,1],'--')
plt.xlabel("Confidence")
plt.ylabel("Accuracy")
plt.title(f"Reliability (ECE≈{ece:.3f})")
plt.tight_layout()
plt.savefig("reports/figs/reliability_maxproba.png", dpi=160)

print("\n[OK] Saved figures to reports/figs/")
