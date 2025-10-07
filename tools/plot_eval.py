# C:\Users\melik\AQRE\tools\plot_eval.py
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report

from src.config import REPORTS_DIR

def reliability_from_csv(csv, bins=10, out="reliability_maxproba.png"):
    df = pd.read_csv(csv)
    classes = ["A","D","H"]
    P = df[[f"proba_{c}" for c in classes]].to_numpy()
    y = df["y_true_label"].astype(str).to_numpy()
    y_int = np.array([classes.index(v) for v in y])
    y_hat = P.argmax(axis=1)
    correct = (y_hat==y_int).astype(int)
    maxp = P.max(axis=1)

    edges = np.linspace(0,1,bins+1)
    mids, accs = [], []
    for b in range(bins):
        m = (maxp>=edges[b]) & (maxp<edges[b+1] if b<bins-1 else maxp<=edges[b+1])
        if m.sum()==0: 
            continue
        mids.append(maxp[m].mean())
        accs.append(correct[m].mean())
    fig, ax = plt.subplots(figsize=(8,6))
    ax.plot([0,1],[0,1],'--',color='tab:orange')
    ax.plot(mids, accs, marker='o')
    ax.set_title("Reliability")
    ax.set_xlabel("Confidence"); ax.set_ylabel("Accuracy")
    plt.tight_layout(); plt.savefig(REPORTS_DIR/out, dpi=150)

def confusion_from_csv(csv, out="confusion_matrix.png"):
    df = pd.read_csv(csv)
    classes = ["A","D","H"]
    y = df["y_true_label"].astype(str).to_numpy()
    y_hat = df[[f"proba_{c}" for c in classes]].to_numpy().argmax(axis=1)
    y_hat = np.array([classes[i] for i in y_hat])
    cm = confusion_matrix(y, y_hat, labels=classes)

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8,6))
    im = ax.imshow(cm, cmap="viridis")
    ax.set_xticks(range(3)); ax.set_yticks(range(3))
    ax.set_xticklabels(classes); ax.set_yticklabels(classes)
    ax.set_title("Confusion matrix (rows=true, cols=pred)")
    for i in range(3):
        for j in range(3):
            ax.text(j, i, str(cm[i,j]), ha="center", va="center", color="w", fontsize=12)
    plt.colorbar(im, ax=ax, fraction=0.046)
    plt.tight_layout(); plt.savefig(REPORTS_DIR/out, dpi=150)

def main():
    csv = REPORTS_DIR / "loo_predictions.csv"
    confusion_from_csv(csv)
    reliability_from_csv(csv)

if __name__ == "__main__":
    main()
