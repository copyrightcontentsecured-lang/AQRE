import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import log_loss
import joblib

ROOT = Path(r"C:\Users\melik\AQRE")
PROC = ROOT / "data" / "processed"
MODELS = ROOT / "models"
MODELS.mkdir(parents=True, exist_ok=True)

def multiclass_brier(y_true, proba, classes=None):
    """
    Çok-sınıflı Brier skoru (one-vs-all MSE ortalaması).
    y_true: (n,) int sınıf etiketleri
    proba : (n, K) sınıf olasılıkları
    """
    y_true = np.asarray(y_true)
    n = len(y_true)
    if classes is None:
        classes = np.unique(y_true)
    K = proba.shape[1]
    # one-hot
    Y = np.zeros((n, K))
    for i, c in enumerate(y_true):
        Y[i, int(c)] = 1.0
    return np.mean((Y - proba) ** 2)

def main():
    print("=== AQRE: Model Eğitimi Başladı ===")
    df_path = PROC / "features.parquet"
    df = pd.read_parquet(df_path)

    # Yalnızca etiketli satırlar
    if "match_outcome" not in df.columns:
        print("⚠️ Etiket sütunu (match_outcome) yok. Deneme modu.")
        df_labeled = pd.DataFrame()
    else:
        df_labeled = df[df["match_outcome"].notna()].copy()

    n_lab = len(df_labeled)
    uniq = sorted(df_labeled["match_outcome"].dropna().unique().tolist()) if n_lab else []
    print(f"▶️ Etiketli satır sayısı: {n_lab} | Sınıflar: {uniq}")

    # Özellik matrisi: sadece sayısal sütunlar, etiketi dışla
    num_cols = df_labeled.select_dtypes(include=["number"]).columns.tolist()
    if "match_outcome" in num_cols:
        num_cols.remove("match_outcome")

    # Kullanılabilir veri yoksa çık
    if n_lab == 0 or len(num_cols) == 0:
        print("⚠️ Yeterli etiketli veri yok. Eğitim atlandı.")
        return

    X = df_labeled[num_cols].fillna(0.0)
    y = df_labeled["match_outcome"].astype(int).values

    # Sabit (variance=0) sütunları at
    var = X.var(axis=0)
    keep_cols = var[var > 0].index.tolist()
    if len(keep_cols) == 0:
        print("⚠️ Tüm sayısal sütunlar sabit görünüyor. Eğitim atlandı.")
        return
    X = X[keep_cols]

    # Küçük veri/sınıf sayısı kontrolü
    enough_classes = len(np.unique(y)) >= 2
    enough_rows = len(X) >= 6  # min güvenli eşik

    scaler = StandardScaler()
    model = GradientBoostingClassifier(random_state=42)

    if not enough_classes or not enough_rows:
        # Çok küçük veri: tümünü tek blokta eğit, metrik raporlama yok
        print("ℹ️ Küçük veri modu: tüm veriyle tek seferde eğitim, metrik yok.")
        X_scaled = scaler.fit_transform(X)
        model.fit(X_scaled, y)
        joblib.dump({"scaler": scaler, "model": model, "features": keep_cols}, MODELS / "gbm_model.pkl")
        print("✅ Model dosyası kaydedildi: gbm_model.pkl")
        print("=== AQRE: Model Eğitimi Tamamlandı ===")
        return

    # Yeterli veri: stratified split + metrikler
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=min(0.3, max(0.2, 2/len(X))), random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    model.fit(X_train_scaled, y_train)
    proba = model.predict_proba(X_test_scaled)

    # Metrikler
    try:
        ll = log_loss(y_test, proba, labels=np.unique(y))
        brier = multiclass_brier(y_test, proba)
        print(f"📉 Log Loss: {ll:.4f}")
        print(f"📊 Brier (multi-class): {brier:.4f}")
    except Exception as e:
        print(f"ℹ️ Metrikler hesaplanamadı: {e}")

    # Kaydet
    joblib.dump({"scaler": scaler, "model": model, "features": keep_cols}, MODELS / "gbm_model.pkl")
    print("✅ Model dosyası kaydedildi: gbm_model.pkl")
    print("=== AQRE: Model Eğitimi Tamamlandı ===")

if __name__ == "__main__":
    main()
