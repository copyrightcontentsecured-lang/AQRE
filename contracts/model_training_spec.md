model_training_spec (ChatGPT’ye brif):
- Girdi: processed/feature taslağı (metin)
- Modeller: Poisson (skor dağılımı), GBM (olasılık)
- Ensemble: weighted
- Kalibrasyon: time-based split + isotonic/logistic
- Kabul: OOS ECE≤0.03, Brier≤0.19
- Artefakt isim planı (sadece isimler)
