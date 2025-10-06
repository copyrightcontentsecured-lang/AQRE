Amaç: EPL için “Top-3 Anomali” + Haftalık Kısa Tez (son 2–3 hafta)
Sahip: AQRE (plan), Gemini Pro (veri), ChatGPT (iskeler), Melik (orkestrasyon)
Run Window: Günlük; haftalık sentez Pazar 22:00 (Europe/Istanbul)

Durum Makinesi: draft → in_progress → needs_revision → complete
Kalite Kapıları: DATA_QC → FEATURE_QC → MODEL_QC → CALIB_QC → ANOMALY_QC → REPORT_QC

TG-001  [GEMINI]  Collect Fixtures/Stats/Odds/Squads/Ref/Weather
  Girdi: configs/data_sources.yaml
  Çıktı: data/raw/*.csv + contracts/dataset_manifest.md (güncellenmiş içerik)
  Kapı (DATA_QC): tazelik ≤ 6h, missing% < 2%, schema alanları eksiksiz
  Hata → needs_revision (eksik kaynak adı ve notu)

TG-002  [CHATGPT] Feature Plan (No-code)
  Girdi: dataset_manifest.md (alanlar, tazelik)
  Çıktı: docs/FEATURE_PLAN.md (özellik listesi, şema, drift kontrolleri)
  Kapı (FEATURE_QC): her feature için kaynak/alan/ölçek/birim tanımlı

TG-003  [CHATGPT] Model Plan (No-code)
  Girdi: FEATURE_PLAN.md
  Çıktı: docs/MODEL_PLAN.md (Poisson + GBM + Ensemble; split/kalibrasyon stratejisi)
  Kapı (MODEL_QC): train/valid/test ayrımı ve metrikler net (Brier, ECE)

TG-004  [AQRE] Calibration Gate + Degraded Policy
  Girdi: MODEL_PLAN.md
  Çıktı: docs/CALIBRATION_GATE.md (eşikler: ECE≤0.03, Brier≤0.19; kontrol yöntemi)
  Kapı (CALIB_QC): yöntem ve kabul eşikleri yazılı; ihlal→degraded

TG-005  [AQRE] Anomaly Definition & Scoring
  Girdi: FEATURE_PLAN + MODEL_PLAN + CALIBRATION_GATE
  Çıktı: docs/ANOMALY_RULES.md (Gap, EV, Confidence; Top-3 seçim)
  Kapı (ANOMALY_QC): seçim kriteri: Gap≥0.06 AND Confidence≥75; bağlaçlar açık

TG-006  [CHATGPT] Dashboard/Report Outline (No-code)
  Girdi: ANOMALY_RULES.md
  Çıktı: docs/REPORT_SPEC.md (kart başlıkları, tablo kolonları, haftalık PDF yapısı)
  Kapı (REPORT_QC): rapor bölümleri ve alan isimleri net

TG-007  [AQRE] Weekly Thesis (Template Fill)
  Girdi: Top-3 anomaly meta + drift/narrative notları (metinsel)
  Çıktı: reports/weekly/THESIS_WEEK_<ISO_WEEK>.md
  Kapı: tez 3 parçalı (Tez, Kanıtlar, Risk/Koşul)
