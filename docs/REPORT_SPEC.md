\# REPORT\_SPEC.md  

\*\*Version:\*\* v1.1 (Sprint 1 — October 2025)  

\*\*Author:\*\* AQRE Core System  

\*\*Purpose:\*\* Standart raporlama çerçevesi (daily / weekly) ve metrik tanımları.



---



\## 🧭 1. Raporlama Çerçevesi



| Rapor Tipi | Frekans | Kaynak Veriler | Özetlenmiş Dosya |

|-------------|----------|----------------|------------------|

| \*\*Günlük\*\* (`reports/daily/\*.md`) | Her çalışma günü (UTC 00:00) | `logs/run\_\*`, `data/processed/` | `daily\_report\_<date>.md` |

| \*\*Haftalık\*\* (`reports/weekly/\*.md`) | Her Pazar (UTC 23:00) | `data/processed/`, `models/`, `logs/validation/` | `sprint1\_weekX\_report.md` |



---



\## 📊 2. Rapor Bölümleri



Her rapor, aşağıdaki bölümlerden oluşmak zorundadır:



| Bölüm | İçerik | Kaynak | Zorunluluk |

|--------|--------|---------|------------|

| \*\*1. Veri Durumu\*\* | Kaynak listesi, missing %, tazelik | `contracts/dataset\_manifest.md` | ✅ |

| \*\*2. Özellik (Feature) Durumu\*\* | Feature sayısı, korelasyon, leakage | `docs/FEATURE\_PLAN.md` | ✅ |

| \*\*3. Model Performansı\*\* | Metrik sonuçları (Brier, ECE, LogLoss) | `models/\*` | ✅ |

| \*\*4. Kalibrasyon Analizi\*\* | Platt vs Isotonic kıyaslaması | `CALIBRATION\_GATE.md` | ✅ |

| \*\*5. Anomali Gözlemleri\*\* | Outlier, drift, yanlış skorlar | `ANOMALY\_RULES.md` | ✅ |

| \*\*6. Risk Durumu\*\* | Günlük max DD, Kill-switch logları | `risk\_guard/telemetry.log` | ⚙️ |

| \*\*7. Öneriler \& Tez Özeti\*\* | Haftalık analiz + öneriler | `reports/weekly/THESIS\_TEMPLATE.md` | ⚙️ |



---



\## 🧮 3. Performans Metrikleri



| Metrik | Tanım | Hedef | Formül |

|---------|--------|--------|---------|

| \*\*Brier Score\*\* | Tahmin olasılığı ile gerçekleşen sonuç farkı | ≤ \*\*0.19\*\* |  !\[Brier Formula](https://wikimedia.org/api/rest\_v1/media/math/render/svg/fbb5cf26de09c2e5b38c) |

| \*\*Expected Calibration Error (ECE)\*\* | Model tahminlerinin güvenilirliği | ≤ \*\*0.03\*\* |  !\[ECE Formula](https://wikimedia.org/api/rest\_v1/media/math/render/svg/fab5d452a81a6a) |

| \*\*Log Loss\*\* | Aşırı güvenli veya düşük güvenli tahminlerin cezalandırılması | Bilgi amaçlı |  |

| \*\*PSI (Population Stability Index)\*\* | Veri drift ölçümü | ≤ \*\*0.20\*\* |  |

| \*\*CLV (Closing Line Value)\*\* | Piyasa kapanış oranına göre performans | ≥ \*\*1.00\*\* |  |



> 💡 \*\*Not:\*\* CLV yalnızca canlı oran verisi varsa hesaplanır. Diğer metrikler offline veriyle yapılabilir.



---



\## ⚙️ 4. Doğrulama \& Kalibrasyon Akışı



\### Veri → Özellik → Model → Kalibrasyon zinciri



```mermaid

graph TD

A\[data/raw/\*.csv] --> B\[data/processed/feature\_matrix.csv]

B --> C\[model\_training\_spec.md]

C --> D\[calibration\_notebook.ipynb]

D --> E\[reports/weekly/sprint1\_weekX\_report.md]



🧾 6. Rapor Format Kuralları

Dosya adı: sprint<no>_week<no>_report.md

Her raporun başında YAML meta etiket bulunur:

report_id: S1W1
model_version: v1.1.0
data_cutoff: 2025-10-06
generated_by: AQRE Orchestrator
