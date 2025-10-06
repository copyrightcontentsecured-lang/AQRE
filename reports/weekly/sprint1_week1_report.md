---
report_id: S1W1
model_version: v1.1.0
data_cutoff: 2025-10-06
generated_by: AQRE Orchestrator
---

# 🧭 Sprint 1 — Hafta 1 Raporu (AQRE)

## 1️⃣ Veri Durumu
**Kaynaklar:** `fixtures.csv`, `xg_stats.csv`, `squads.csv`, `odds.csv`, `referees.csv`, `weather.csv`

| Dosya | Satır | Ortalama Eksik % | Durum | Not |
|:--|:--:|:--:|:--|:--|
| fixtures.csv | 5 | 0 |✅ OK| Tutarlı, ID yapısı sağlam |
| xg_stats.csv | 5 | 34 |⚠️ Eksik| Oynanmamış maçlarda boş |
| squads.csv | 6 | 13 |⚠️ Eksik| Bazı oyuncularda `reason` boş |
| odds.csv | 5 | 0 |✅ OK| Açılış-kapanış değerleri tutarlı |
| referees.csv | 5 | 0 |✅ OK| Statik veri |
| weather.csv | 5 | 32 |⚠️ Eksik| Gelecek maçlarda tahmin eksik |

> **Genel:** Veri yapısı sağlam, fakat `xg_stats` ve `weather` setlerinde gelecek maçlardan kaynaklı doğal eksiklik var.

---

## 2️⃣ Özellik (Feature) Durumu
Toplam 📊 30 özellik üretildi.

| Kategori | Örnek Özellikler | Not |
|:--|:--|:--|
| xG & Şut | `home_xg`, `away_xg`, `xg_diff`, `home_shots_on_target` | Eksik veriler 0 ile dolduruldu |
| Takım Formu | `team_form_last5`, `recent_points` | Küçük örnekleme nedeniyle temsili |
| Hakem & Hava | `ref_card_rate`, `temperature_celsius`, `humidity_percent` | Tutarlı fakat incomplete |
| Oran Türevleri | `odds_home_win`, `odds_draw`, `odds_away_win` | Normalize edildi |
| Kombinasyon | `xg_diff * odds_home_win_inv`, `temp_scaled` | Feature etkileşimi eklendi |

> Eksik değerlerin yüksekliği nedeniyle feature engineering çıktısı uyarı sınırındaydı (%40 civarı null).

---

## 3️⃣ Model Planlama ve Sonuçlar
**Model:** Gradient Boosting Classifier (deneme modu, pseudo-label ile)  
**Veri bölünmesi:** 70 % train / 30 % test  
