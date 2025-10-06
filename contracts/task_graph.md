Amaç: Haftalık EPL “Top-3 Anomali” pipeline’ı.
Bağlam: Sezon 2025/26, pencere son 2–3 hafta.

Alt Görevler:
- TG-001 (GEMINI): Fikstür/xG/kadro/odds/hakem/hava veri raporu (tazelik≤6h, missing<2%).
- TG-002 (CHATGPT): Feature planı (şema + drift kontrolü).
- TG-003 (CHATGPT): Model planı (Poisson+GBM ensemble, kalibrasyon).
- TG-004 (AQRE): Kalibrasyon kabul kapısı (ECE/Brier), degraded kontrol.
- TG-005 (AQRE): Anomali çıkarımı (Gap, Confidence, EV).
- TG-006 (CHATGPT): Dashboard/rapor planı (kart başlıkları, pdf/export).

Metrikler: ECE≤0.03, Brier≤0.19, top-k anomaly precision hedef 0.6.
