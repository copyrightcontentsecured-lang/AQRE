Proje: AQRE – Greenfield Quant Factory
Sahip: Melik
Tarih: 2025-10-06

Nihai Amaç
- Spor + finans için otonom, test edilebilir, risk-kontrollü nicel sistem.
- Üçlü zekâ (AQRE/GPT-5, Gemini Pro, ChatGPT) + Orchestrator + I/O sözleşmeleri.

MVP (Sprint 0–2)
- EPL için Top-3 Anomali kartları + Haftalık kısa tez (rapor).
- Risk Guard çekirdeği: eşikler, degraded tetikleri, kill-switch (metin).
- Orchestrator akışı (collect→feature→model→calibrate→anomaly→report) no-code provası.

Kabul Kriterleri
- OOS ECE ≤ 0.03, Brier ≤ 0.19 (kalibrasyon sonrası).
- Kartlar boş gelmez: ≥1 öneri (Gap≥0.06, Confidence≥75).
- Veri tazeliği penceresi ≤ 6 saat; ihlalde throttle/halt notu.
- Her adım run_id ile audit’e düşer (kim/ne/ne zaman/sonuç).

Risk İlkeleri
- max_single_risk_pct: 1.0
- daily_drawdown_stop_pct: 2.0
- max_concurrent_positions: 8
- correlation_cluster_cap: 0.4
- ece_max: 0.03, brier_max: 0.20
- heartbeat_timeout_s: 60

Durum Makinesi
- draft → in_progress → needs_revision → complete
- kalite kapısı ihlali → throttle/halt + needs_revision
