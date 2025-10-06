Amaç: Olasılıkların güvenilirliğini garanti altına almak ve sapma varsa otomatik kısma/durdurma.

Metrikler:
- ECE (Expected Calibration Error) – hedef ≤ 0.03 (OOS)
- Brier Score – hedef ≤ 0.19 (OOS)
- Drift göstergeleri: son 200 olaylık kayan pencerede ECE↑ veya Brier↑ trendi

Kontrol Prosedürü:
1) Zaman bazlı validasyon: rolling window split (haftalık).
2) Kalibrasyon: isotonic veya logistic; seçimi A/B karşılaştır.
3) Kabul:
   - Eğer ECE≤0.03 VE Brier≤0.19 → OK
   - Aksi → DEGRADED: new_entries = throttle (0.25x stake) veya halt

Degraded Politika:
- İlk ihlalde: throttle (stake çarpanı 0.25, Top-3 yerine Top-1)
- 2 ardışık ihlalde: halt (lig için yeni giriş yok)
- İyileşme kriteri: 2 ardışık “OK” döneminde otomatik normale dönüş

Audit:
- Her karar (OK/THROTTLE/HALT) zaman damgalı olarak audit/ altına eklenir.
