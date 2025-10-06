Tanımlar:
- Market Implied Probability (MIP): oranlardan vig düzeltilmiş olasılık.
- Model Probability (MP): kalibrasyon sonrası tahmin.
- Gap = MP - MIP
- EV (Expected Value) = MP*odds - 1
- Confidence: model sağlık skoru (0–100), veri kalitesiyle ağırlıklı.

Seçim Kriteri:
- Asgari filtre: Gap ≥ 0.06 VE Confidence ≥ 75
- İkincil: EV > 0; CLV proxy (kapanışa yakın hareketler) gözlemi uygunsa +puan
- Çakışma kuralı: Aynı hikâyeyi temsil eden yüksek korelasyonlu seçimlerden yalnızca biri

Top-3 Seçim:
- Yukarıdaki filtreden geçenler EV sıralamasına göre derecelenir, ilk 3 seçilir.
- Bağlayıcı alanlar:
  - match_id, kickoff_utc, selection, odds, MP, MIP, Gap, EV, Confidence
  - supporting_notes (kısa 2–3 madde)
