import os
import pandas as pd
from datetime import datetime

# 🔹 Yollar
BASE_DIR = r"C:\Users\melik\AQRE"
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
REPORT_PATH = os.path.join(BASE_DIR, "reports", "weekly", "sprint1_week1_report.md")
LOG_PATH = os.path.join(BASE_DIR, "logs", "verify_raw_data.log")

# 🔹 Log yazma fonksiyonu
def log(message: str):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(f"[{timestamp}] {message}\n")
    print(message)

# 🔹 Eksik veri oranını hesaplama
def check_missing_ratio(df: pd.DataFrame):
    return (df.isnull().sum() / len(df) * 100).round(2).to_dict()

# 🔹 Rapor dosyasına yazma
def append_to_report(section_title: str, content: str):
    with open(REPORT_PATH, "a", encoding="utf-8") as f:
        f.write(f"\n\n## 🔍 {section_title}\n{content}\n")

# 🔹 Ana kontrol fonksiyonu
def main():
    log("=== AQRE: Veri Doğrulama Pipeline Başladı ===")

    report_summary = []
    for file in os.listdir(RAW_DIR):
        if file.endswith(".csv"):
            path = os.path.join(RAW_DIR, file)
            try:
                df = pd.read_csv(path)
                missing_ratios = check_missing_ratio(df)
                avg_missing = sum(missing_ratios.values()) / len(missing_ratios)
                report_summary.append((file, len(df), round(avg_missing, 2)))

                status = "✅ OK" if avg_missing < 10 else "⚠️ WARNING"
                log(f"{file}: {len(df)} satır | Ortalama eksik oranı: %{avg_missing} -> {status}")

            except Exception as e:
                log(f"❌ Hata ({file}): {str(e)}")
                report_summary.append((file, 0, "HATA"))

    # 🔹 Rapor çıktısını oluştur
    summary_md = "| Dosya | Satır Sayısı | Ortalama Eksik (%) |\n|:------|:--------------:|:------------------:|\n"
    for name, rows, miss in report_summary:
        summary_md += f"| {name} | {rows} | {miss} |\n"

    append_to_report("Veri Doğrulama Özeti", summary_md)
    log("=== AQRE: Veri Doğrulama Tamamlandı ===")

if __name__ == "__main__":
    main()
