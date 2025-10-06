import pandas as pd
from pathlib import Path

p = Path(r"C:\Users\melik\AQRE\data\processed\features.parquet")
df = pd.read_parquet(p)

print("Satır sayısı:", len(df))
print("Sütun sayısı:", df.shape[1])
print("Kolonlar:", list(df.columns))

if "match_outcome" in df.columns:
    print("\nmatch_outcome null sayısı:", df["match_outcome"].isna().sum())
    print("Değer dağılımı:", df["match_outcome"].value_counts(dropna=True).to_dict())
    print("\nNaN olan satırlar (ilk 5):")
    print(df[df["match_outcome"].isna()].head())
else:
    print("\n⚠️ match_outcome kolonu yok!")
