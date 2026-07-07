from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

DATA = ROOT / "data" / "raw" / "xauusd_h1_raw.parquet"

print("=" * 60)
print("QUINTYX H1 DATA VALIDATOR")
print("=" * 60)

df = pd.read_parquet(DATA)

print(f"Rows              : {len(df):,}")
print(f"Columns           : {len(df.columns)}")
print(f"Duplicate Rows    : {df.duplicated().sum()}")
print(f"Missing Values    : {df.isna().sum().sum()}")

print("\nDate Range")
print(df["datetime"].min())
print(df["datetime"].max())

df["datetime"] = pd.to_datetime(df["datetime"])

expected = pd.date_range(
    start=df["datetime"].min(),
    end=df["datetime"].max(),
    freq="1h"
)

missing = expected.difference(df["datetime"])

print(f"Missing Candles   : {len(missing)}")

print("\nFirst 5 Rows")
print(df.head())

print("\nLast 5 Rows")
print(df.tail())