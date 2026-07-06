from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

file = ROOT / "data" / "raw" / "xauusd_m5_raw.parquet"

df = pd.read_parquet(file)

print("=" * 60)
print("QUINTYX DATA VALIDATOR")
print("=" * 60)

print(f"Rows              : {len(df):,}")
print(f"Columns           : {len(df.columns)}")
print(f"Duplicate Rows    : {df.duplicated().sum()}")
print(f"Missing Values    : {df.isnull().sum().sum()}")

print()
print("Date Range")
print(df["datetime"].min())
print(df["datetime"].max())

time_diff = df["datetime"].diff()

missing = time_diff[time_diff > pd.Timedelta(minutes=5)]

print(f"Missing Candles   : {len(missing)}")

print()
print(df.head())

print()
print(df.tail())