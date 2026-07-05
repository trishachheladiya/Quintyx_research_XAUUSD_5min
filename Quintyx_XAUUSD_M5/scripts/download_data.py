from pathlib import Path

import MetaTrader5 as mt5
import pandas as pd
import yaml

# Project root
ROOT = Path(__file__).resolve().parent.parent

# Load config
with open(ROOT / "config" / "config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Timeframe mapping
TIMEFRAME_MAP = {
    "M1": mt5.TIMEFRAME_M1,
    "M5": mt5.TIMEFRAME_M5,
    "M15": mt5.TIMEFRAME_M15,
    "M30": mt5.TIMEFRAME_M30,
    "H1": mt5.TIMEFRAME_H1,
    "H4": mt5.TIMEFRAME_H4,
    "D1": mt5.TIMEFRAME_D1,
}

symbol = config["symbol"]
timeframe = TIMEFRAME_MAP[config["timeframe"]]
bars_to_download = 90000


print("=" * 60)
print("QUINTYX XAUUSD M5 DATA DOWNLOADER")
print("=" * 60)

# Connect to MT5
if not mt5.initialize():
    raise RuntimeError(f"MT5 Initialization Failed: {mt5.last_error()}")

# Make sure symbol is enabled
if not mt5.symbol_select(symbol, True):
    mt5.shutdown()
    raise RuntimeError(f"Could not select symbol: {symbol}")

print("Connected to MT5")
print(f"Downloading {bars_to_download:,} bars of {symbol}...")

# Download latest bars
rates = mt5.copy_rates_from_pos(
    symbol,
    timeframe,
    0,
    bars_to_download
)

if rates is None:
    print("MT5 Error:", mt5.last_error())
    mt5.shutdown()
    raise RuntimeError("Failed to download data.")

# Convert to DataFrame
df = pd.DataFrame(rates)

# Convert timestamp
df["time"] = pd.to_datetime(df["time"], unit="s")
df.rename(columns={"time": "datetime"}, inplace=True)

# Sort oldest → newest
df.sort_values("datetime", inplace=True)
df.reset_index(drop=True, inplace=True)

# Save
output_dir = ROOT / "data" / "raw"
output_dir.mkdir(parents=True, exist_ok=True)

output_file = output_dir / "xauusd_m5_raw.parquet"

df.to_parquet(output_file, index=False)

print(f"Downloaded {len(df):,} candles")
print(f"Saved to: {output_file}")

mt5.shutdown()

print("Done.")