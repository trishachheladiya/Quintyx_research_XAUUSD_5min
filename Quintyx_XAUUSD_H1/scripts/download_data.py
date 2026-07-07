from pathlib import Path
from datetime import datetime

import MetaTrader5 as mt5
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

OUTPUT = ROOT / "data" / "raw" / "xauusd_h1_raw.parquet"
OUTPUT.parent.mkdir(parents=True, exist_ok=True)

SYMBOL = "XAUUSD"
TIMEFRAME = mt5.TIMEFRAME_H1
BARS = 90000

print("=" * 60)
print("QUINTYX XAUUSD H1 DOWNLOADER")
print("=" * 60)

if not mt5.initialize():
    raise RuntimeError(mt5.last_error())

print("Connected to MT5")

rates = mt5.copy_rates_from_pos(
    SYMBOL,
    TIMEFRAME,
    0,
    BARS
)

if rates is None:
    print("MT5 Error:", mt5.last_error())
    raise RuntimeError("No data returned.")

df = pd.DataFrame(rates)

df["datetime"] = pd.to_datetime(df["time"], unit="s")

df = df[
    [
        "datetime",
        "open",
        "high",
        "low",
        "close",
        "tick_volume",
        "spread",
        "real_volume",
    ]
]

df.to_parquet(OUTPUT, index=False)

mt5.shutdown()

print(f"Downloaded {len(df):,} candles")
print("Saved to")
print(OUTPUT)