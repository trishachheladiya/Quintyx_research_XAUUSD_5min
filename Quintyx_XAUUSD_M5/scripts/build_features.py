from pathlib import Path

import pandas as pd
import ta

ROOT = Path(__file__).resolve().parent.parent

INPUT = ROOT / "data" / "raw" / "xauusd_m5_raw.parquet"
OUTPUT_DIR = ROOT / "data" / "features"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT = OUTPUT_DIR / "xauusd_m5_features.parquet"

df = pd.read_parquet(INPUT)

print("=" * 60)
print("QUINTYX FEATURE ENGINEERING")
print("=" * 60)

# -----------------------------
# EMA
# -----------------------------

df["ema20"] = ta.trend.EMAIndicator(df["close"], window=20).ema_indicator()
df["ema50"] = ta.trend.EMAIndicator(df["close"], window=50).ema_indicator()
df["ema200"] = ta.trend.EMAIndicator(df["close"], window=200).ema_indicator()

# -----------------------------
# RSI
# -----------------------------

df["rsi14"] = ta.momentum.RSIIndicator(df["close"], window=14).rsi()

# -----------------------------
# ATR
# -----------------------------

df["atr14"] = ta.volatility.AverageTrueRange(
    df["high"],
    df["low"],
    df["close"],
    window=14,
).average_true_range()

# -----------------------------
# Distance from EMA
# -----------------------------

df["dist_ema20"] = df["close"] - df["ema20"]
df["dist_ema50"] = df["close"] - df["ema50"]
df["dist_ema200"] = df["close"] - df["ema200"]

# -----------------------------
# EMA Slopes
# -----------------------------

df["ema20_slope"] = df["ema20"].diff()
df["ema50_slope"] = df["ema50"].diff()
df["ema200_slope"] = df["ema200"].diff()

# -----------------------------
# Returns
# -----------------------------

df["ret1"] = df["close"].pct_change(1)
df["ret3"] = df["close"].pct_change(3)
df["ret5"] = df["close"].pct_change(5)
df["ret10"] = df["close"].pct_change(10)

# -----------------------------
# Volatility
# -----------------------------

df["volatility20"] = df["close"].rolling(20).std()

# -----------------------------
# Time Features
# -----------------------------

df["hour"] = df["datetime"].dt.hour
df["day_of_week"] = df["datetime"].dt.dayofweek

# -----------------------------
# NEW FEATURES (Upgrade)
# -----------------------------

df["body"] = abs(df["close"] - df["open"])
df["upper_wick"] = df["high"] - df[["open", "close"]].max(axis=1)
df["lower_wick"] = df[["open", "close"]].min(axis=1) - df["low"]

# Session
df["session"] = 0
df.loc[(df["hour"] >= 0) & (df["hour"] < 8), "session"] = 1
df.loc[(df["hour"] >= 8) & (df["hour"] < 16), "session"] = 2
df.loc[(df["hour"] >= 16), "session"] = 3

# -----------------------------
# Remove NaNs
# -----------------------------

df.dropna(inplace=True)
df.reset_index(drop=True, inplace=True)

df.to_parquet(OUTPUT, index=False)

print(f"Final Rows : {len(df):,}")
print(f"Columns    : {len(df.columns)}")
print(f"Saved To   : {OUTPUT}")

print("Done.")