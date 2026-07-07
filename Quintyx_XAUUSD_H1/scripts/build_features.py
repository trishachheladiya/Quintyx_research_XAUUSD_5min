from pathlib import Path

import numpy as np
import pandas as pd
from ta.trend import EMAIndicator, ADXIndicator, MACD
from ta.momentum import RSIIndicator, StochasticOscillator, ROCIndicator, WilliamsRIndicator
from ta.volatility import AverageTrueRange, BollingerBands
from ta.trend import CCIIndicator

ROOT = Path(__file__).resolve().parent.parent

INPUT = ROOT / "data" / "raw" / "xauusd_h1_raw.parquet"
OUTPUT = ROOT / "data" / "features" / "xauusd_h1_features.parquet"

print("=" * 60)
print("QUINTYX H1 FEATURE ENGINEERING")
print("=" * 60)

df = pd.read_parquet(INPUT)

# -----------------------------
# EMA
# -----------------------------

df["ema20"] = EMAIndicator(df["close"], 20).ema_indicator()
df["ema50"] = EMAIndicator(df["close"], 50).ema_indicator()
df["ema100"] = EMAIndicator(df["close"], 100).ema_indicator()
df["ema200"] = EMAIndicator(df["close"], 200).ema_indicator()

# -----------------------------
# RSI
# -----------------------------

df["rsi14"] = RSIIndicator(df["close"]).rsi()

# -----------------------------
# ATR
# -----------------------------

atr = AverageTrueRange(
    df["high"],
    df["low"],
    df["close"]
)

df["atr14"] = atr.average_true_range()

# -----------------------------
# ADX
# -----------------------------

adx = ADXIndicator(
    df["high"],
    df["low"],
    df["close"]
)

df["adx14"] = adx.adx()

# -----------------------------
# MACD
# -----------------------------

macd = MACD(df["close"])

df["macd"] = macd.macd()
df["macd_signal"] = macd.macd_signal()
df["macd_hist"] = macd.macd_diff()

# -----------------------------
# Bollinger
# -----------------------------

bb = BollingerBands(df["close"])

df["bb_upper"] = bb.bollinger_hband()
df["bb_lower"] = bb.bollinger_lband()
df["bb_width"] = bb.bollinger_wband()

# -----------------------------
# Stochastic
# -----------------------------

stoch = StochasticOscillator(
    df["high"],
    df["low"],
    df["close"]
)

df["stoch"] = stoch.stoch()

# -----------------------------
# CCI
# -----------------------------

cci = CCIIndicator(
    df["high"],
    df["low"],
    df["close"]
)

df["cci"] = cci.cci()

# -----------------------------
# ROC
# -----------------------------

df["roc"] = ROCIndicator(df["close"]).roc()

# -----------------------------
# Williams %R
# -----------------------------

df["williams"] = WilliamsRIndicator(
    df["high"],
    df["low"],
    df["close"]
).williams_r()

# -----------------------------
# EMA Distance
# -----------------------------

df["dist_ema20"] = df["close"] - df["ema20"]
df["dist_ema50"] = df["close"] - df["ema50"]
df["dist_ema100"] = df["close"] - df["ema100"]
df["dist_ema200"] = df["close"] - df["ema200"]

# -----------------------------
# EMA Slopes
# -----------------------------

df["ema20_slope"] = df["ema20"].diff()
df["ema50_slope"] = df["ema50"].diff()
df["ema100_slope"] = df["ema100"].diff()
df["ema200_slope"] = df["ema200"].diff()

# -----------------------------
# Returns
# -----------------------------

for n in [1,3,5,10,20]:
    df[f"return_{n}"] = df["close"].pct_change(n)

# -----------------------------
# Rolling Volatility
# -----------------------------

df["volatility20"] = df["close"].pct_change().rolling(20).std()

# -----------------------------
# Rolling High/Low
# -----------------------------

df["high20"] = df["high"].rolling(20).max()
df["low20"] = df["low"].rolling(20).min()

df["high50"] = df["high"].rolling(50).max()
df["low50"] = df["low"].rolling(50).min()

# -----------------------------
# Candle Structure
# -----------------------------

df["body"] = abs(df["close"] - df["open"])
df["upper_wick"] = df["high"] - df[["open","close"]].max(axis=1)
df["lower_wick"] = df[["open","close"]].min(axis=1) - df["low"]

# -----------------------------
# Time
# -----------------------------

df["hour"] = df["datetime"].dt.hour
df["day_of_week"] = df["datetime"].dt.dayofweek
df["month"] = df["datetime"].dt.month
df["quarter"] = df["datetime"].dt.quarter

df.dropna(inplace=True)

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
df.to_parquet(OUTPUT,index=False)

print(f"Rows : {len(df):,}")
print(f"Columns : {len(df.columns)}")
print()
print("Saved to")
print(OUTPUT)