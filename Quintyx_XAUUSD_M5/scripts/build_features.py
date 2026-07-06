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

# =============================
# Trend Strength
# =============================

adx = ta.trend.ADXIndicator(
    high=df["high"],
    low=df["low"],
    close=df["close"],
    window=14,
)

df["adx14"] = adx.adx()
df["plus_di"] = adx.adx_pos()
df["minus_di"] = adx.adx_neg()

# =============================
# MACD
# =============================

macd = ta.trend.MACD(df["close"])

df["macd"] = macd.macd()
df["macd_signal"] = macd.macd_signal()
df["macd_hist"] = macd.macd_diff()

# =============================
# Bollinger Bands
# =============================

bb = ta.volatility.BollingerBands(df["close"], window=20)

df["bb_upper"] = bb.bollinger_hband()
df["bb_lower"] = bb.bollinger_lband()
df["bb_width"] = bb.bollinger_wband()

# =============================
# Stochastic
# =============================

stoch = ta.momentum.StochasticOscillator(
    high=df["high"],
    low=df["low"],
    close=df["close"]
)

df["stoch_k"] = stoch.stoch()
df["stoch_d"] = stoch.stoch_signal()

# =============================
# Momentum
# =============================

df["roc10"] = ta.momentum.ROCIndicator(
    df["close"],
    window=10
).roc()

# =============================
# ATR Percentage
# =============================

df["atr_percent"] = df["atr14"] / df["close"]

# =============================
# Candle Range
# =============================

df["range"] = df["high"] - df["low"]

df["body_percent"] = (
    df["body"] /
    df["range"].replace(0, 1e-9)
)

# =============================
# Previous Candle
# =============================

df["prev_high"] = df["high"].shift(1)
df["prev_low"] = df["low"].shift(1)
df["prev_close"] = df["close"].shift(1)

# =============================
# Rolling Statistics
# =============================

df["rolling_high20"] = df["high"].rolling(20).max()
df["rolling_low20"] = df["low"].rolling(20).min()

df["rolling_mean20"] = df["close"].rolling(20).mean()

# =============================
# Volume
# =============================

df["volume_ma20"] = df["tick_volume"].rolling(20).mean()

df["volume_ratio"] = (
    df["tick_volume"] /
    df["volume_ma20"]
)

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