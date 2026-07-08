from pathlib import Path
import MetaTrader5 as mt5
import pandas as pd
import joblib

from ta.trend import EMAIndicator, ADXIndicator, MACD, CCIIndicator
from ta.momentum import (
    RSIIndicator,
    ROCIndicator,
    StochasticOscillator,
    WilliamsRIndicator,
)
from ta.volatility import AverageTrueRange, BollingerBands

ROOT = Path(__file__).resolve().parent.parent

BUY_MODEL = ROOT / "models" / "buy_model.pkl"
SELL_MODEL = ROOT / "models" / "sell_model.pkl"

print("=" * 60)
print("QUINTYX LIVE AI")
print("=" * 60)

# ------------------------------------
# CONNECT MT5
# ------------------------------------

if not mt5.initialize():
    raise RuntimeError(mt5.last_error())

print("Connected to MT5")

SYMBOL = "XAUUSD"

if not mt5.symbol_select(SYMBOL, True):
    raise RuntimeError("Cannot select symbol.")

# Download enough history for indicators

rates = mt5.copy_rates_from_pos(
    SYMBOL,
    mt5.TIMEFRAME_H1,
    0,
    500
)

mt5.shutdown()

if rates is None:
    raise RuntimeError("No candles downloaded.")

df = pd.DataFrame(rates)

df["datetime"] = pd.to_datetime(df["time"], unit="s")

print(f"Downloaded {len(df)} candles")

# ==========================================================
# FEATURE ENGINEERING
# ==========================================================

print("Calculating Features...")

# EMA

df["ema20"] = EMAIndicator(df["close"], 20).ema_indicator()
df["ema50"] = EMAIndicator(df["close"], 50).ema_indicator()
df["ema100"] = EMAIndicator(df["close"], 100).ema_indicator()
df["ema200"] = EMAIndicator(df["close"], 200).ema_indicator()

# RSI

df["rsi14"] = RSIIndicator(df["close"]).rsi()

# ATR

atr = AverageTrueRange(
    df["high"],
    df["low"],
    df["close"]
)

df["atr14"] = atr.average_true_range()

# ADX

adx = ADXIndicator(
    df["high"],
    df["low"],
    df["close"]
)

df["adx14"] = adx.adx()

# MACD

macd = MACD(df["close"])

df["macd"] = macd.macd()
df["macd_signal"] = macd.macd_signal()
df["macd_hist"] = macd.macd_diff()

# Bollinger

bb = BollingerBands(df["close"])

df["bb_upper"] = bb.bollinger_hband()
df["bb_lower"] = bb.bollinger_lband()
df["bb_width"] = bb.bollinger_wband()

# Stochastic

stoch = StochasticOscillator(
    df["high"],
    df["low"],
    df["close"]
)

df["stoch"] = stoch.stoch()

# CCI

cci = CCIIndicator(
    df["high"],
    df["low"],
    df["close"]
)

df["cci"] = cci.cci()

# ROC

df["roc"] = ROCIndicator(df["close"]).roc()

# Williams

df["williams"] = WilliamsRIndicator(
    df["high"],
    df["low"],
    df["close"]
).williams_r()

# EMA Distance

df["dist_ema20"] = df["close"] - df["ema20"]
df["dist_ema50"] = df["close"] - df["ema50"]
df["dist_ema100"] = df["close"] - df["ema100"]
df["dist_ema200"] = df["close"] - df["ema200"]

# EMA Slopes

df["ema20_slope"] = df["ema20"].diff()
df["ema50_slope"] = df["ema50"].diff()
df["ema100_slope"] = df["ema100"].diff()
df["ema200_slope"] = df["ema200"].diff()

# Returns

for n in [1, 3, 5, 10, 20]:
    df[f"return_{n}"] = df["close"].pct_change(n)

# Volatility

df["volatility20"] = (
    df["close"]
    .pct_change()
    .rolling(20)
    .std()
)

# Rolling High Low

df["high20"] = df["high"].rolling(20).max()
df["low20"] = df["low"].rolling(20).min()

df["high50"] = df["high"].rolling(50).max()
df["low50"] = df["low"].rolling(50).min()

# Candle Structure

df["body"] = abs(df["close"] - df["open"])

df["upper_wick"] = (
    df["high"]
    - df[["open", "close"]].max(axis=1)
)

df["lower_wick"] = (
    df[["open", "close"]].min(axis=1)
    - df["low"]
)

# Time

df["hour"] = df["datetime"].dt.hour
df["day_of_week"] = df["datetime"].dt.dayofweek
df["month"] = df["datetime"].dt.month
df["quarter"] = df["datetime"].dt.quarter

df.dropna(inplace=True)

print("Features Complete")

# Use the last CLOSED candle (ignore the current forming candle)
latest = df.iloc[[-2]].copy()

print("Latest CLOSED Candle")

print(
    latest[
        [
            "datetime",
            "close",
            "rsi14",
            "adx14",
            "atr14",
        ]
    ]
)

# ==========================================================
# LOAD MODELS
# ==========================================================

buy_model = joblib.load(BUY_MODEL)
sell_model = joblib.load(SELL_MODEL)

drop_cols = [
    "time",          # <-- ADD THIS
    "datetime",
    "buy_label",
    "sell_label",
    "buy_entry",
    "buy_sl",
    "buy_tp",
    "sell_entry",
    "sell_sl",
    "sell_tp",
]

features = [c for c in latest.columns if c not in drop_cols]

X = latest[features]

# ==========================================================
# PREDICTIONS
# ==========================================================

print("\nModel Input Columns:")
print(list(X.columns))

buy_prob = buy_model.predict_proba(X)[0][1]
sell_prob = sell_model.predict_proba(X)[0][1]

entry = latest["close"].iloc[0]
atr = latest["atr14"].iloc[0]

print()
print("=" * 60)
print("AI PREDICTION")
print("=" * 60)

print(f"BUY Probability  : {buy_prob*100:.2f}%")
print(f"SELL Probability : {sell_prob*100:.2f}%")

BUY_THRESHOLD = 0.70
SELL_THRESHOLD = 0.80

signal = "NO TRADE"

if buy_prob >= BUY_THRESHOLD and buy_prob > sell_prob:
    signal = "BUY"

elif sell_prob >= SELL_THRESHOLD and sell_prob > buy_prob:
    signal = "SELL"

ATR_MULT = 1.5
RR = 2.0

sl = None
tp = None

if signal == "BUY":
    sl = entry - ATR_MULT * atr
    tp = entry + ATR_MULT * RR * atr

elif signal == "SELL":
    sl = entry + ATR_MULT * atr
    tp = entry - ATR_MULT * RR * atr

print()
print("=" * 60)
print("FINAL SIGNAL")
print("=" * 60)

print(f"Signal      : {signal}")

if signal != "NO TRADE":
    print(f"Entry Price : {entry:.2f}")
    print(f"Stop Loss  : {sl:.2f}")
    print(f"Take Profit: {tp:.2f}")
    print(f"ATR(14)    : {atr:.2f}")
    print(f"Risk Reward: 1:2")
    print(f"Candle Time: {latest['datetime'].iloc[0]}")