from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

INPUT = ROOT / "data" / "features" / "xauusd_h1_features.parquet"
OUTPUT = ROOT / "data" / "processed" / "xauusd_h1_ml_dataset.parquet"

OUTPUT.parent.mkdir(parents=True, exist_ok=True)

df = pd.read_parquet(INPUT)

print("=" * 60)
print("QUINTYX H1 LABEL GENERATOR")
print("=" * 60)

# --------------------------------
# PARAMETERS
# --------------------------------

LOOKAHEAD = 24      # 24 H1 candles = 1 day
ATR_MULT = 1.5
RR = 2.0

buy_labels = []
sell_labels = []

buy_entry = []
buy_sl = []
buy_tp = []

sell_entry = []
sell_sl = []
sell_tp = []

for i in range(len(df)):

    if i + LOOKAHEAD >= len(df):

        buy_labels.append(None)
        sell_labels.append(None)

        buy_entry.append(None)
        buy_sl.append(None)
        buy_tp.append(None)

        sell_entry.append(None)
        sell_sl.append(None)
        sell_tp.append(None)

        continue

    entry = df.iloc[i]["close"]
    atr = df.iloc[i]["atr14"]

    future = df.iloc[i + 1:i + LOOKAHEAD + 1]

    # ---------------- BUY ----------------

    sl_buy = entry - ATR_MULT * atr
    tp_buy = entry + RR * ATR_MULT * atr

    buy = None

    for _, row in future.iterrows():

        if row["low"] <= sl_buy:
            buy = 0
            break

        if row["high"] >= tp_buy:
            buy = 1
            break

    if buy is None:
        buy = 0

    # ---------------- SELL ----------------

    sl_sell = entry + ATR_MULT * atr
    tp_sell = entry - RR * ATR_MULT * atr

    sell = None

    for _, row in future.iterrows():

        if row["high"] >= sl_sell:
            sell = 0
            break

        if row["low"] <= tp_sell:
            sell = 1
            break

    if sell is None:
        sell = 0

    buy_labels.append(buy)
    sell_labels.append(sell)

    buy_entry.append(entry)
    buy_sl.append(sl_buy)
    buy_tp.append(tp_buy)

    sell_entry.append(entry)
    sell_sl.append(sl_sell)
    sell_tp.append(tp_sell)

df["buy_label"] = buy_labels
df["sell_label"] = sell_labels

df["buy_entry"] = buy_entry
df["buy_sl"] = buy_sl
df["buy_tp"] = buy_tp

df["sell_entry"] = sell_entry
df["sell_sl"] = sell_sl
df["sell_tp"] = sell_tp

df.dropna(inplace=True)

df.to_parquet(OUTPUT, index=False)

print()
print("BUY Win Rate :", round(df["buy_label"].mean() * 100, 2), "%")
print("SELL Win Rate:", round(df["sell_label"].mean() * 100, 2), "%")
print()
print("Rows:", len(df))
print()
print("Saved to")
print(OUTPUT)
