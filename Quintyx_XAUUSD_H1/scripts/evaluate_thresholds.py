from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

DATA = ROOT / "data" / "processed" / "xauusd_h1_ml_dataset.parquet"

BUY_MODEL = ROOT / "models" / "buy_model.pkl"
SELL_MODEL = ROOT / "models" / "sell_model.pkl"

df = pd.read_parquet(DATA)

drop_cols = [
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

features = [c for c in df.columns if c not in drop_cols]

split = int(len(df) * 0.8)

test = df.iloc[split:]

X = test[features]

buy_model = joblib.load(BUY_MODEL)
sell_model = joblib.load(SELL_MODEL)

buy_prob = buy_model.predict_proba(X)[:, 1]
sell_prob = sell_model.predict_proba(X)[:, 1]

print("=" * 60)
print("BUY MODEL")
print("=" * 60)

for t in range(50, 100, 5):

    threshold = t / 100

    mask = buy_prob >= threshold

    trades = mask.sum()

    if trades == 0:
        continue

    winrate = test.loc[mask, "buy_label"].mean() * 100

    print(
        f"{t}% | Trades: {trades:5d} | Win Rate: {winrate:.2f}%"
    )

print()

print("=" * 60)
print("SELL MODEL")
print("=" * 60)

for t in range(50, 100, 5):

    threshold = t / 100

    mask = sell_prob >= threshold

    trades = mask.sum()

    if trades == 0:
        continue

    winrate = test.loc[mask, "sell_label"].mean() * 100

    print(
        f"{t}% | Trades: {trades:5d} | Win Rate: {winrate:.2f}%"
    )