from pathlib import Path

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent.parent

DATA = ROOT / "data" / "processed" / "xauusd_m5_ml_dataset.parquet"

MODEL = ROOT / "models" / "buy_model.pkl"

df = pd.read_parquet(DATA)

DROP = [
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

X = df.drop(columns=DROP)
y = df["buy_label"]

_, X_test, _, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    shuffle=False,
)

model = joblib.load(MODEL)

prob = model.predict_proba(X_test)[:, 1]

print("=" * 60)
print("THRESHOLD ANALYSIS")
print("=" * 60)

for threshold in range(50, 100, 5):

    t = threshold / 100

    mask = prob >= t

    trades = mask.sum()

    if trades == 0:
        continue

    accuracy = (y_test[mask] == 1).mean()

    print(
        f"{threshold}% | Trades: {trades:5d} | Win Rate: {accuracy*100:.2f}%"
    )