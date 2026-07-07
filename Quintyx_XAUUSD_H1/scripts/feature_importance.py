from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

DATA = ROOT / "data" / "processed" / "xauusd_h1_ml_dataset.parquet"

BUY_MODEL = ROOT / "models" / "buy_model.pkl"
SELL_MODEL = ROOT / "models" / "sell_model.pkl"

REPORTS = ROOT / "reports"
REPORTS.mkdir(exist_ok=True)

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

buy_model = joblib.load(BUY_MODEL)
sell_model = joblib.load(SELL_MODEL)

buy_importance = pd.DataFrame({
    "Feature": features,
    "Importance": buy_model.feature_importances_
}).sort_values(
    by="Importance",
    ascending=False
)

sell_importance = pd.DataFrame({
    "Feature": features,
    "Importance": sell_model.feature_importances_
}).sort_values(
    by="Importance",
    ascending=False
)

print("=" * 60)
print("BUY FEATURE IMPORTANCE")
print("=" * 60)

print(buy_importance.head(20))

print()

print("=" * 60)
print("SELL FEATURE IMPORTANCE")
print("=" * 60)

print(sell_importance.head(20))

buy_importance.to_csv(
    REPORTS / "buy_feature_importance.csv",
    index=False
)

sell_importance.to_csv(
    REPORTS / "sell_feature_importance.csv",
    index=False
)

print()
print("Reports saved.")