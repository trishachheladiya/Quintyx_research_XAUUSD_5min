from pathlib import Path
import joblib
import pandas as pd

from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parent.parent

DATA = ROOT / "data" / "processed" / "xauusd_h1_ml_dataset.parquet"
MODEL_DIR = ROOT / "models"

MODEL_DIR.mkdir(exist_ok=True)

print("=" * 60)
print("QUINTYX H1 MODEL TRAINER")
print("=" * 60)

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

train = df.iloc[:split]
test = df.iloc[split:]

X_train = train[features]
X_test = test[features]

# -------------------------------------------------
# BUY MODEL
# -------------------------------------------------

print("\nBUY MODEL")
print("-" * 60)

y_train = train["buy_label"]
y_test = test["buy_label"]

scale = (len(y_train) - y_train.sum()) / y_train.sum()

buy_model = XGBClassifier(
    n_estimators=400,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="binary:logistic",
    eval_metric="logloss",
    random_state=42,
    scale_pos_weight=scale,
)

buy_model.fit(X_train, y_train)

pred = buy_model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, pred))
print(classification_report(y_test, pred))

joblib.dump(
    buy_model,
    MODEL_DIR / "buy_model.pkl"
)

# -------------------------------------------------
# SELL MODEL
# -------------------------------------------------

print("\nSELL MODEL")
print("-" * 60)

y_train = train["sell_label"]
y_test = test["sell_label"]

scale = (len(y_train) - y_train.sum()) / y_train.sum()

sell_model = XGBClassifier(
    n_estimators=400,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="binary:logistic",
    eval_metric="logloss",
    random_state=42,
    scale_pos_weight=scale,
)

sell_model.fit(X_train, y_train)

pred = sell_model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, pred))
print(classification_report(y_test, pred))

joblib.dump(
    sell_model,
    MODEL_DIR / "sell_model.pkl"
)

print("\nModels saved successfully.")