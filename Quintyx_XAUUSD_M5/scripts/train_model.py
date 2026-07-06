from pathlib import Path
import joblib
import pandas as pd

from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent.parent

DATA = ROOT / "data" / "processed" / "xauusd_m5_ml_dataset.parquet"

MODELS = ROOT / "models"
MODELS.mkdir(exist_ok=True)

df = pd.read_parquet(DATA)

DROP_COLUMNS = [
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

X = df.drop(columns=DROP_COLUMNS)

# ---------------- BUY ----------------

y = df["buy_label"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    shuffle=False,
)

model = XGBClassifier(
    n_estimators=500,
    learning_rate=0.03,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    eval_metric="logloss",
)

model.fit(X_train, y_train)

pred = model.predict(X_test)

print("=" * 60)
print("BUY MODEL")
print("=" * 60)

print("Accuracy:", accuracy_score(y_test, pred))
print(classification_report(y_test, pred))

joblib.dump(model, MODELS / "buy_model.pkl")

# ---------------- SELL ----------------

y = df["sell_label"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    shuffle=False,
)

model = XGBClassifier(
    n_estimators=500,
    learning_rate=0.03,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    eval_metric="logloss",
)

model.fit(X_train, y_train)

pred = model.predict(X_test)

print("=" * 60)
print("SELL MODEL")
print("=" * 60)

print("Accuracy:", accuracy_score(y_test, pred))
print(classification_report(y_test, pred))

joblib.dump(model, MODELS / "sell_model.pkl")

print()
print("Models saved successfully.")