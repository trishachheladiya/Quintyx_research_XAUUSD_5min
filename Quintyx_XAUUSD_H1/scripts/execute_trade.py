from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

from pathlib import Path
import MetaTrader5 as mt5
from config.trading_config import *

ROOT = Path(__file__).resolve().parent.parent

print("=" * 60)
print("QUINTYX EXECUTION ENGINE")
print("=" * 60)

if not mt5.initialize():
    raise RuntimeError("Cannot connect to MT5")

print("Connected to MT5")

account = mt5.account_info()

if account is None:
    raise RuntimeError("Cannot read account")

print()
print("Account Information")
print("-" * 60)

print(f"Login      : {account.login}")
print(f"Server     : {account.server}")
print(f"Balance    : {account.balance}")
print(f"Equity     : {account.equity}")
print(f"Leverage   : {account.leverage}")

symbol = mt5.symbol_info(SYMBOL)

if symbol is None:
    raise RuntimeError(f"{SYMBOL} not found.")

print()
print("Symbol Information")
print("-" * 60)

print(f"Symbol     : {symbol.name}")
print(f"Point      : {symbol.point}")
print(f"Digits     : {symbol.digits}")
print(f"Min Lot    : {symbol.volume_min}")
print(f"Max Lot    : {symbol.volume_max}")
print(f"Lot Step   : {symbol.volume_step}")