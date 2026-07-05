import MetaTrader5 as mt5
from datetime import datetime

if not mt5.initialize():
    print(mt5.last_error())
    quit()

print("Connected:", mt5.version())

symbol = "XAUUSD"

print("Symbol info:", mt5.symbol_info(symbol))

rates = mt5.copy_rates_from_pos(
    symbol,
    mt5.TIMEFRAME_M5,
    0,
    10
)

print("Last Error:", mt5.last_error())
print("Rates:", rates)

mt5.shutdown()