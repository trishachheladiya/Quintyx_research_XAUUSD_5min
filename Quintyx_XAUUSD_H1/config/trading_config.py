# ==========================================================
# GENERAL
# ==========================================================

SYMBOL = "XAUUSD"

TIMEFRAME = "H1"

COMMENT = "Quintyx AI"

MAGIC_NUMBER = 20260701

SLIPPAGE = 20

# ==========================================================
# RISK MANAGEMENT
# ==========================================================

RISK_PERCENT = 1.0

MAX_OPEN_TRADES = 1

# ==========================================================
# MODEL
# ==========================================================

BUY_THRESHOLD = 0.70

SELL_THRESHOLD = 0.80

# ==========================================================
# STOP LOSS / TAKE PROFIT
# ==========================================================

ATR_PERIOD = 14

ATR_MULTIPLIER = 1.5

RISK_REWARD = 2.0

# TP = ATR_MULTIPLIER × RISK_REWARD

# BUY
# SL = Entry - ATR_MULTIPLIER × ATR
# TP = Entry + ATR_MULTIPLIER × RISK_REWARD × ATR

# SELL
# SL = Entry + ATR_MULTIPLIER × ATR
# TP = Entry - ATR_MULTIPLIER × RISK_REWARD × ATR

# ==========================================================
# BOT
# ==========================================================

CHECK_NEW_CANDLE_DELAY = 5

LOG_TRADES = True