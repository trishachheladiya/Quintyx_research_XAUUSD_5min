from datetime import datetime
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent

PREDICT_SCRIPT = ROOT / "scripts" / "predict_live.py"

print("=" * 60)
print("QUINTYX XAUUSD H1 BOT")
print("=" * 60)
print("Bot Started")
print("Press CTRL+C to stop.")
print("=" * 60)

last_candle = None


def get_last_closed_hour():
    now = datetime.utcnow()
    return now.replace(minute=0, second=0, microsecond=0)


while True:

    try:

        candle = get_last_closed_hour()

        if candle != last_candle:

            print()
            print("=" * 60)
            print(f"Running Prediction : {candle}")
            print("=" * 60)

            subprocess.run(
                ["python", str(PREDICT_SCRIPT)],
                check=False
            )

            last_candle = candle

        time.sleep(10)

    except KeyboardInterrupt:

        print("\nBot Stopped")
        break

    except Exception as e:

        print(e)
        time.sleep(30)