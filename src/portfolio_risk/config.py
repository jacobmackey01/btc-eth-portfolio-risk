"""Study-wide settings. Everything that changes a result lives here."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"
REFERENCE_DIR = ROOT / "data" / "reference"
DB_PATH = ROOT / "data" / "prices.sqlite"
SQL_DIR = ROOT / "sql"
REPORT_DIR = ROOT / "reports"

ASSETS = ("BTC", "ETH")
PRODUCTS = {"BTC": "BTC-USD", "ETH": "ETH-USD"}

# Daily closes are 00:00 UTC Coinbase candles. The end date is pinned so the
# report reproduces exactly; re-run `make fetch` and move it to extend.
START_DATE = "2016-05-18"   # first full ETH-USD day on Coinbase
END_DATE = "2026-10-14"     # last complete UTC day when the data was pulled

# Chronological split. Parameters are chosen on TRAIN only; TEST is touched
# once, after every choice is fixed. The first 180 days are warm-up so the
# longest volatility lookback is populated before trading starts.
WARMUP_DAYS = 180
TRAIN_END = "2021-12-31"
TEST_START = "2022-01-01"

DAYS_PER_YEAR = 365          # crypto trades every day
COST_BPS = 25                # base-case cost per unit of notional traded
COST_GRID_BPS = (0, 10, 25, 50, 100)
EXECUTION_LAG = 1            # signal at close t, trade at close t+1
