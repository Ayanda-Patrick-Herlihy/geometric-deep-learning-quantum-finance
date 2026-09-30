"""Downloads per-ticker stock-split histories from EODHD.

Stage 3 needs them to rebuild as-traded ("point-in-time") prices for the
tradability gate: EODHD's Close and Volume are back-adjusted for every later
split, so a stock that later split 10:1 appears ten times cheaper in the past
than it traded, and a price floor applied to those values uses information
from the future.

One CSV per ticker is written to data/raw/splits/<TICKER>.csv with columns
``date`` and ``split`` ("new/old"). Tickers without splits get an empty file
so a rerun skips them. Existing files are never re-downloaded.

The API key is read from config.yaml (api_keys.eodhd) or the EODHD_API_KEY
environment variable.

Usage:
    uv run python src/data_pipeline/stage1_fetch_splits.py
    uv run python src/data_pipeline/stage1_fetch_splits.py --tickers AAPL MSFT
"""

import argparse
import logging
import os
import time
from pathlib import Path

import pandas as pd
import requests
import yaml

try:
    from .stage1_consolidate_equities import get_equity_files
except ImportError:  # run as a script
    from stage1_consolidate_equities import get_equity_files

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SPLITS_URL = "https://eodhd.com/api/splits/{code}"
MAX_RETRIES = 4


def eodhd_code(ticker: str, exchange: str = "US") -> str:
    """EODHD symbol for a raw-file ticker (adds the exchange suffix if absent)."""
    return ticker if "." in ticker else f"{ticker}.{exchange}"


def fetch_splits(ticker: str, api_key: str, start: str, session: requests.Session) -> pd.DataFrame:
    """Returns the split history of one ticker as columns date, split."""
    params = {"api_token": api_key, "fmt": "json", "from": start}
    for attempt in range(MAX_RETRIES):
        try:
            response = session.get(SPLITS_URL.format(code=eodhd_code(ticker)), params=params, timeout=30)
            if response.status_code == 404:
                return pd.DataFrame(columns=["date", "split"])
            response.raise_for_status()
            records = response.json()
            return pd.DataFrame(records, columns=["date", "split"])
        except (requests.RequestException, ValueError) as exc:
            wait = 2 ** (attempt + 1)
            logger.warning("%s: attempt %d failed (%s); retrying in %ds.", ticker, attempt + 1, exc, wait)
            time.sleep(wait)
    raise RuntimeError(f"Could not fetch splits for {ticker} after {MAX_RETRIES} attempts.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--raw-equities", type=Path, default=PROJECT_ROOT / "data" / "raw" / "equities")
    parser.add_argument("--out", type=Path, default=PROJECT_ROOT / "data" / "raw" / "splits")
    parser.add_argument("--tickers", nargs="+", default=None, help="Default: every raw equity file.")
    parser.add_argument("--start", default="1999-01-01")
    parser.add_argument("--pause", type=float, default=0.05, help="Seconds between requests.")
    args = parser.parse_args()

    with open(PROJECT_ROOT / "config.yaml") as fh:
        config = yaml.safe_load(fh)
    api_key = os.environ.get("EODHD_API_KEY") or (config.get("api_keys") or {}).get("eodhd")
    if not api_key:
        raise SystemExit("Set EODHD_API_KEY or api_keys.eodhd in config.yaml.")

    tickers = args.tickers or [path.stem for path in get_equity_files(args.raw_equities)]
    args.out.mkdir(parents=True, exist_ok=True)
    todo = [t for t in tickers if not (args.out / f"{t}.csv").exists()]
    logger.info("%d tickers, %d already fetched, %d to fetch.", len(tickers), len(tickers) - len(todo), len(todo))

    n_with_splits = 0
    with requests.Session() as session:
        for i, ticker in enumerate(todo, 1):
            splits = fetch_splits(ticker, api_key, args.start, session)
            splits.to_csv(args.out / f"{ticker}.csv", index=False)
            n_with_splits += int(not splits.empty)
            if i % 500 == 0:
                logger.info("  %d / %d fetched (%d with splits).", i, len(todo), n_with_splits)
            time.sleep(args.pause)
    logger.info("Done: %d tickers fetched, %d had at least one split.", len(todo), n_with_splits)


if __name__ == "__main__":
    main()
