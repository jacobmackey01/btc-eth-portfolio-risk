"""Download daily USD candles for BTC and ETH.

Primary source: Coinbase Exchange public candles (no key, 300 candles per
request). Reference source: Bitstamp OHLC, used only to cross-check the
primary series in `validate.py`.

Run:  python -m portfolio_risk.fetch
"""
from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone

import pandas as pd
import requests

from . import config

COINBASE_URL = "https://api.exchange.coinbase.com/products/{product}/candles"
BITSTAMP_URL = "https://www.bitstamp.net/api/v2/ohlc/{pair}/"
DAY = 86_400


def _utc(date: str) -> datetime:
    return datetime.strptime(date, "%Y-%m-%d").replace(tzinfo=timezone.utc)


def fetch_coinbase(product: str, start: str, end: str) -> pd.DataFrame:
    """Daily candles for [start, end] inclusive, oldest first."""
    rows = []
    cursor, stop = _utc(start), _utc(end) + timedelta(days=1)
    while cursor < stop:
        window_end = min(cursor + timedelta(days=299), stop - timedelta(days=1))
        resp = requests.get(
            COINBASE_URL.format(product=product),
            params={
                "granularity": DAY,
                "start": cursor.isoformat(),
                "end": window_end.isoformat(),
            },
            headers={"User-Agent": "portfolio-risk-study"},
            timeout=30,
        )
        resp.raise_for_status()
        rows.extend(resp.json())
        cursor = window_end + timedelta(days=1)
        time.sleep(0.4)  # stay well under the public rate limit

    df = pd.DataFrame(rows, columns=["ts", "low", "high", "open", "close", "volume"])
    df["date"] = pd.to_datetime(df["ts"], unit="s", utc=True).dt.strftime("%Y-%m-%d")
    df = df.drop(columns="ts").drop_duplicates("date").sort_values("date")
    df = df[(df["date"] >= start) & (df["date"] <= end)]
    return df[["date", "open", "high", "low", "close", "volume"]].reset_index(drop=True)


def fetch_bitstamp(pair: str, start: str, end: str) -> pd.DataFrame:
    rows = []
    cursor, stop = int(_utc(start).timestamp()), int(_utc(end).timestamp())
    while cursor <= stop:
        resp = requests.get(
            BITSTAMP_URL.format(pair=pair),
            params={"step": DAY, "limit": 1000, "start": cursor},
            timeout=30,
        )
        resp.raise_for_status()
        batch = resp.json()["data"]["ohlc"]
        if not batch:
            break
        rows.extend(batch)
        cursor = int(batch[-1]["timestamp"]) + DAY
        time.sleep(0.4)

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["timestamp"].astype(int), unit="s", utc=True).dt.strftime("%Y-%m-%d")
    for col in ("open", "high", "low", "close", "volume"):
        df[col] = df[col].astype(float)
    df = df.drop_duplicates("date").sort_values("date")
    df = df[(df["date"] >= start) & (df["date"] <= end)]
    return df[["date", "open", "high", "low", "close", "volume"]].reset_index(drop=True)


def main() -> None:
    config.RAW_DIR.mkdir(parents=True, exist_ok=True)
    config.REFERENCE_DIR.mkdir(parents=True, exist_ok=True)
    for asset, product in config.PRODUCTS.items():
        df = fetch_coinbase(product, config.START_DATE, config.END_DATE)
        df.to_csv(config.RAW_DIR / f"{product}.csv", index=False)
        print(f"coinbase {product}: {len(df)} days {df.date.iloc[0]} → {df.date.iloc[-1]}")

        ref = fetch_bitstamp(f"{asset.lower()}usd", config.START_DATE, config.END_DATE)
        ref.to_csv(config.REFERENCE_DIR / f"bitstamp-{asset.lower()}usd.csv", index=False)
        print(f"bitstamp {asset}: {len(ref)} days {ref.date.iloc[0]} → {ref.date.iloc[-1]}")


if __name__ == "__main__":
    main()
