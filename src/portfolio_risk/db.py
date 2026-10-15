"""Load the raw CSVs into SQLite, build the SQL feature tables, read them back.

Run:  python -m portfolio_risk.db
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd

from . import config

REFERENCE_FILES = {"BTC": "bitstamp-btcusd.csv", "ETH": "bitstamp-ethusd.csv"}


def connect(path: Path = config.DB_PATH) -> sqlite3.Connection:
    return sqlite3.connect(path)


def run_sql_file(con: sqlite3.Connection, name: str) -> None:
    con.executescript((config.SQL_DIR / name).read_text())


def load_raw(con: sqlite3.Connection) -> None:
    frames = []
    for asset, product in config.PRODUCTS.items():
        df = pd.read_csv(config.RAW_DIR / f"{product}.csv")
        frames.append(df.assign(source="coinbase", asset=asset))
        ref = config.REFERENCE_DIR / REFERENCE_FILES[asset]
        if ref.exists():
            frames.append(pd.read_csv(ref).assign(source="bitstamp", asset=asset))
    cols = ["source", "asset", "date", "open", "high", "low", "close", "volume"]
    rows = pd.concat(frames)[cols].itertuples(index=False, name=None)
    con.executemany(f"INSERT INTO prices VALUES ({','.join('?' * len(cols))})", rows)
    con.commit()


def build(path: Path = config.DB_PATH) -> None:
    path.unlink(missing_ok=True)
    with connect(path) as con:
        run_sql_file(con, "schema.sql")
        load_raw(con)
        run_sql_file(con, "features.sql")


def query(sql: str, path: Path = config.DB_PATH) -> pd.DataFrame:
    with connect(path) as con:
        return pd.read_sql_query(sql, con)


def query_file(name: str, path: Path = config.DB_PATH) -> pd.DataFrame:
    return query((config.SQL_DIR / "queries" / name).read_text(), path)


def load_panel(path: Path = config.DB_PATH) -> pd.DataFrame:
    """Daily returns joined to the rolling statistics, indexed by date."""
    df = query(
        """
        SELECT d.*, r.*
        FROM daily d
        LEFT JOIN rolling_stats r USING (date)
        ORDER BY d.date
        """,
        path,
    )
    df = df.loc[:, ~df.columns.duplicated()]
    df["date"] = pd.to_datetime(df["date"])
    return df.set_index("date")


if __name__ == "__main__":
    build()
    with connect() as con:
        for table in ("prices", "prices_filled", "daily", "rolling_stats_long", "rolling_stats"):
            (n,) = con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()
            print(f"{table:20s} {n:>7,d} rows")
