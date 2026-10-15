-- Raw daily candles exactly as downloaded (one row per asset per UTC day).
DROP TABLE IF EXISTS prices;
CREATE TABLE prices (
    source  TEXT NOT NULL,          -- 'coinbase' (primary) or 'bitstamp' (reference)
    asset   TEXT NOT NULL,          -- 'BTC' | 'ETH'
    date    TEXT NOT NULL,          -- YYYY-MM-DD, candle opening 00:00 UTC
    open    REAL NOT NULL,
    high    REAL NOT NULL,
    low     REAL NOT NULL,
    close   REAL NOT NULL CHECK (close > 0),
    volume  REAL NOT NULL,
    PRIMARY KEY (source, asset, date)
);
