-- Derived daily features for the primary (Coinbase) series.
-- Every value on date t uses prices up to and including the close of t only;
-- the backtest applies the execution lag on top of this.

-- 1. Complete daily calendar, carrying the last close forward over gaps.
DROP TABLE IF EXISTS prices_filled;
CREATE TABLE prices_filled AS
WITH RECURSIVE calendar(date) AS (
    SELECT MIN(date) FROM prices WHERE source = 'coinbase'
    UNION ALL
    SELECT DATE(date, '+1 day') FROM calendar
    WHERE date < (SELECT MAX(date) FROM prices WHERE source = 'coinbase')
),
grid AS (
    SELECT a.asset, c.date, p.close
    FROM calendar c
    CROSS JOIN (SELECT DISTINCT asset FROM prices WHERE source = 'coinbase') a
    LEFT JOIN prices p
        ON p.source = 'coinbase' AND p.asset = a.asset AND p.date = c.date
),
grouped AS (
    -- each observed close starts a group; gaps inherit the group before them
    SELECT *, COUNT(close) OVER (PARTITION BY asset ORDER BY date) AS grp
    FROM grid
)
SELECT asset, date,
       MAX(close) OVER (PARTITION BY asset, grp) AS close,
       close IS NULL                              AS is_filled
FROM grouped
WHERE grp > 0;

-- 2. Daily simple and log returns.
DROP TABLE IF EXISTS returns;
CREATE TABLE returns AS
SELECT asset, date, close, is_filled,
       close / LAG(close) OVER w - 1   AS ret,
       LN(close / LAG(close) OVER w)   AS log_ret
FROM prices_filled
WINDOW w AS (PARTITION BY asset ORDER BY date);

-- 3. One row per day, both assets side by side.
DROP TABLE IF EXISTS daily;
CREATE TABLE daily AS
SELECT b.date,
       b.close AS btc_close, e.close AS eth_close,
       b.ret   AS btc_ret,   e.ret   AS eth_ret,
       b.log_ret AS btc_log, e.log_ret AS eth_log,
       b.is_filled OR e.is_filled AS any_filled
FROM returns b
JOIN returns e ON e.date = b.date AND e.asset = 'ETH'
WHERE b.asset = 'BTC'
  AND b.ret IS NOT NULL AND e.ret IS NOT NULL;

-- 4. Rolling volatility (annualised sample st.dev of log returns) and rolling
--    BTC/ETH correlation over trailing 30/60/90/180-day windows. Built long
--    (one row per date and window) from running sums, then pivoted wide.
--    A window only produces a row once it is full.
DROP TABLE IF EXISTS rolling_stats_long;
CREATE TABLE rolling_stats_long AS
WITH windows(n) AS (VALUES (30), (60), (90), (180)),
numbered AS (
    SELECT ROW_NUMBER() OVER (ORDER BY date) AS i, date, btc_log, eth_log
    FROM daily
),
sums AS (
    SELECT t.date, w.n,
           COUNT(*)                     AS obs,
           SUM(s.btc_log)               AS sx,
           SUM(s.eth_log)               AS sy,
           SUM(s.btc_log * s.btc_log)   AS sxx,
           SUM(s.eth_log * s.eth_log)   AS syy,
           SUM(s.btc_log * s.eth_log)   AS sxy
    FROM numbered t
    CROSS JOIN windows w
    JOIN numbered s ON s.i BETWEEN t.i - w.n + 1 AND t.i
    GROUP BY t.date, w.n
    HAVING COUNT(*) = w.n
)
SELECT date, n AS window,
       SQRT((sxx - sx * sx / obs) / (obs - 1)) * SQRT(365)   AS btc_vol,
       SQRT((syy - sy * sy / obs) / (obs - 1)) * SQRT(365)   AS eth_vol,
       (obs * sxy - sx * sy)
         / SQRT((obs * sxx - sx * sx) * (obs * syy - sy * sy)) AS corr
FROM sums;

DROP TABLE IF EXISTS rolling_stats;
CREATE TABLE rolling_stats AS
SELECT date,
       MAX(CASE WHEN window = 30  THEN btc_vol END) AS btc_vol_30,
       MAX(CASE WHEN window = 60  THEN btc_vol END) AS btc_vol_60,
       MAX(CASE WHEN window = 90  THEN btc_vol END) AS btc_vol_90,
       MAX(CASE WHEN window = 180 THEN btc_vol END) AS btc_vol_180,
       MAX(CASE WHEN window = 30  THEN eth_vol END) AS eth_vol_30,
       MAX(CASE WHEN window = 60  THEN eth_vol END) AS eth_vol_60,
       MAX(CASE WHEN window = 90  THEN eth_vol END) AS eth_vol_90,
       MAX(CASE WHEN window = 180 THEN eth_vol END) AS eth_vol_180,
       MAX(CASE WHEN window = 30  THEN corr END)    AS corr_30,
       MAX(CASE WHEN window = 60  THEN corr END)    AS corr_60,
       MAX(CASE WHEN window = 90  THEN corr END)    AS corr_90,
       MAX(CASE WHEN window = 180 THEN corr END)    AS corr_180
FROM rolling_stats_long
GROUP BY date;

-- 5. Drawdown from running peak for each asset.
DROP TABLE IF EXISTS asset_drawdowns;
CREATE TABLE asset_drawdowns AS
SELECT asset, date, close,
       MAX(close) OVER (PARTITION BY asset ORDER BY date
                        ROWS UNBOUNDED PRECEDING)               AS peak,
       close / MAX(close) OVER (PARTITION BY asset ORDER BY date
                                ROWS UNBOUNDED PRECEDING) - 1   AS drawdown
FROM prices_filled;
