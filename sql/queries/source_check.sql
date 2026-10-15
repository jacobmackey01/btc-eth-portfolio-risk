-- Cross-check the primary Coinbase closes against Bitstamp on overlapping
-- days: size of price disagreement and agreement of daily returns.
WITH paired AS (
    SELECT c.asset, c.date,
           c.close AS cb, b.close AS bs,
           c.close / LAG(c.close) OVER w - 1 AS cb_ret,
           b.close / LAG(b.close) OVER w - 1 AS bs_ret
    FROM prices c
    JOIN prices b ON b.source = 'bitstamp' AND b.asset = c.asset AND b.date = c.date
    WHERE c.source = 'coinbase'
    WINDOW w AS (PARTITION BY c.asset ORDER BY c.date)
),
stats AS (
    SELECT asset, COUNT(*) AS n, MIN(date) AS first_day, MAX(date) AS last_day,
           AVG(ABS(cb / bs - 1))                     AS mean_abs_gap,
           MAX(ABS(cb / bs - 1))                     AS max_abs_gap,
           AVG(ABS(cb / bs - 1) > 0.01)              AS share_days_gap_over_1pct,
           SUM(cb_ret) sx, SUM(bs_ret) sy, SUM(cb_ret * cb_ret) sxx,
           SUM(bs_ret * bs_ret) syy, SUM(cb_ret * bs_ret) sxy,
           COUNT(cb_ret) AS m
    FROM paired
    GROUP BY asset
)
SELECT asset, n AS overlapping_days, first_day, last_day,
       mean_abs_gap, max_abs_gap, share_days_gap_over_1pct,
       (m * sxy - sx * sy) / SQRT((m * sxx - sx * sx) * (m * syy - sy * sy)) AS return_correlation
FROM stats;
