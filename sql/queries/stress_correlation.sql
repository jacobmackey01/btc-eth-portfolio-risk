-- Does diversification between BTC and ETH hold up when it is needed?
-- Correlation of daily log returns split by market regime, where the regime
-- is set by the BTC return that day. Pearson correlation from sums.
-- Conditioning on a return range shrinks every within-bucket correlation, so
-- compare the down-5%+ bucket with the up-5%+ bucket (same width), not with
-- 'All days'. same_direction = share of days ETH moved the same way as BTC.
WITH tagged AS (
    SELECT btc_log AS x, eth_log AS y,
           CASE
               WHEN btc_ret <= -0.05 THEN '1. BTC down 5%+'
               WHEN btc_ret <  0     THEN '2. BTC down 0-5%'
               WHEN btc_ret <  0.05  THEN '3. BTC up 0-5%'
               ELSE                       '4. BTC up 5%+'
           END AS regime
    FROM daily
    WHERE date >= :start AND date <= :end
),
grouped AS (
    SELECT regime, COUNT(*) AS n,
           SUM(x) sx, SUM(y) sy, SUM(x*x) sxx, SUM(y*y) syy, SUM(x*y) sxy,
           AVG(x) AS avg_btc, AVG(y) AS avg_eth, AVG((x > 0) = (y > 0)) AS same_direction
    FROM tagged GROUP BY regime
    UNION ALL
    SELECT '0. All days', COUNT(*), SUM(x), SUM(y), SUM(x*x), SUM(y*y), SUM(x*y), AVG(x), AVG(y), AVG((x > 0) = (y > 0))
    FROM tagged
)
SELECT regime, n AS days, avg_btc, avg_eth, same_direction,
       (n * sxy - sx * sy) / SQRT((n * sxx - sx * sx) * (n * syy - sy * sy)) AS correlation
FROM grouped
ORDER BY regime;
