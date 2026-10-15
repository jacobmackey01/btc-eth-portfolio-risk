-- Calendar-year return, annualised volatility, worst drawdown and average
-- 90-day correlation for each asset. Feeds the market-context table.
WITH yearly AS (
    SELECT r.asset,
           STRFTIME('%Y', r.date)                          AS year,
           EXP(SUM(r.log_ret)) - 1                         AS total_return,
           SQRT((SUM(r.log_ret * r.log_ret)
                 - SUM(r.log_ret) * SUM(r.log_ret) / COUNT(*))
                / (COUNT(*) - 1)) * SQRT(365)              AS ann_vol,
           COUNT(*)                                        AS days
    FROM returns r
    WHERE r.log_ret IS NOT NULL
    GROUP BY r.asset, year
),
yearly_dd AS (
    SELECT asset, STRFTIME('%Y', date) AS year, MIN(drawdown) AS worst_drawdown
    FROM asset_drawdowns
    GROUP BY asset, year
),
yearly_corr AS (
    SELECT STRFTIME('%Y', date) AS year, AVG(corr_90) AS avg_corr_90
    FROM rolling_stats
    GROUP BY year
)
SELECT y.year, y.asset, y.days, y.total_return, y.ann_vol,
       d.worst_drawdown, c.avg_corr_90
FROM yearly y
JOIN yearly_dd d USING (asset, year)
LEFT JOIN yearly_corr c USING (year)
ORDER BY y.year, y.asset;
