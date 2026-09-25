WITH valid_days AS (
    SELECT *, report_date BETWEEN DATE '{month}-01' AND DATE '{month}-14' AS is_past,
              report_date BETWEEN DATE '{month}-17' AND DATE '{month}-30' AS is_future
    FROM march
    WHERE gsc_data_available IS TRUE
      AND gsc_impressions IS NOT NULL AND gsc_impressions >= 0
      AND gsc_clicks IS NOT NULL AND gsc_clicks >= 0
), aggregated AS (
    SELECT client_hash_id, content_hash_id,
           COUNT(*) FILTER (WHERE is_past) AS past_valid_days,
           COUNT(*) FILTER (WHERE is_future) AS future_valid_days,
           MIN(report_date) FILTER (WHERE is_past) AS past_first,
           MAX(report_date) FILTER (WHERE is_past) AS past_last,
           MIN(report_date) FILTER (WHERE is_future) AS future_first,
           MAX(report_date) FILTER (WHERE is_future) AS future_last,
           SUM(gsc_impressions) FILTER (WHERE is_past) AS past_impressions,
           SUM(gsc_clicks) FILTER (WHERE is_past) AS past_clicks,
           SUM(gsc_avg_position * gsc_impressions)
               FILTER (WHERE is_past AND gsc_avg_position >= 1) AS weighted_position_sum,
           SUM(gsc_impressions)
               FILTER (WHERE is_past AND gsc_avg_position >= 1) AS position_impressions,
           COUNT(*) FILTER (WHERE is_past AND gsc_impressions > 0) AS past_active_days,
           STDDEV_POP(gsc_impressions) FILTER (WHERE is_past) AS past_impression_sd,
           SUM(gsc_impressions) FILTER (WHERE is_future) AS future_impressions
    FROM valid_days GROUP BY client_hash_id, content_hash_id
)
SELECT * FROM aggregated
