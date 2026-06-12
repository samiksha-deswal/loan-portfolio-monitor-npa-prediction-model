-- ============================================================
-- LOAN PORTFOLIO — SQL ANALYSIS QUERIES
-- loan_portfolio database | MySQL
-- ============================================================


-- ============================================================
-- QUERY 1: PORTFOLIO OVERVIEW
-- Business Question: What is the current state of the portfolio?
-- ============================================================

SELECT
    dpd_bucket,
    COUNT(*) AS loan_count,
    ROUND(SUM(outstanding_principal) / 10000000, 2) AS outstanding_cr,
    ROUND(AVG(interest_rate), 2)  AS avg_interest_rate,
    ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 2) AS pct_of_portfolio
FROM loans
WHERE dpd_bucket != 'Closed'
GROUP BY dpd_bucket
ORDER BY FIELD(dpd_bucket, 'Current', '30 DPD', '60 DPD', '90+ DPD');


-- ============================================================
-- QUERY 2: NPA RATE BY LOAN TYPE
-- Business Question: Which loan type is driving the most stress?
-- ============================================================

SELECT
    loan_type,
    COUNT(*) AS total_loans,
    SUM(CASE WHEN dpd_bucket = '90+ DPD' THEN 1 ELSE 0 END) AS npa_count,
    ROUND(SUM(CASE WHEN dpd_bucket = '90+ DPD'
                   THEN outstanding_principal ELSE 0 END)
          / SUM(outstanding_principal) * 100, 2) AS gross_npa_pct,
    ROUND(AVG(interest_rate), 2) AS avg_rate,
    ROUND(SUM(outstanding_principal) / 10000000, 2) AS aum_cr
FROM loans
WHERE dpd_bucket != 'Closed'
GROUP BY loan_type
ORDER BY gross_npa_pct DESC;


-- ============================================================
-- QUERY 3: COLLECTION EFFICIENCY BY SEGMENT
-- Business Question: Are we collecting what we expect to, segment by segment?
-- ============================================================

SELECT
    l.loan_type,
    l.dpd_bucket,
    COUNT(DISTINCT l.loan_id) AS loan_count,
    ROUND(SUM(mp.expected) / 10000000, 2) AS expected_cr,
    ROUND(SUM(mp.collected) / 10000000, 2) AS collected_cr,
    ROUND(SUM(mp.collected) / SUM(mp.expected) * 100, 2) AS collection_efficiency_pct
FROM loans l
JOIN monthly_payments mp ON l.loan_id = mp.loan_id
WHERE mp.month_offset = 1
  AND l.dpd_bucket != 'Closed'
GROUP BY l.loan_type, l.dpd_bucket
ORDER BY l.loan_type,
         FIELD(l.dpd_bucket, 'Current', '30 DPD', '60 DPD', '90+ DPD');


-- ============================================================
-- QUERY 4: VINTAGE ANALYSIS — NPA RATE BY LOAN AGE & TYPE
-- Business Question: Does NPA risk increase or decrease as loans age?
-- ============================================================

SELECT
    vintage,
    loan_type,
    COUNT(*) AS loans,
    ROUND(AVG(base_risk_score) * 100, 2) AS avg_risk_score,
    SUM(CASE WHEN dpd_bucket = '90+ DPD' THEN 1 ELSE 0 END) AS npa_count,
    ROUND(SUM(CASE WHEN dpd_bucket = '90+ DPD' THEN 1 ELSE 0 END)
          / COUNT(*) * 100, 2) AS actual_npa_rate_pct,
    ROUND(SUM(outstanding_principal) / 10000000, 2) AS aum_cr
FROM loans
WHERE dpd_bucket != 'Closed'
GROUP BY vintage, loan_type
ORDER BY FIELD(vintage, '0-6M', '6-12M', '12-24M', '24M+'), loan_type;


-- ============================================================
-- QUERY 5: EARLY WARNING — CHRONIC DEFAULTERS
-- Business Question: Which borrowers have been consistently missing payments across multiple months, not just last month?
-- ============================================================
SELECT
    mp.loan_id, l.loan_type, l.ticket_size, l.dpd_bucket,
    ROUND(l.outstanding_principal / 100000, 2) AS outstanding_lakh,
    COUNT(mp.month_offset) AS months_tracked,
    SUM(CASE WHEN mp.collected = 0 THEN 1 ELSE 0 END) AS months_zero_payment,
    SUM(CASE WHEN mp.collected > 0
             AND mp.collected < mp.expected THEN 1 ELSE 0 END) AS months_partial,
    ROUND(SUM(mp.collected)
          / NULLIF(SUM(mp.expected), 0) * 100, 2)  AS overall_collection_pct,
    MAX(mp.delayed_days)  AS max_delay_days
FROM monthly_payments mp
JOIN loans l ON mp.loan_id = l.loan_id
WHERE l.dpd_bucket != 'Closed'
GROUP BY mp.loan_id, l.loan_type, l.ticket_size,
         l.dpd_bucket, l.outstanding_principal
HAVING months_zero_payment >= 2
    OR overall_collection_pct < 60
ORDER BY overall_collection_pct ASC
LIMIT 50;


-- ============================================================
-- QUERY 6: ROLL RATE ANALYSIS - Business Question: How many loans are rolling from one DPD bucket to worse?
-- ============================================================
WITH payment_behaviour AS ( SELECT loan_id,
        SUM(CASE WHEN month_offset = 1 AND collected = 0        THEN 1 ELSE 0 END) AS missed_last,
        SUM(CASE WHEN month_offset = 2 AND collected = 0        THEN 1 ELSE 0 END) AS missed_prev,
        SUM(CASE WHEN month_offset = 1 AND delayed_days >= 30   THEN 1 ELSE 0 END) AS late_30_last,
        SUM(CASE WHEN month_offset = 2 AND delayed_days >= 30   THEN 1 ELSE 0 END) AS late_30_prev
    FROM monthly_payments
    GROUP BY loan_id),
risk_movement AS ( SELECT l.loan_id, l.loan_type, l.dpd_bucket AS current_bucket,
					CASE WHEN pb.missed_prev >= 1 OR pb.late_30_prev >= 1 THEN 'Was Stressed' ELSE 'Was Current' END AS previous_state
    FROM loans l
    JOIN payment_behaviour pb ON l.loan_id = pb.loan_id
    WHERE l.dpd_bucket != 'Closed')
SELECT previous_state, current_bucket, COUNT(*) AS loan_count,
    ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (PARTITION BY previous_state), 2)   AS roll_rate_pct
FROM risk_movement
GROUP BY previous_state, current_bucket
ORDER BY previous_state,
         FIELD(current_bucket, 'Current', '30 DPD', '60 DPD', '90+ DPD');


-- ============================================================
-- QUERY 7: CONCENTRATION RISK — PARETO OF NPA
-- Business Question: Which segments hold the most NPA concentration?  Are we over-exposed to any one segment?
-- ============================================================
WITH segment_npa AS ( SELECT loan_type, ticket_size,
        SUM(outstanding_principal) AS total_outstanding,
        SUM(CASE WHEN dpd_bucket = '90+ DPD' THEN outstanding_principal ELSE 0 END) AS npa_outstanding
    FROM loans
    WHERE dpd_bucket != 'Closed'
    GROUP BY loan_type, ticket_size ),
ranked AS ( SELECT *, ROUND(npa_outstanding / NULLIF(total_outstanding,0) * 100, 2)  AS npa_pct,
        ROUND(npa_outstanding / SUM(npa_outstanding) OVER () * 100, 2)  AS share_of_total_npa_pct,
        RANK() OVER (ORDER BY npa_outstanding DESC)  AS npa_rank
    FROM segment_npa
    WHERE npa_outstanding > 0 )
SELECT npa_rank, loan_type, ticket_size, ROUND(total_outstanding / 10000000, 2) AS total_aum_cr,
    ROUND(npa_outstanding  / 10000000, 2) AS npa_aum_cr,
    npa_pct, share_of_total_npa_pct,
    ROUND(SUM(share_of_total_npa_pct) OVER ( ORDER BY npa_outstanding DESC ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW), 2) AS cumulative_npa_pct
FROM ranked
ORDER BY npa_rank
LIMIT 15;


-- ============================================================
-- QUERY 8: COLLECTION EFFICIENCY TREND — LAST 3 MONTHS
-- Business Question: Is collection performance improving or deteriorating?
-- ============================================================

SELECT mp.month_offset,
    CASE mp.month_offset
        WHEN 1 THEN 'Last Month'
        WHEN 2 THEN '2 Months Ago'
        WHEN 3 THEN '3 Months Ago'
    END AS period,
    COUNT(DISTINCT mp.loan_id) AS active_loans,
    ROUND(SUM(mp.expected)  / 10000000, 2) AS expected_cr,
    ROUND(SUM(mp.collected) / 10000000, 2) AS collected_cr,
    ROUND(SUM(mp.collected) / SUM(mp.expected) * 100, 2) AS collection_efficiency_pct,
    SUM(CASE WHEN mp.collected = 0 THEN 1 ELSE 0 END) AS zero_payment_accounts
FROM monthly_payments mp
JOIN loans l ON mp.loan_id = l.loan_id
WHERE l.dpd_bucket != 'Closed'
  AND mp.month_offset <= 3
GROUP BY mp.month_offset
ORDER BY mp.month_offset;


-- ============================================================
-- QUERY 9: AT-RISK ACCOUNTS — PRE-NPA ACTION LIST
-- Business Question: Which 30-60 DPD loans are most likely to roll to 90+ DPD?  Give collections team a prioritised list.
-- ============================================================

WITH at_risk_scored AS ( SELECT l.loan_id, l.loan_type, l.ticket_size,l.dpd_bucket,
        ROUND(l.outstanding_principal / 100000, 2)AS outstanding_lakh,
        ROUND(l.interest_rate, 2) AS rate,
        ROUND(SUM(mp.collected)
              / NULLIF(SUM(mp.expected), 0) * 100, 2) AS collection_rate_3m,
        SUM(CASE WHEN mp.collected = 0 THEN 1 ELSE 0 END) AS zero_pay_months,
        MAX(mp.delayed_days) AS max_delay,
        CASE WHEN SUM(CASE WHEN mp.collected=0 THEN 1 ELSE 0 END) >= 2 OR SUM(mp.collected)/NULLIF(SUM(mp.expected),0) < 0.40 THEN 'High Roll Risk'
            WHEN SUM(CASE WHEN mp.collected=0 THEN 1 ELSE 0 END) = 1 OR SUM(mp.collected)/NULLIF(SUM(mp.expected),0) BETWEEN 0.40 AND 0.70THEN 'Medium Roll Risk'
            ELSE 'Low Roll Risk' END AS roll_risk_category
    FROM loans l
    JOIN monthly_payments mp ON l.loan_id = mp.loan_id
    WHERE l.dpd_bucket IN ('30 DPD', '60 DPD')
    GROUP BY l.loan_id, l.loan_type, l.ticket_size,
             l.dpd_bucket, l.outstanding_principal, l.interest_rate )
SELECT * FROM at_risk_scored
ORDER BY FIELD(roll_risk_category,'High Roll Risk','Medium Roll Risk','Low Roll Risk'),
         outstanding_lakh DESC;


-- ============================================================
-- QUERY 10: INTEREST RATE BAND VS NPA — UNDERWRITING SIGNAL
-- Business Question: Is our pricing risk-adjusted? Are high-rate loans actually defaulting more?
-- ============================================================

SELECT CASE
        WHEN interest_rate < 10 THEN '< 10%'
        WHEN interest_rate BETWEEN 10 AND 14 THEN '10–14%'
        WHEN interest_rate BETWEEN 14 AND 18 THEN '14–18%'
        ELSE '18%+' END  AS rate_band,
    COUNT(*)  AS loan_count,
    ROUND(AVG(loan_amount)    / 100000, 2) AS avg_ticket_lakh,
    ROUND(AVG(months_active), 1) AS avg_months_active,
    SUM(CASE WHEN dpd_bucket = '90+ DPD' THEN 1 ELSE 0 END) AS npa_count,
    ROUND(SUM(CASE WHEN dpd_bucket = '90+ DPD' THEN 1 ELSE 0 END)
          / COUNT(*) * 100, 2)  AS npa_rate_pct,
    ROUND(SUM(CASE WHEN dpd_bucket IN ('30 DPD','60 DPD')
                   THEN 1 ELSE 0 END)
          / COUNT(*) * 100, 2)  AS stress_rate_pct
FROM loans
WHERE dpd_bucket != 'Closed'
GROUP BY rate_band
ORDER BY MIN(interest_rate);


-- ============================================================
-- QUERY 11: TICKET SIZE vs RECOVERY BEHAVIOUR
-- Business Question: Do large-ticket borrowers pay differently from small ones?
-- ============================================================

SELECT
    l.ticket_size,
    COUNT(DISTINCT l.loan_id) AS loan_count,
    ROUND(AVG(l.outstanding_principal) / 100000, 2) AS avg_outstanding_lakh,
    ROUND(SUM(mp.collected) / SUM(mp.expected) * 100, 2)  AS overall_collection_pct,
    SUM(CASE WHEN mp.collected = 0 THEN 1 ELSE 0 END)  AS total_missed_payments,
    ROUND(SUM(CASE WHEN l.dpd_bucket = '90+ DPD'
                   THEN l.outstanding_principal ELSE 0 END)
          / SUM(l.outstanding_principal) * 100, 2)  AS npa_concentration_pct
FROM loans l
JOIN monthly_payments mp ON l.loan_id = mp.loan_id
WHERE l.dpd_bucket != 'Closed'
GROUP BY l.ticket_size
ORDER BY FIELD(l.ticket_size, '<2L', '2-10L', '10-50L', '50L+');


-- ============================================================
-- QUERY 12: PORTFOLIO YIELD vs NPA TRADEOFF
-- Business Question: Are we being compensated for the risk we're taking? High yield should correlate with manageable NPA.
-- ============================================================
SELECT loan_type,
    ROUND(SUM(interest_rate * outstanding_principal)
          / SUM(outstanding_principal), 2)  AS weighted_avg_yield,
    ROUND(SUM(CASE WHEN dpd_bucket = '90+ DPD'
                   THEN outstanding_principal ELSE 0 END)
          / SUM(outstanding_principal) * 100, 2)  AS gross_npa_pct,
    ROUND(SUM(interest_rate * outstanding_principal)
          / SUM(outstanding_principal)
          - SUM(CASE WHEN dpd_bucket = '90+ DPD'
                     THEN outstanding_principal ELSE 0 END)
          / SUM(outstanding_principal) * 100, 2) AS risk_adjusted_yield,
    ROUND(SUM(outstanding_principal) / 10000000, 2) AS aum_cr
FROM loans
WHERE dpd_bucket != 'Closed'
GROUP BY loan_type
ORDER BY risk_adjusted_yield DESC;
