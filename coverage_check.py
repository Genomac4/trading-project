"""
coverage_check.py — DIAGNOSTIC ONLY (no logic changes).

Question: is quarterly-YoY's 2007-2012 underperformance a COVERAGE artifact
(fewer names qualify, so a thinner/different portfolio) or a real regime effect
(coverage comparable, signal just worse in that regime)?

Method: replay the exact per-month eligibility the backtest uses for the
2007-2012 slice. For each rebalance month, count the momentum-eligible universe
(in-index that month + valid 12-1 momentum), then how many of those have a
non-None ANNUAL F-score vs a non-None QUARTERLY-YoY F-score — using the same
cached_fscore / cached_fscore_quarterly the backtest calls, same June-30 as-of.

Run: python coverage_check.py
"""

import numpy as np
import pandas as pd

from explore_constituents import get_sp500_membership, members_on_date
from backtest_composite import (
    load_all_prices, cached_fscore, cached_fscore_quarterly,
)

START, END = "2007-01-01", "2012-12-31"


def main():
    print("Loading prices...")
    prices = load_all_prices()
    membership = get_sp500_membership()

    monthly = prices.resample("ME").last().loc[START:END]
    dates = monthly.index

    rows = []
    # Same loop bounds as run_backtest: needs 12 months of momentum history,
    # and the final month is not a rebalance (no forward return there).
    for i in range(12, len(dates) - 1):
        today = dates[i]
        year = today.year
        index_members = members_on_date(membership, today)

        base = ann = qtr = both = 0
        for ticker in index_members:
            if ticker not in monthly.columns:
                continue
            p_12 = monthly[ticker].iloc[i - 12]
            p_1 = monthly[ticker].iloc[i - 1]
            p_now = monthly[ticker].iloc[i]
            if (pd.isna(p_12) or pd.isna(p_1) or pd.isna(p_now)
                    or p_12 <= 0 or p_now <= 0):
                continue
            base += 1                                   # momentum-eligible
            a = cached_fscore(ticker, year) is not None
            q = cached_fscore_quarterly(ticker, year) is not None
            ann += a
            qtr += q
            both += (a and q)

        rows.append({"date": today, "year": year, "base": base,
                     "ann": ann, "qtr": qtr, "both": both})

    df = pd.DataFrame(rows)
    df["ann_pct"] = 100 * df["ann"] / df["base"]
    df["qtr_pct"] = 100 * df["qtr"] / df["base"]

    print(f"\nReplayed {len(df)} rebalance months "
          f"({df['date'].iloc[0].date()} .. {df['date'].iloc[-1].date()}).")
    print("Eligible universe = in-index that month AND valid 12-1 momentum.\n")

    print("=" * 78)
    print("COVERAGE BY YEAR — of the momentum-eligible universe, % with a "
          "valid F-score")
    print("=" * 78)
    print(f"{'year':<6}{'months':>7}{'elig(avg)':>11}"
          f"{'annual n':>10}{'annual %':>10}{'qtr n':>8}{'qtr %':>8}")
    print("-" * 78)
    for year, g in df.groupby("year"):
        print(f"{year:<6}{len(g):>7}{g['base'].mean():>11.0f}"
              f"{g['ann'].mean():>10.0f}{g['ann_pct'].mean():>9.1f}%"
              f"{g['qtr'].mean():>8.0f}{g['qtr_pct'].mean():>7.1f}%")
    print("-" * 78)
    print(f"{'ALL':<6}{len(df):>7}{df['base'].mean():>11.0f}"
          f"{df['ann'].mean():>10.0f}{df['ann_pct'].mean():>9.1f}%"
          f"{df['qtr'].mean():>8.0f}{df['qtr_pct'].mean():>7.1f}%")

    ann_avg = df["ann_pct"].mean()
    qtr_avg = df["qtr_pct"].mean()
    gap = ann_avg - qtr_avg
    print(f"\nMonthly quarterly coverage range: "
          f"{df['qtr_pct'].min():.1f}% .. {df['qtr_pct'].max():.1f}%   "
          f"(annual: {df['ann_pct'].min():.1f}% .. {df['ann_pct'].max():.1f}%)")
    print(f"Names in BOTH (avg/month): {df['both'].mean():.0f}  "
          f"({100*df['both'].mean()/df['base'].mean():.1f}% of eligible)")

    print("\n" + "=" * 78)
    print("VERDICT")
    print("=" * 78)
    print(f"Avg annual coverage:    {ann_avg:5.1f}% of eligible universe")
    print(f"Avg quarterly coverage: {qtr_avg:5.1f}% of eligible universe")
    print(f"Coverage gap (annual - quarterly): {gap:+.1f} pts\n")

    THIN = 8.0   # percentage points considered a meaningful coverage shortfall
    if gap >= THIN:
        print(f"==> Evidence toward (b) COVERAGE ARTIFACT. Quarterly-YoY covers "
              f"{gap:.1f} fewer points of the eligible universe in 2007-2012 "
              f"(~{df['ann'].mean() - df['qtr'].mean():.0f} fewer names/month), "
              f"so its portfolio is drawn from a thinner/different pool. The "
              f"underperformance is at least partly a coverage effect, not "
              f"purely signal quality.")
    elif gap <= -THIN:
        print(f"==> Quarterly coverage is actually WIDER by {-gap:.1f} pts, so a "
              f"coverage shortfall does NOT explain the underperformance — "
              f"evidence toward (a) a real regime-dependent signal effect.")
    else:
        print(f"==> Evidence toward (a) REAL SIGNAL EFFECT. Coverage is "
              f"comparable (within {abs(gap):.1f} pts / "
              f"~{abs(df['ann'].mean() - df['qtr'].mean()):.0f} names per month), "
              f"yet quarterly-YoY still underperforms in 2007-2012 — the effect "
              f"is regime-dependent signal quality, not a thinner universe.")


if __name__ == "__main__":
    main()
