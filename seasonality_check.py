"""
seasonality_check.py — DIAGNOSTIC ONLY (modifies no factor code).

Question: does pit_fscore_quarterly carry a systematic seasonal bias? Because it
compares a quarter against the IMMEDIATELY PRIOR quarter (not year-over-year),
a business with a seasonally strong quarter can score high on the flow-based
signals (ROA up, margin up, asset turnover up, CFO>0 ...) purely because that
quarter is seasonally bigger than the one before it — not because quality
actually improved.

Method:
  1. Broad sector-diverse sample of current S&P 500 members (cached data only).
  2. For each ticker, take its 4 most recent fully-available quarters. Score
     pit_fscore_quarterly at as_of = (that quarter's filing_date + 2 days), so
     the "current" quarter cycles through the ticker's own fiscal Q1..Q4.
  3. Assign each current quarter a FISCAL position (Q1-Q4) using the company's
     own FiscalYearEnd month — not calendar quarters.
  4. Group scores by fiscal position; report mean + distribution per group.
  5. Same tickers/dates scored with the ANNUAL F-score as a baseline (annual is
     already YoY, so it should show no seasonal pattern).

Run: python seasonality_check.py
"""

import os
import json
from collections import defaultdict

import numpy as np

from explore_constituents import get_sp500_membership
from pit_factors import pit_fscore_quarterly, pit_fscore_quarterly_yoy, pit_fscore

FUND_DIR = "data/pit_fundamentals"
PER_SECTOR_CAP = 9          # keep the sample sector-balanced
QUARTERS_PER_TICKER = 4     # one full fiscal-year cycle -> one obs per position

MONTHS = {
    "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
    "july": 7, "august": 8, "september": 9, "october": 10, "november": 11,
    "december": 12,
}
POS_BY_INDEX = ["Q4", "Q1", "Q2", "Q3"]   # index = round((month-FYE)/3) % 4


def _load(ticker):
    try:
        with open(f"{FUND_DIR}/{ticker}.json") as fh:
            return json.load(fh)
    except Exception:
        return None


def _fiscal_position(period_end, fye_month):
    """Fiscal quarter label (Q1-Q4) for a period-end date, given the company's
    fiscal-year-end month. Q4 is the quarter that ends the fiscal year."""
    month = int(period_end[5:7])
    offset = (month - fye_month) % 12
    return POS_BY_INDEX[int(round(offset / 3.0)) % 4]


def build_sample():
    """Sector-balanced sample of current members with cached fundamentals."""
    m = get_sp500_membership()
    current = sorted(m[m["end"].isna()]["ticker"].dropna().unique())

    by_sector = defaultdict(list)
    for t in current:
        d = _load(t)
        if not d:
            continue
        gen = d.get("General", {}) or {}
        sector = gen.get("Sector") or "Unknown"
        fye = MONTHS.get(str(gen.get("FiscalYearEnd", "")).strip().lower())
        if fye is None:
            continue
        by_sector[sector].append((t, fye))

    sample = []
    for sector, rows in sorted(by_sector.items()):
        sample.extend((t, fye, sector) for t, fye in rows[:PER_SECTOR_CAP])
    return sample


def recent_quarter_asofs(ticker, n=QUARTERS_PER_TICKER):
    """The n most recent quarters that carry a filing_date, as
    (period_end, as_of=filing_date+2d) pairs, newest first."""
    d = _load(ticker)
    if not d:
        return []
    q = (d.get("Financials", {}).get("Income_Statement", {}) or {}).get("quarterly", {})
    out = []
    for period in sorted(q.keys(), reverse=True):
        fd = (q[period] or {}).get("filing_date")
        if fd:
            as_of = (np.datetime64(fd) + np.timedelta64(2, "D"))
            out.append((period, str(as_of)))
        if len(out) >= n:
            break
    return out


def summarize(scores):
    arr = np.array(scores, dtype=float)
    if len(arr) == 0:
        return dict(n=0, mean=float("nan"), std=float("nan"),
                    mn=float("nan"), mx=float("nan"))
    return dict(n=len(arr), mean=arr.mean(), std=arr.std(),
                mn=arr.min(), mx=arr.max())


def main():
    sample = build_sample()
    sectors = sorted(set(s for _, _, s in sample))
    print(f"Sample: {len(sample)} tickers across {len(sectors)} sectors "
          f"(cap {PER_SECTOR_CAP}/sector), {QUARTERS_PER_TICKER} quarters each.\n")

    q_by_pos = defaultdict(list)     # sequential quarterly F-scores by position
    y_by_pos = defaultdict(list)     # YoY quarterly F-scores by position
    a_by_pos = defaultdict(list)     # annual F-scores by same position label

    for ticker, fye, _sector in sample:
        for period_end, as_of in recent_quarter_asofs(ticker):
            pos = _fiscal_position(period_end, fye)
            fq = pit_fscore_quarterly(ticker, as_of)
            fy = pit_fscore_quarterly_yoy(ticker, as_of)
            fa = pit_fscore(ticker, as_of)
            if fq is not None:
                q_by_pos[pos].append(fq)
            if fy is not None:
                y_by_pos[pos].append(fy)
            if fa is not None:
                a_by_pos[pos].append(fa)

    n_seq = sum(len(v) for v in q_by_pos.values())
    n_yoy = sum(len(v) for v in y_by_pos.values())
    print(f"Scored {n_seq} sequential and {n_yoy} YoY quarterly observations.\n")

    order = ["Q1", "Q2", "Q3", "Q4"]
    print("=" * 88)
    print("AVERAGE F-SCORE BY FISCAL-QUARTER POSITION (current quarter)")
    print("=" * 88)
    print(f"{'pos':<4} | {'SEQUENTIAL (Q vs prior Q)':^26} | "
          f"{'YoY (Q vs same Q -1yr)':^26} | {'ANNUAL (baseline)':^20}")
    print(f"{'':<4} | {'n':>3} {'mean':>5} {'std':>5} {'min':>3} {'max':>3}  | "
          f"{'n':>3} {'mean':>5} {'std':>5} {'min':>3} {'max':>3}  | "
          f"{'n':>3} {'mean':>5} {'std':>5}")
    print("-" * 88)
    q_means, y_means, a_means = [], [], []
    for pos in order:
        q, y, a = (summarize(q_by_pos[pos]), summarize(y_by_pos[pos]),
                   summarize(a_by_pos[pos]))
        q_means.append(q["mean"]); y_means.append(y["mean"]); a_means.append(a["mean"])
        print(f"{pos:<4} | {q['n']:>3} {q['mean']:>5.2f} {q['std']:>5.2f} "
              f"{q['mn']:>3.0f} {q['mx']:>3.0f}  | "
              f"{y['n']:>3} {y['mean']:>5.2f} {y['std']:>5.2f} "
              f"{y['mn']:>3.0f} {y['mx']:>3.0f}  | "
              f"{a['n']:>3} {a['mean']:>5.2f} {a['std']:>5.2f}")

    q_all = summarize([s for v in q_by_pos.values() for s in v])
    y_all = summarize([s for v in y_by_pos.values() for s in v])
    a_all = summarize([s for v in a_by_pos.values() for s in v])
    print("-" * 88)
    print(f"{'ALL':<4} | {q_all['n']:>3} {q_all['mean']:>5.2f} {q_all['std']:>5.2f} "
          f"{q_all['mn']:>3.0f} {q_all['mx']:>3.0f}  | "
          f"{y_all['n']:>3} {y_all['mean']:>5.2f} {y_all['std']:>5.2f} "
          f"{y_all['mn']:>3.0f} {y_all['mx']:>3.0f}  | "
          f"{a_all['n']:>3} {a_all['mean']:>5.2f} {a_all['std']:>5.2f}")

    # ---- Verdict, with a permutation test so we don't mistake noise for skew ----
    def perm_test(by_pos, n_perm=5000, seed=0):
        """Observed group-mean range vs its null distribution under randomly
        shuffled fiscal-position labels. Returns (obs_range, noise_floor, p)."""
        rng = np.random.default_rng(seed)
        vals, labs = [], []
        for pos in order:
            for v in by_pos[pos]:
                vals.append(v); labs.append(pos)
        vals = np.array(vals, float); labs = np.array(labs)

        def rng_of(labels):
            means = [vals[labels == pos].mean() for pos in order]
            return max(means) - min(means)

        obs = rng_of(labs)
        null = np.array([rng_of(rng.permutation(labs)) for _ in range(n_perm)])
        p = (np.sum(null >= obs) + 1) / (n_perm + 1)
        return obs, null.mean(), p

    q_obs, q_floor, q_p = perm_test(q_by_pos)
    y_obs, y_floor, y_p = perm_test(y_by_pos)
    a_obs, a_floor, a_p = perm_test(a_by_pos)

    print("\n" + "=" * 88)
    print("VERDICT — did YoY remove the seasonal skew?")
    print("=" * 88)
    print("Group-mean spread across fiscal-quarter position, tested against the")
    print("noise floor (expected spread if position had NO effect, via 5000-shuffle")
    print("permutation test). p = chance of seeing this spread if there's no skew.\n")
    print(f"{'variant':<12} {'obs range':>10} {'noise floor':>12} {'p-value':>9}   verdict")
    print("-" * 88)
    for name, obs, floor, p in [
        ("SEQUENTIAL", q_obs, q_floor, q_p),
        ("YoY",        y_obs, y_floor, y_p),
        ("ANNUAL",     a_obs, a_floor, a_p),
    ]:
        tag = "SKEW (p<0.05)" if p < 0.05 else "within noise"
        print(f"{name:<12} {obs:>10.2f} {floor:>12.2f} {p:>9.3f}   {tag}")

    reduction = (1 - y_obs / q_obs) * 100 if q_obs else 0.0
    print("\n" + "-" * 88)
    if q_p < 0.05 and y_p >= 0.05:
        print(f"==> YoY FIX CONFIRMED. The sequential comparison shows a real "
              f"seasonal skew (range {q_obs:.2f}, p={q_p:.3f} — beyond the "
              f"{q_floor:.2f} noise floor). Switching to YoY cuts the spread to "
              f"{y_obs:.2f} ({reduction:.0f}% lower) and it is no longer "
              f"distinguishable from noise (p={y_p:.3f}), matching the annual "
              f"baseline (p={a_p:.3f}). YoY is the variant safe to rank on.")
    elif q_p < 0.05 and y_p < 0.05:
        print(f"==> PARTIAL. YoY reduces the spread {q_obs:.2f} -> {y_obs:.2f} "
              f"({reduction:.0f}%) but a residual skew is still significant "
              f"(p={y_p:.3f}). Investigate before wiring downstream.")
    else:
        print(f"==> INCONCLUSIVE on sequential (p={q_p:.3f}); re-check the sample.")


if __name__ == "__main__":
    main()
