"""Guards against look-ahead in the backtests' fundamentals lookups.

The bug this exists to catch: backtest_composite.run_backtest ranked each
rebalance month with cached_fscore(ticker, today.year), which resolves to
{year}-06-30 REGARDLESS of the month being simulated. A January rebalance was
therefore scored on annual reports filed up to five months in its own future.
backtest_sector_cap.run_backtest_cap had its own copy of the same pattern.

Three layers of defence:
  1. WIRING  — the ranking loop must pass a DATE to its F-score function, and
     that date must be the rebalance date. Uses a synthetic universe, so it
     runs with no repo data and no network.
  2. SEMANTICS — a January as-of must not see a filing that lands after it, and
     where January and June genuinely disagree the backtest must take January.
     Reads the local PIT fundamentals cache; no network.
  3. EQUIVALENCE — the memoized as-of timelines must return exactly what the
     direct pit_* calls return, or the speed-up would be a silent rewrite of
     the point-in-time rules.

Run:  python test_backtest_lookahead.py
"""
import sys

import numpy as np
import pandas as pd

import backtest_composite
import backtest_sector_cap
import pit_factors
from backtest_composite import asof_fscore, cached_fscore, run_backtest
from pit_factors import (_annual_periods, pit_fscore, pit_fscore_asof,
                         pit_fscore_quarterly_yoy,
                         pit_fscore_quarterly_yoy_asof, pit_value_inputs,
                         pit_value_inputs_asof)

fails = []


def check(name, cond):
    print(("PASS " if cond else "FAIL ") + name)
    if not cond:
        fails.append(name)


# --------------------------------------------------------------------------- #
# 1. WIRING — synthetic universe, no repo data, no network
# --------------------------------------------------------------------------- #
TICKERS = [f"T{i:02d}" for i in range(40)]
DATES = pd.date_range("2018-01-31", "2021-12-31", freq="ME")

rng = np.random.default_rng(0)
_prices = pd.DataFrame(
    100 * np.cumprod(1 + rng.normal(0.01, 0.05, size=(len(DATES), len(TICKERS))),
                     axis=0),
    index=DATES, columns=TICKERS)
_membership = pd.DataFrame({"ticker": TICKERS,
                            "start": pd.Timestamp("2000-01-01"),
                            "end": pd.NaT})
_trend = pd.DataFrame(0.0, index=DATES, columns=TICKERS)

seen = []


def spy_fscore(ticker, as_of):
    """Stand-in quality factor that records what the loop asked it for."""
    seen.append((ticker, as_of))
    return 5.0


run_backtest(_prices, _membership, _trend, top_n=10,
             start="2018-01-01", end="2021-12-31",
             w_mom=0.5, w_qual=0.5, fscore_fn=spy_fscore)

check("wiring: the ranking loop actually called the F-score function",
      len(seen) > 0)

check("wiring: F-score is asked for by DATE, not by year "
      "(an int year here is the bug itself)",
      all(isinstance(as_of, pd.Timestamp) for _, as_of in seen))

asked = sorted({as_of for _, as_of in seen})
check("wiring: every as-of is a rebalance date in the simulated window",
      all(d in set(DATES) for d in asked))

jan_asks = [d for d in asked if d.month == 1]
check("wiring: January rebalances were exercised", len(jan_asks) > 0)

check("wiring: a January rebalance asks as-of JANUARY, never June of the same "
      "year (the exact substitution the bug made)",
      all(d.month == 1 for d in jan_asks)
      and not any(d.month == 6 and d.year in {j.year for j in jan_asks}
                  and d not in set(DATES[DATES.month == 6]) for d in asked))

# The month-end grid should be hit once per eligible ticker per month, so the
# as-of set must be as wide as the trading window, not one date per year.
check("wiring: as-of dates span months, not years "
      "(>12 distinct as-of dates over a 4-year window)",
      len(asked) > 12)


# --------------------------------------------------------------------------- #
# 2. SEMANTICS — real point-in-time data from the local cache
# --------------------------------------------------------------------------- #
SAMPLE = ["AAPL", "JPM", "MSFT", "WMT", "XOM", "PG", "JNJ", "KO", "CAT", "HD",
          "MRK", "PFE", "T", "CVX", "UNH", "COST", "ORCL", "NKE", "IBM", "GE"]
YEARS = range(2010, 2025)

# Where does the January view genuinely differ from the June view? Those are
# the cases the bug silently got wrong; if none exist the test proves nothing.
divergent = []
for t in SAMPLE:
    for y in YEARS:
        jan = pit_fscore_asof(t, f"{y}-01-31")
        jun = pit_fscore_asof(t, f"{y}-06-30")
        if jan != jun:
            divergent.append((t, y, jan, jun))

check(f"semantics: found January/June disagreements to test against "
      f"({len(divergent)} ticker-years)", len(divergent) > 0)

check("semantics: at a January rebalance the backtest takes the JANUARY "
      "F-score, not June's",
      all(asof_fscore(t, pd.Timestamp(f"{y}-01-31")) == jan
          for t, y, jan, _ in divergent))

check("semantics: the old year-keyed helper returns June's value — kept for "
      "the live scan, and this is why it must not rank a simulated date",
      all(cached_fscore(t, y) == jun for t, y, _, jun in divergent))

check("semantics: fixed and buggy paths genuinely disagree on those dates "
      "(so the fix is doing something)",
      any(asof_fscore(t, pd.Timestamp(f"{y}-01-31")) != cached_fscore(t, y)
          for t, y, _, _ in divergent))


def periods_behind_score(ticker, as_of):
    """The fiscal years the annual F-score is allowed to draw on at as_of."""
    ctx = _annual_periods(ticker)
    if ctx is None:
        return []
    as_of = pd.Timestamp(as_of)
    return [(fy, avail) for fy, avail in ctx[3] if avail <= as_of]


unfiled = []
for t in SAMPLE:
    for y in YEARS:
        as_of = pd.Timestamp(f"{y}-01-31")
        if pit_fscore_asof(t, as_of) is None:
            continue
        for fy, avail in periods_behind_score(t, as_of):
            if avail > as_of:
                unfiled.append((t, y, fy, avail))

check("semantics: no January score draws on a filing dated after that January",
      not unfiled)

# A June cutoff would routinely admit a fiscal year that was not yet filed in
# January. Confirm that is true in this sample, so test above is not vacuous.
june_only = 0
for t in SAMPLE:
    for y in YEARS:
        jan_p = {fy for fy, _ in periods_behind_score(t, f"{y}-01-31")}
        jun_p = {fy for fy, _ in periods_behind_score(t, f"{y}-06-30")}
        june_only += len(jun_p - jan_p)

check(f"semantics: a June cutoff really does leak unfiled years into January "
      f"({june_only} ticker-year periods)", june_only > 0)


# --------------------------------------------------------------------------- #
# 3. EQUIVALENCE — timelines must not rewrite the point-in-time rules
# --------------------------------------------------------------------------- #
PROBE_DATES = [pd.Timestamp("2009-03-31") + pd.offsets.MonthEnd(i)
               for i in range(0, 200, 7)]

mismatch_a = [(t, d) for t in SAMPLE for d in PROBE_DATES
              if pit_fscore(t, d) != pit_fscore_asof(t, d)]
check("equivalence: pit_fscore_asof == pit_fscore on every probe",
      not mismatch_a)

mismatch_q = [(t, d) for t in SAMPLE for d in PROBE_DATES
              if pit_fscore_quarterly_yoy(t, d)
              != pit_fscore_quarterly_yoy_asof(t, d)]
check("equivalence: pit_fscore_quarterly_yoy_asof == pit_fscore_quarterly_yoy",
      not mismatch_q)

mismatch_v = [(t, d) for t in SAMPLE for d in PROBE_DATES
              if pit_value_inputs(t, d) != pit_value_inputs_asof(t, d)]
check("equivalence: pit_value_inputs_asof == pit_value_inputs", not mismatch_v)


# --------------------------------------------------------------------------- #
# 4. NO SURVIVING COPIES of the pattern
# --------------------------------------------------------------------------- #
check("no copies: backtest_composite.run_backtest defaults to the as-of F-score",
      run_backtest.__defaults__ is not None
      and asof_fscore in run_backtest.__defaults__)

src_composite = open(backtest_composite.__file__, encoding="utf-8").read()
src_sectorcap = open(backtest_sector_cap.__file__, encoding="utf-8").read()

check("no copies: backtest_composite's ranking loop has no `year = today.year`",
      "year = today.year" not in src_composite)
check("no copies: backtest_sector_cap's ranking loop has no `year = today.year`",
      "year = today.year" not in src_sectorcap)
check("no copies: backtest_sector_cap ranks with asof_fscore",
      "asof_fscore(ticker, today)" in src_sectorcap)


print()
if fails:
    print(f"{len(fails)} FAILED: " + "; ".join(fails))
    sys.exit(1)
print("All look-ahead guards passed.")
