import bisect

import pandas as pd
from point_in_time import (
    _fetch_raw, _build_availability_map, _quarter_availability,
)

# YoY prior-quarter window: the same fiscal quarter one year earlier sits ~4
# quarters (~364 days) back. This window isolates that quarter while excluding
# the 3-back (~273d) and 5-back (~455d) neighbours, with slack for 52/53-week
# fiscal calendars and filing drift.
YOY_MIN_DAYS = 310
YOY_MAX_DAYS = 420


_ANNUAL_MEM = {}


def _extract_annual(income, balance, cashflow, fy_end):
    """Pull the F-score input fields for one fiscal year into the shared dict
    shape. Was a closure inside pit_two_years; lifted out unchanged so the
    as-of timeline below can reuse it instead of restating the field mapping."""
    key_year = str(fy_end.year)

    def find(stmt):
        for k in stmt:
            if k.startswith(key_year):
                return stmt[k]
        return {}

    inc, bal, cf = find(income), find(balance), find(cashflow)

    def f(v):
        try:
            return float(v) if v is not None else None
        except (ValueError, TypeError):
            return None

    return {
        "Revenue": f(inc.get("totalRevenue")),
        "NetIncome": f(inc.get("netIncome")),
        "GrossProfit": f(inc.get("grossProfit")),
        "CostOfRevenue": f(inc.get("costOfRevenue")),
        "TotalAssets": f(bal.get("totalAssets")),
        "CurrentAssets": f(bal.get("totalCurrentAssets")),
        "CurrentLiabilities": f(bal.get("totalCurrentLiabilities")),
        "LongTermDebt": f(bal.get("longTermDebt")),
        "SharesOutstanding": f(bal.get("commonStockSharesOutstanding")),
        "OperatingCashFlow": f(cf.get("totalCashFromOperatingActivities")),
    }


def _annual_periods(ticker):
    """Shared ANNUAL machinery, parsed once per ticker: returns
    (income, balance, cashflow, [(fiscal_year_end, date_available), ...]) sorted
    by fiscal year-end, or None if unusable.

    `date_available` is the same rule the annual path has always applied: the
    earliest reportDate on/after the fiscal year-end, else fiscal_end + 90 days.
    Splitting it out (from pit_two_years) makes availability a per-ticker fact
    rather than something re-derived on every as-of lookup."""
    if ticker in _ANNUAL_MEM:
        return _ANNUAL_MEM[ticker]

    ctx = None
    data = _fetch_raw(ticker)
    if data is not None:
        fin = data.get("Financials", {})
        income = fin.get("Income_Statement", {}).get("yearly", {})
        balance = fin.get("Balance_Sheet", {}).get("yearly", {})
        cashflow = fin.get("Cash_Flow", {}).get("yearly", {})
        if income:
            report_dates = _build_availability_map(data, ticker)
            periods = []
            for date_str in income.keys():
                fy_end = pd.to_datetime(date_str)
                later = [rd for pe, rd in report_dates.items()
                         if pe >= fy_end and rd >= fy_end]
                avail = min(later) if later else fy_end + pd.Timedelta(days=90)
                periods.append((fy_end, avail))
            periods.sort()
            ctx = (income, balance, cashflow, periods)

    _ANNUAL_MEM[ticker] = ctx
    return ctx


def pit_two_years(ticker, as_of_date):
    """
    Return (current, prior) annual fundamentals dicts as known on
    as_of_date — the most recent AVAILABLE year and the one before it.
    Both must have been filed by as_of_date. Returns (None, None) if
    fewer than two years are available.
    """
    as_of = pd.Timestamp(as_of_date)
    ctx = _annual_periods(ticker)
    if ctx is None:
        return None, None
    income, balance, cashflow, periods = ctx

    available_years = [fy_end for fy_end, avail in periods if avail <= as_of]
    if len(available_years) < 2:
        return None, None

    available_years.sort()
    curr_end = available_years[-1]
    prev_end = available_years[-2]

    return (_extract_annual(income, balance, cashflow, curr_end),
            _extract_annual(income, balance, cashflow, prev_end))


def _score_pair(curr, prev):
    """The 9 Piotroski signals over a (current, prior) pair of period dicts.
    Returns (signals_scored, signals_available). Shared by the annual and
    quarterly F-scores so both use the exact same signal logic. Each dict must
    expose the keys produced by pit_two_years / pit_two_quarters. Missing
    inputs (JPM-style gaps) simply make a signal unavailable rather than crash.
    """
    signals = 0
    available = 0

    def roa(d):
        if d["NetIncome"] is not None and d["TotalAssets"]:
            return d["NetIncome"] / d["TotalAssets"]
        return None

    roa_c, roa_p = roa(curr), roa(prev)

    if roa_c is not None:
        available += 1
        signals += 1 if roa_c > 0 else 0
    if curr["OperatingCashFlow"] is not None:
        available += 1
        signals += 1 if curr["OperatingCashFlow"] > 0 else 0
    if roa_c is not None and roa_p is not None:
        available += 1
        signals += 1 if roa_c > roa_p else 0
    if curr["OperatingCashFlow"] is not None and curr["TotalAssets"] and roa_c is not None:
        available += 1
        signals += 1 if (curr["OperatingCashFlow"] / curr["TotalAssets"]) > roa_c else 0
    if (curr["LongTermDebt"] is not None and curr["TotalAssets"]
            and prev["LongTermDebt"] is not None and prev["TotalAssets"]):
        available += 1
        signals += 1 if (curr["LongTermDebt"] / curr["TotalAssets"]) < (prev["LongTermDebt"] / prev["TotalAssets"]) else 0
    if (curr["CurrentAssets"] and curr["CurrentLiabilities"]
            and prev["CurrentAssets"] and prev["CurrentLiabilities"]):
        available += 1
        signals += 1 if (curr["CurrentAssets"] / curr["CurrentLiabilities"]) > (prev["CurrentAssets"] / prev["CurrentLiabilities"]) else 0
    if curr["SharesOutstanding"] is not None and prev["SharesOutstanding"] is not None:
        available += 1
        signals += 1 if curr["SharesOutstanding"] <= prev["SharesOutstanding"] else 0

    def gm(d):
        if d["GrossProfit"] is not None and d["Revenue"]:
            return d["GrossProfit"] / d["Revenue"]
        if d["Revenue"] and d["CostOfRevenue"] is not None:
            return (d["Revenue"] - d["CostOfRevenue"]) / d["Revenue"]
        return None

    gm_c, gm_p = gm(curr), gm(prev)
    if gm_c is not None and gm_p is not None:
        available += 1
        signals += 1 if gm_c > gm_p else 0
    if curr["Revenue"] and curr["TotalAssets"] and prev["Revenue"] and prev["TotalAssets"]:
        available += 1
        signals += 1 if (curr["Revenue"] / curr["TotalAssets"]) > (prev["Revenue"] / prev["TotalAssets"]) else 0

    return signals, available


def pit_fscore(ticker, as_of_date):
    """Piotroski F-score from point-in-time ANNUAL data. Returns 0-9 or None."""
    curr, prev = pit_two_years(ticker, as_of_date)
    if curr is None or prev is None:
        return None
    signals, available = _score_pair(curr, prev)
    if available < 5:
        return None
    return signals


_QUARTERLY_MEM = {}


def _quarterly_periods(ticker):
    """Shared QUARTERLY machinery, parsed once per ticker: returns
    (income, balance, cashflow, [(quarter_end, date_available), ...]) sorted by
    quarter-end, or None if unusable. Availability uses each quarter's own
    embedded filing_date (fallback quarter_end + QUARTERLY_LAG_DAYS)."""
    if ticker in _QUARTERLY_MEM:
        return _QUARTERLY_MEM[ticker]

    ctx = None
    data = _fetch_raw(ticker)
    if data is not None:
        fin = data.get("Financials", {})
        income = fin.get("Income_Statement", {}).get("quarterly", {})
        balance = fin.get("Balance_Sheet", {}).get("quarterly", {})
        cashflow = fin.get("Cash_Flow", {}).get("quarterly", {})
        if income:
            periods = []
            for date_str, inc_row in income.items():
                q_end = pd.to_datetime(date_str)
                avail = _quarter_availability(date_str, inc_row,
                                              balance.get(date_str),
                                              cashflow.get(date_str))
                periods.append((q_end, avail))
            periods.sort()
            ctx = (income, balance, cashflow, periods)

    _QUARTERLY_MEM[ticker] = ctx
    return ctx


def _quarterly_context(ticker, as_of_date):
    """Shared quarterly machinery: returns (income, balance, cashflow,
    available_quarters_sorted) where availability uses each quarter's own
    filing_date (fallback quarter_end + QUARTERLY_LAG_DAYS) and only quarters
    filed on/before as_of_date are included. Returns None if unusable."""
    as_of = pd.Timestamp(as_of_date)
    ctx = _quarterly_periods(ticker)
    if ctx is None:
        return None
    income, balance, cashflow, periods = ctx

    available = [q_end for q_end, avail in periods if avail <= as_of]
    available.sort()
    return income, balance, cashflow, available


def _extract_quarter(income, balance, cashflow, q_end):
    """Pull the F-score input fields for one quarter-end into the same dict
    shape pit_two_years produces. Missing statements/fields -> None (no crash)."""
    key = q_end.strftime("%Y-%m-%d")
    inc = income.get(key, {}) or {}
    bal = balance.get(key, {}) or {}
    cf = cashflow.get(key, {}) or {}

    def f(v):
        try:
            return float(v) if v is not None else None
        except (ValueError, TypeError):
            return None

    return {
        "Revenue": f(inc.get("totalRevenue")),
        "NetIncome": f(inc.get("netIncome")),
        "GrossProfit": f(inc.get("grossProfit")),
        "CostOfRevenue": f(inc.get("costOfRevenue")),
        "TotalAssets": f(bal.get("totalAssets")),
        "CurrentAssets": f(bal.get("totalCurrentAssets")),
        "CurrentLiabilities": f(bal.get("totalCurrentLiabilities")),
        "LongTermDebt": f(bal.get("longTermDebt")),
        "SharesOutstanding": f(bal.get("commonStockSharesOutstanding")),
        "OperatingCashFlow": f(cf.get("totalCashFromOperatingActivities")),
    }


def pit_two_quarters(ticker, as_of_date):
    """
    Quarterly analog of pit_two_years: return (current_quarter, prior_quarter)
    fundamentals dicts as known on as_of_date — the most recent AVAILABLE
    quarter and the one immediately before it. Availability uses each quarter's
    own embedded filing_date (fallback: quarter_end + QUARTERLY_LAG_DAYS). Both
    quarters must have been filed by as_of_date. Returns (None, None) if fewer
    than two quarters are available.

    NOTE: this compares CONSECUTIVE quarters (Q vs the immediately prior
    quarter), which carries a confirmed seasonal bias for the flow-based
    signals. Prefer pit_two_quarters_yoy for ranking. Kept for reference.
    """
    ctx = _quarterly_context(ticker, as_of_date)
    if ctx is None:
        return None, None
    income, balance, cashflow, available = ctx
    if len(available) < 2:
        return None, None

    curr_end = available[-1]
    prev_end = available[-2]
    return (_extract_quarter(income, balance, cashflow, curr_end),
            _extract_quarter(income, balance, cashflow, prev_end))


def pit_two_quarters_yoy(ticker, as_of_date):
    """
    Year-over-year quarterly pair: return (current_quarter, prior_quarter) where
    the prior quarter is the SAME fiscal quarter one year earlier (~4 quarters
    back), not the immediately preceding quarter. This removes the seasonal
    bias the diagnostic found in the consecutive-quarter comparison.

    Both quarters must have been filed by as_of_date. If the exact year-earlier
    quarter isn't available (younger ticker, gap in the data), returns
    (None, None) rather than substituting a wrong-season quarter.
    """
    ctx = _quarterly_context(ticker, as_of_date)
    if ctx is None:
        return None, None
    income, balance, cashflow, available = ctx
    if len(available) < 2:
        return None, None

    curr_end = available[-1]

    # Pick the available quarter closest to one year before curr_end, within the
    # YoY window so we never fall back to an adjacent (wrong-season) quarter.
    best, best_gap = None, None
    for q in available[:-1]:
        days = (curr_end - q).days
        if YOY_MIN_DAYS <= days <= YOY_MAX_DAYS:
            gap = abs(days - 365)
            if best is None or gap < best_gap:
                best, best_gap = q, gap
    if best is None:
        return None, None

    return (_extract_quarter(income, balance, cashflow, curr_end),
            _extract_quarter(income, balance, cashflow, best))


def pit_fscore_quarterly(ticker, as_of_date):
    """Piotroski F-score from point-in-time QUARTERLY data (consecutive quarters).
    Returns 0-9 or None. Same 9 signals and same '>=5 of 9 available' rule as
    the annual F-score, reusing _score_pair.

    PREFER pit_fscore_quarterly_yoy for any ranking/backtest use: the
    consecutive-quarter comparison has a confirmed systematic seasonal bias
    (~0.58 F-point spread across fiscal-quarter position vs the annual
    baseline's ~0.12). This function is retained for reference/comparison and
    is intentionally not wired downstream.
    """
    curr, prev = pit_two_quarters(ticker, as_of_date)
    if curr is None or prev is None:
        return None
    signals, available = _score_pair(curr, prev)
    if available < 5:
        return None
    return signals


def pit_fscore_quarterly_yoy(ticker, as_of_date):
    """Piotroski F-score from point-in-time QUARTERLY data compared YEAR-OVER-YEAR
    (Q vs the same quarter one year earlier). Returns 0-9 or None. Same 9 signals
    and same '>=5 of 9 available' rule, reusing _score_pair — this is the
    seasonally-unbiased quarterly F-score and the one intended for downstream use."""
    curr, prev = pit_two_quarters_yoy(ticker, as_of_date)
    if curr is None or prev is None:
        return None
    signals, available = _score_pair(curr, prev)
    if available < 5:
        return None
    return signals


def pit_value_inputs(ticker, as_of_date):
    """Point-in-time NetIncome + shares for earnings yield. None if unavailable."""
    curr, prev = pit_two_years(ticker, as_of_date)
    if curr is None:
        return None
    ni = curr["NetIncome"]
    sh = curr["SharesOutstanding"]
    if ni is None or sh is None or sh <= 0:
        return None
    return {"net_income": ni, "shares": sh}


# --------------------------------------------------------------------------- #
# AS-OF TIMELINES — the look-ahead-safe lookup a backtest should use
#
# A backtest rebalancing on date D may only see fundamentals FILED on or before
# D. Calling pit_fscore(ticker, D) once per rebalance month is the honest thing
# to do but costs ~80ms a call, which is why the backtests originally reached
# for a single fixed as-of per year instead — and that is exactly how the
# look-ahead crept in.
#
# These lookups are the same answer, made affordable. Every pit_* function
# above depends on as_of ONLY through WHICH periods are available at as_of, so
# the value is a step function that can change only on a filing-availability
# date. Evaluate once per such date, then bisect. Exactness (timeline lookup ==
# direct pit_* call, for every ticker and date) is asserted in
# test_backtest_lookahead.py.
# --------------------------------------------------------------------------- #

_FSCORE_TL = {}
_FSCORE_Q_TL = {}
_VALUE_TL = {}


def _build_timeline(periods, score_at):
    """[(effective_from, value), ...] — one evaluation per distinct availability
    date, held flat until the next filing lands."""
    dates, values = [], []
    for avail in sorted({avail for _, avail in periods}):
        dates.append(avail)
        values.append(score_at(avail))
    return dates, values


def _lookup_asof(cache, ticker, as_of_date, periods_fn, score_at):
    tl = cache.get(ticker)
    if tl is None:
        ctx = periods_fn(ticker)
        tl = ([], []) if ctx is None else _build_timeline(ctx[3], score_at)
        cache[ticker] = tl
    dates, values = tl
    i = bisect.bisect_right(dates, pd.Timestamp(as_of_date)) - 1
    return values[i] if i >= 0 else None


def pit_fscore_asof(ticker, as_of_date):
    """Annual F-score as known on as_of_date. Identical to
    pit_fscore(ticker, as_of_date), memoized per ticker so a backtest can
    afford a true per-rebalance-date as-of instead of one fixed date per year."""
    return _lookup_asof(_FSCORE_TL, ticker, as_of_date, _annual_periods,
                        lambda d: pit_fscore(ticker, d))


def pit_fscore_quarterly_yoy_asof(ticker, as_of_date):
    """Quarterly-YoY F-score as known on as_of_date. Identical to
    pit_fscore_quarterly_yoy(ticker, as_of_date), memoized per ticker."""
    return _lookup_asof(_FSCORE_Q_TL, ticker, as_of_date, _quarterly_periods,
                        lambda d: pit_fscore_quarterly_yoy(ticker, d))


def pit_value_inputs_asof(ticker, as_of_date):
    """Value inputs as known on as_of_date. Identical to
    pit_value_inputs(ticker, as_of_date), memoized per ticker."""
    return _lookup_asof(_VALUE_TL, ticker, as_of_date, _annual_periods,
                        lambda d: pit_value_inputs(ticker, d))


def drop_ticker_cache(ticker):
    """Release one ticker's parsed payload and timelines. A full-universe
    precompute holds ~825 parsed EODHD payloads otherwise; the timelines are
    tiny, the raw JSON is not."""
    from point_in_time import _RAW_MEM, _AVAIL_MEM
    _RAW_MEM.pop(ticker, None)
    _AVAIL_MEM.pop(ticker, None)
    _ANNUAL_MEM.pop(ticker, None)
    _QUARTERLY_MEM.pop(ticker, None)


if __name__ == "__main__":
    for date in ["2018-06-30", "2020-06-30", "2022-06-30", "2024-06-30"]:
        print(f"AAPL F-score {date}: {pit_fscore('AAPL', date)}")
        print(f"AAPL value {date}: {pit_value_inputs('AAPL', date)}")