import requests
import pandas as pd
from datetime import timedelta
from config import EODHD_API_KEY
import os
import json

# Conservative fallback lag if a real reportDate is missing
ANNUAL_LAG_DAYS = 90
# Conservative fallback lag if a quarterly row's filing_date is missing
# (10-Q filing deadline is ~40-45 days after quarter end for large filers)
QUARTERLY_LAG_DAYS = 45


_RAW_MEM = {}

def _fetch_raw(ticker):
    """Pull the full EODHD fundamentals payload, cached in memory + on disk."""
    if ticker in _RAW_MEM:
        return _RAW_MEM[ticker]
    os.makedirs("data/pit_fundamentals", exist_ok=True)
    path = f"data/pit_fundamentals/{ticker}.json"

    if os.path.exists(path):
        try:
            with open(path, "r") as fh:
                data = json.load(fh)
                _RAW_MEM[ticker] = data
                return data
        except Exception:
            pass

    url = f"https://eodhd.com/api/fundamentals/{ticker}.US?api_token={EODHD_API_KEY}&fmt=json"
    try:
        resp = requests.get(url, timeout=30)
        if resp.status_code != 200:
            return None
        data = resp.json()
    except Exception:
        return None

    try:
        with open(path, "w") as fh:
            json.dump(data, fh)
    except Exception:
        pass
    _RAW_MEM[ticker] = data
    return data


_AVAIL_MEM = {}


def _build_availability_map(data, ticker=None):
    """
    Build {fiscal_year_end_date -> date_it_became_public} for annual data.
    Uses real reportDate from Earnings where available, else fiscal_end + 90d.

    The result depends only on `data`, which is fixed per ticker for the life of
    the process, so passing `ticker` memoizes it. That is a pure speed-up: the
    parse is ~300 pd.to_datetime calls and dominated every point-in-time lookup
    (~85% of runtime), which is what made a per-rebalance-date as-of lookup
    prohibitively slow. Callers that omit `ticker` get the original uncached
    behaviour.
    """
    if ticker is not None and ticker in _AVAIL_MEM:
        return _AVAIL_MEM[ticker]

    earnings = data.get("Earnings", {}).get("History", {})

    # Map each fiscal-period-end to its actual reportDate (8-K filing date)
    report_dates = {}
    for period, info in earnings.items():
        rd = info.get("reportDate")
        if rd:
            report_dates[pd.to_datetime(period)] = pd.to_datetime(rd)

    if ticker is not None:
        _AVAIL_MEM[ticker] = report_dates
    return report_dates


def point_in_time_fundamentals(ticker, as_of_date, _cache={}):
    """
    Return the most recent ANNUAL fundamentals that were actually
    FILED on or before as_of_date. Never returns data filed later.

    Implements the 4 rules:
      - annual report available only after its filing date
      - use real reportDate if present, else fiscal_end + 90 days
      - rank uses most recent AVAILABLE annual
      - nothing available yet -> returns None (stock ineligible)
    """
    as_of = pd.Timestamp(as_of_date)

    # Cache the raw payload per ticker (avoid re-pulling every month)
    if ticker not in _cache:
        _cache[ticker] = _fetch_raw(ticker)
    data = _cache[ticker]
    if data is None:
        return None

    report_dates = _build_availability_map(data, ticker)

    # Annual financial statements, keyed by fiscal-year-end date
    fin = data.get("Financials", {})
    income = fin.get("Income_Statement", {}).get("yearly", {})
    balance = fin.get("Balance_Sheet", {}).get("yearly", {})
    cashflow = fin.get("Cash_Flow", {}).get("yearly", {})
    if not income:
        return None

    # For each annual period, determine when it became public
    candidates = []
    for date_str in income.keys():
        fy_end = pd.to_datetime(date_str)

        # When did this annual data become available?
        # Prefer a real reportDate at/after fiscal year-end; else +90 days.
        avail = None
        # find the earliest reportDate on/after this fiscal year-end
        later_reports = [rd for pe, rd in report_dates.items()
                         if pe >= fy_end and rd >= fy_end]
        if later_reports:
            avail = min(later_reports)
        else:
            avail = fy_end + timedelta(days=ANNUAL_LAG_DAYS)

        if avail <= as_of:
            candidates.append((fy_end, avail))

    if not candidates:
        return None  # Rule 4: nothing filed yet -> ineligible

    # Most recent AVAILABLE annual period
    fy_end, avail = max(candidates, key=lambda x: x[0])
    key = fy_end.strftime("%Y-%m-%d")

    # Some EODHD keys are exact date strings; match by year if needed
    def _get(stmt, k):
        if k in stmt:
            return stmt[k]
        # fallback: match by year
        for kk in stmt:
            if kk.startswith(str(fy_end.year)):
                return stmt[kk]
        return {}

    inc = _get(income, key)
    bal = _get(balance, key)
    cf = _get(cashflow, key)

    def f(v):
        try:
            return float(v) if v is not None else None
        except (ValueError, TypeError):
            return None

    return {
        "ticker": ticker,
        "fiscal_year_end": key,
        "available_since": avail.strftime("%Y-%m-%d"),
        "Revenue": f(inc.get("totalRevenue")),
        "NetIncome": f(inc.get("netIncome")),
        "GrossProfit": f(inc.get("grossProfit")),
        "EBITDA": f(inc.get("ebitda")),
        "TotalAssets": f(bal.get("totalAssets")),
        "TotalLiabilities": f(bal.get("totalLiab")),
        "CurrentAssets": f(bal.get("totalCurrentAssets")),
        "CurrentLiabilities": f(bal.get("totalCurrentLiabilities")),
        "LongTermDebt": f(bal.get("longTermDebt")),
        "SharesOutstanding": f(bal.get("commonStockSharesOutstanding")),
        "OperatingCashFlow": f(cf.get("totalCashFromOperatingActivities")),
        "FreeCashFlow": f(cf.get("freeCashFlow")),
    }


def _quarter_availability(period_end, *rows):
    """When did a quarter become public? Use the embedded filing_date from the
    first statement row that carries one (the real 10-Q/10-K date); if none is
    present, fall back to period_end + QUARTERLY_LAG_DAYS. Returns a Timestamp."""
    for row in rows:
        fd = row.get("filing_date") if isinstance(row, dict) else None
        if fd:
            try:
                return pd.to_datetime(fd)
            except (ValueError, TypeError):
                pass
    return pd.to_datetime(period_end) + timedelta(days=QUARTERLY_LAG_DAYS)


def point_in_time_fundamentals_quarterly(ticker, as_of_date, _cache={}):
    """
    Return the most recent QUARTERLY fundamentals that were actually FILED on
    or before as_of_date. Never returns data filed later than as_of_date.

    This is the quarterly analog of point_in_time_fundamentals. Availability
    comes from each quarter's OWN filing_date field (the actual 10-Q/10-K date
    embedded on every Financials::*::quarterly row — confirmed by the inventory
    on 100% of AAPL/JPM quarters), so no join against Earnings::History is
    needed. Falls back to period_end + 45 days only if a filing_date is missing.

    The existing annual functions are unchanged; this is purely additive.
    """
    as_of = pd.Timestamp(as_of_date)

    # Cache the raw payload per ticker (avoid re-pulling every month)
    if ticker not in _cache:
        _cache[ticker] = _fetch_raw(ticker)
    data = _cache[ticker]
    if data is None:
        return None

    fin = data.get("Financials", {})
    income = fin.get("Income_Statement", {}).get("quarterly", {})
    balance = fin.get("Balance_Sheet", {}).get("quarterly", {})
    cashflow = fin.get("Cash_Flow", {}).get("quarterly", {})
    if not income:
        return None

    # Each quarter becomes available on its own filing_date.
    candidates = []
    for date_str, inc_row in income.items():
        q_end = pd.to_datetime(date_str)
        avail = _quarter_availability(date_str, inc_row,
                                      balance.get(date_str), cashflow.get(date_str))
        if avail <= as_of:                      # refuse anything filed later
            candidates.append((q_end, avail))

    if not candidates:
        return None  # nothing filed yet -> ineligible (same as annual rule 4)

    # Most recent AVAILABLE quarter
    q_end, avail = max(candidates, key=lambda x: x[0])
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
        "ticker": ticker,
        "fiscal_quarter_end": key,
        "available_since": avail.strftime("%Y-%m-%d"),
        "Revenue": f(inc.get("totalRevenue")),
        "NetIncome": f(inc.get("netIncome")),
        "GrossProfit": f(inc.get("grossProfit")),
        "EBITDA": f(inc.get("ebitda")),
        "TotalAssets": f(bal.get("totalAssets")),
        "TotalLiabilities": f(bal.get("totalLiab")),
        "CurrentAssets": f(bal.get("totalCurrentAssets")),
        "CurrentLiabilities": f(bal.get("totalCurrentLiabilities")),
        "LongTermDebt": f(bal.get("longTermDebt")),
        "SharesOutstanding": f(bal.get("commonStockSharesOutstanding")),
        "OperatingCashFlow": f(cf.get("totalCashFromOperatingActivities")),
        "FreeCashFlow": f(cf.get("freeCashFlow")),
    }


if __name__ == "__main__":
    # THE CRITICAL TEST: prove it refuses future data.
    print("Test 1 — CEG as of 2020-01-15:")
    r = point_in_time_fundamentals("CEG", "2020-01-15")
    if r:
        print(f"  Used fiscal year: {r['fiscal_year_end']} (available since {r['available_since']})")
        print(f"  Revenue: {r['Revenue']:,.0f}" if r['Revenue'] else "  Revenue: n/a")
    else:
        print("  None — no annual data filed by this date")

    print("\nTest 2 — CEG as of 2020-06-01:")
    r = point_in_time_fundamentals("CEG", "2020-06-01")
    if r:
        print(f"  Used fiscal year: {r['fiscal_year_end']} (available since {r['available_since']})")
        print(f"  Revenue: {r['Revenue']:,.0f}" if r['Revenue'] else "  Revenue: n/a")
    else:
        print("  None — no annual data filed by this date")

    print("\nTest 3 — CEG as of 2026-06-30 (most recent available):")
    r = point_in_time_fundamentals("CEG", "2026-06-30")
    if r:
        print(f"  Used fiscal year: {r['fiscal_year_end']} (available since {r['available_since']})")

    print("\nDiagnostic — CEG's annual periods and when each became available:")
    data = _fetch_raw("CEG")
    report_dates = _build_availability_map(data)
    income = data["Financials"]["Income_Statement"]["yearly"]
    for date_str in sorted(income.keys()):
        fy_end = pd.to_datetime(date_str)
        later = [rd for pe, rd in report_dates.items() if pe >= fy_end and rd >= fy_end]
        avail = min(later) if later else fy_end + pd.Timedelta(days=90)
        print(f"  FY {date_str[:10]} -> available {avail.date()}")

    print("\nTest 1b — AAPL as of 2020-01-15 (real look-ahead test):")
    r = point_in_time_fundamentals("AAPL", "2020-01-15")
    if r:
        print(f"  Used fiscal year: {r['fiscal_year_end']} (available since {r['available_since']})")
        print(f"  Revenue: {r['Revenue']:,.0f}" if r['Revenue'] else "  Revenue: n/a")
    else:
        print("  None")