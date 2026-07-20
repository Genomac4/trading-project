import pandas as pd
from point_in_time import (
    _fetch_raw, _build_availability_map, _quarter_availability,
)


def pit_two_years(ticker, as_of_date):
    """
    Return (current, prior) annual fundamentals dicts as known on
    as_of_date — the most recent AVAILABLE year and the one before it.
    Both must have been filed by as_of_date. Returns (None, None) if
    fewer than two years are available.
    """
    as_of = pd.Timestamp(as_of_date)
    data = _fetch_raw(ticker)
    if data is None:
        return None, None

    report_dates = _build_availability_map(data)
    fin = data.get("Financials", {})
    income = fin.get("Income_Statement", {}).get("yearly", {})
    balance = fin.get("Balance_Sheet", {}).get("yearly", {})
    cashflow = fin.get("Cash_Flow", {}).get("yearly", {})
    if not income:
        return None, None

    available_years = []
    for date_str in income.keys():
        fy_end = pd.to_datetime(date_str)
        later = [rd for pe, rd in report_dates.items() if pe >= fy_end and rd >= fy_end]
        avail = min(later) if later else fy_end + pd.Timedelta(days=90)
        if avail <= as_of:
            available_years.append(fy_end)

    if len(available_years) < 2:
        return None, None

    available_years.sort()
    curr_end = available_years[-1]
    prev_end = available_years[-2]

    def extract(fy_end):
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

    return extract(curr_end), extract(prev_end)


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


def pit_two_quarters(ticker, as_of_date):
    """
    Quarterly analog of pit_two_years: return (current_quarter, prior_quarter)
    fundamentals dicts as known on as_of_date — the most recent AVAILABLE
    quarter and the one immediately before it. Availability uses each quarter's
    own embedded filing_date (fallback: quarter_end + QUARTERLY_LAG_DAYS). Both
    quarters must have been filed by as_of_date. Returns (None, None) if fewer
    than two quarters are available.

    NOTE: this compares consecutive quarters (Q vs the prior quarter), the
    direct analog of the annual Q-vs-prior-year path. That carries seasonality
    for the flow-based signals; a year-over-year variant is a possible refinement
    but is intentionally out of scope for this additive engine.
    """
    as_of = pd.Timestamp(as_of_date)
    data = _fetch_raw(ticker)
    if data is None:
        return None, None

    fin = data.get("Financials", {})
    income = fin.get("Income_Statement", {}).get("quarterly", {})
    balance = fin.get("Balance_Sheet", {}).get("quarterly", {})
    cashflow = fin.get("Cash_Flow", {}).get("quarterly", {})
    if not income:
        return None, None

    available_quarters = []
    for date_str, inc_row in income.items():
        q_end = pd.to_datetime(date_str)
        avail = _quarter_availability(date_str, inc_row,
                                      balance.get(date_str), cashflow.get(date_str))
        if avail <= as_of:
            available_quarters.append(q_end)

    if len(available_quarters) < 2:
        return None, None

    available_quarters.sort()
    curr_end = available_quarters[-1]
    prev_end = available_quarters[-2]

    def extract(q_end):
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

    return extract(curr_end), extract(prev_end)


def pit_fscore_quarterly(ticker, as_of_date):
    """Piotroski F-score from point-in-time QUARTERLY data. Returns 0-9 or None.
    Same 9 signals and same '>=5 of 9 available' rule as the annual F-score,
    reusing _score_pair — just fed consecutive quarters instead of years."""
    curr, prev = pit_two_quarters(ticker, as_of_date)
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


if __name__ == "__main__":
    for date in ["2018-06-30", "2020-06-30", "2022-06-30", "2024-06-30"]:
        print(f"AAPL F-score {date}: {pit_fscore('AAPL', date)}")
        print(f"AAPL value {date}: {pit_value_inputs('AAPL', date)}")