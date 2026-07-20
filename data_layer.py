import os
import time
import requests
from datetime import datetime
import pandas as pd
import yfinance as yf
from config import EODHD_API_KEY

# ---- Settings ----
PRICE_MAX_AGE_HOURS = 24
FUNDAMENTALS_MAX_AGE_HOURS = 24 * 7

SEC_HEADERS = {
    "User-Agent": "Eugene McEnery genomac44@gmail.com"
}

METRICS = {
    "Revenue": [
        "Revenues",
        "RevenueFromContractWithCustomerIncludingAssessedTax",
        "RevenueFromContractWithCustomerExcludingAssessedTax",
    ],
    "NetIncome": ["NetIncomeLoss"],
    "TotalAssets": ["Assets"],
    "TotalLiabilities": ["Liabilities"],
    "OperatingCashFlow": ["NetCashProvidedByUsedInOperatingActivities"],
    "GrossProfit": ["GrossProfit"],
    "CostOfRevenue": [
        "CostOfRevenue",
        "CostOfGoodsAndServicesSold",
        "CostOfGoodsSold",
    ],
    "CurrentAssets": ["AssetsCurrent"],
    "CurrentLiabilities": ["LiabilitiesCurrent"],
    "LongTermDebt": [
        "LongTermDebtNoncurrent",
        "LongTermDebt",
    ],
    "SharesOutstanding": [
        "CommonStockSharesOutstanding",
        "WeightedAverageNumberOfDilutedSharesOutstanding",
        "WeightedAverageNumberOfSharesOutstandingBasic",
    ],
}


# ---- Freshness helpers ----
def is_fresh(filepath, max_age_hours):
    if not os.path.exists(filepath):
        return False
    file_age_hours = (time.time() - os.path.getmtime(filepath)) / 3600
    return file_age_hours < max_age_hours


def file_age_string(filepath):
    if not os.path.exists(filepath):
        return "does not exist"
    dt = datetime.fromtimestamp(os.path.getmtime(filepath))
    return dt.strftime("%Y-%m-%d %H:%M:%S")


# ---- Prices ----
def get_prices(ticker, max_age_hours=PRICE_MAX_AGE_HOURS):
    ticker = ticker.upper()
    filepath = f"data/{ticker}_prices.csv"

    if is_fresh(filepath, max_age_hours):
        print(f"[{ticker}] Using cached prices (pulled {file_age_string(filepath)})")
        data = pd.read_csv(filepath, index_col=0, parse_dates=True)
    else:
        print(f"[{ticker}] Cache stale or missing — downloading fresh prices...")
        data = yf.download(ticker, period="10y", interval="1d", progress=False)
        data.columns = data.columns.get_level_values(0)
        os.makedirs("data", exist_ok=True)
        data.to_csv(filepath)
        print(f"[{ticker}] Downloaded and saved {len(data)} rows")

    latest_date = data.index[-1].strftime("%Y-%m-%d")
    latest_close = data["Close"].iloc[-1]
    print(f"[{ticker}] Most recent bar: {latest_date} | Close: ${latest_close:,.2f}")
    return data


# ---- Fundamentals ----
def get_fundamentals(ticker, max_age_hours=FUNDAMENTALS_MAX_AGE_HOURS):
    ticker = ticker.upper()
    filepath = f"data/{ticker}_fundamentals.csv"

    if is_fresh(filepath, max_age_hours):
        print(f"[{ticker}] Using cached fundamentals (pulled {file_age_string(filepath)})")
        return pd.read_csv(filepath, index_col=0)

    print(f"[{ticker}] Cache stale or missing — pulling fundamentals from EODHD...")
    url = f"https://eodhd.com/api/fundamentals/{ticker}.US?api_token={EODHD_API_KEY}&fmt=json"
    resp = requests.get(url)
    if resp.status_code != 200:
        print(f"[{ticker}] EODHD request failed (status {resp.status_code})")
        return None

    data = resp.json()
    fin = data.get("Financials", {})
    income = fin.get("Income_Statement", {}).get("yearly", {})
    balance = fin.get("Balance_Sheet", {}).get("yearly", {})
    cashflow = fin.get("Cash_Flow", {}).get("yearly", {})

    if not income:
        print(f"[{ticker}] No fundamentals returned from EODHD")
        return None

    # Map EODHD fields -> our standard column names, by statement
    income_map = {
        "Revenue": "totalRevenue",
        "NetIncome": "netIncome",
        "GrossProfit": "grossProfit",
        "CostOfRevenue": "costOfRevenue",
        "EBITDA": "ebitda",
        "OperatingIncome": "operatingIncome",
    }
    
    
    balance_map = {
        "TotalAssets": "totalAssets",
        "TotalLiabilities": "totalLiab",
        "CurrentAssets": "totalCurrentAssets",
        "CurrentLiabilities": "totalCurrentLiabilities",
        "LongTermDebt": "longTermDebt",
        "SharesOutstanding": "commonStockSharesOutstanding",
    }
    cashflow_map = {
        "OperatingCashFlow": "totalCashFromOperatingActivities",
        "FreeCashFlow": "freeCashFlow",
        "CapEx": "capitalExpenditures",
    }

    def to_float(v):
        try:
            return float(v) if v is not None else None
        except (ValueError, TypeError):
            return None

    rows = {}
    for date_str in income.keys():
        year = date_str[:4]
        inc = income.get(date_str, {})
        bal = balance.get(date_str, {})
        cf = cashflow.get(date_str, {})

        row = {}
        for col, field in income_map.items():
            row[col] = to_float(inc.get(field))
        for col, field in balance_map.items():
            row[col] = to_float(bal.get(field))
        for col, field in cashflow_map.items():
            row[col] = to_float(cf.get(field))

        rows[year] = row

    df = pd.DataFrame.from_dict(rows, orient="index")
    df.index.name = "Year"
    df = df.sort_index()
    df["Ticker"] = ticker

    os.makedirs("data", exist_ok=True)
    df.to_csv(filepath)
    print(f"[{ticker}] Pulled and saved fundamentals ({len(df)} years) from EODHD")
    return df
def get_valuation(ticker, max_age_hours=FUNDAMENTALS_MAX_AGE_HOURS):
    """
    Pull pre-computed valuation metrics and analyst estimates from EODHD:
    trailing/forward P/E, EV/EBITDA, EPS estimates, analyst target & rating.
    Returns a flat dictionary. Cached like fundamentals.
    """
    ticker = ticker.upper()
    filepath = f"data/{ticker}_valuation.csv"

    if is_fresh(filepath, max_age_hours):
        print(f"[{ticker}] Using cached valuation (pulled {file_age_string(filepath)})")
        return pd.read_csv(filepath, index_col=0).iloc[0].to_dict()

    print(f"[{ticker}] Pulling valuation from EODHD...")
    url = f"https://eodhd.com/api/fundamentals/{ticker}.US?api_token={EODHD_API_KEY}&fmt=json"
    resp = requests.get(url)
    if resp.status_code != 200:
        print(f"[{ticker}] EODHD valuation request failed (status {resp.status_code})")
        return None

    data = resp.json()
    val = data.get("Valuation", {})
    high = data.get("Highlights", {})
    ratings = data.get("AnalystRatings", {})

    def to_float(v):
        try:
            return float(v) if v is not None else None
        except (ValueError, TypeError):
            return None

    result = {
        "ticker": ticker,
        "trailing_pe": to_float(val.get("TrailingPE")),
        "forward_pe": to_float(val.get("ForwardPE")),
        "ev_ebitda": to_float(val.get("EnterpriseValueEbitda")),
        "ev_revenue": to_float(val.get("EnterpriseValueRevenue")),
        "price_sales": to_float(val.get("PriceSalesTTM")),
        "enterprise_value": to_float(val.get("EnterpriseValue")),
        "eps_est_current_yr": to_float(high.get("EPSEstimateCurrentYear")),
        "eps_est_next_yr": to_float(high.get("EPSEstimateNextYear")),
        "peg": to_float(high.get("PEGRatio")),
        "wall_st_target": to_float(high.get("WallStreetTargetPrice")),
        "market_cap": to_float(high.get("MarketCapitalizationMln")),
        "analyst_rating": to_float(ratings.get("Rating")),
        "analyst_target": to_float(ratings.get("TargetPrice")),
        "strong_buy": ratings.get("StrongBuy"),
        "buy": ratings.get("Buy"),
        "hold": ratings.get("Hold"),
        "sell": ratings.get("Sell"),
    }

    os.makedirs("data", exist_ok=True)
    pd.DataFrame([result]).to_csv(filepath, index=False)
    print(f"[{ticker}] Pulled and saved valuation")
    return result

# ---- Bulk end-of-day prices (batch refresh) ----
UNIVERSE_PRICE_DIR = "data/universe_prices"
BULK_EOD_URL = "https://eodhd.com/api/eod-bulk-last-day/US"
MIN_HISTORY_ROWS = 260      # factors need ~252 contiguous trading days
MAX_BACKFILL_DAYS = 14      # walk back at most 2 weeks before a full refetch


def bulk_last_day_prices(date=None, symbols=None, timeout=120):
    """One HTTP call for the whole US market's most recent close (or a given
    `date`). Returns {ticker: {"date": 'YYYY-MM-DD', "close": float}}.

    This is EODHD's eod-bulk-last-day endpoint: ~50k tickers in a single
    request (~4s) instead of one call per ticker. `symbols` can restrict the
    result but isn't needed for a universe-wide refresh. Uses adjusted_close to
    stay consistent with the cached history."""
    params = {"api_token": EODHD_API_KEY, "fmt": "json"}
    if date:
        params["date"] = date
    if symbols:
        params["symbols"] = ",".join(symbols)
    try:
        resp = requests.get(BULK_EOD_URL, params=params, timeout=timeout)
        if resp.status_code != 200:
            return {}
        rows = resp.json()
    except Exception:
        return {}

    out = {}
    for row in rows or []:
        code = row.get("code")
        d = row.get("date")
        ac = row.get("adjusted_close")
        if ac is None:
            ac = row.get("close")
        if code and d and ac is not None:
            try:
                out[code] = {"date": d, "close": float(ac)}
            except (TypeError, ValueError):
                pass
    return out


def _cached_last_date(path):
    """(last_date_str, n_rows) for a cached price CSV, reading only the date
    column. Assumes the file is sorted ascending (as build_universe writes it).
    Returns (None, 0) if missing/empty/unreadable."""
    if not os.path.exists(path):
        return None, 0
    try:
        col = pd.read_csv(path, usecols=[0], header=0)
        if len(col) == 0:
            return None, 0
        return str(col.iloc[-1, 0])[:10], len(col)
    except Exception:
        return None, 0


def refresh_universe_prices(tickers, price_dir=UNIVERSE_PRICE_DIR, progress=None,
                            per_ticker_fetch=None, max_backfill_days=MAX_BACKFILL_DAYS):
    """Top up cached daily price CSVs to the latest close using the bulk
    endpoint instead of one call per ticker.

    Strategy:
      - one whole-market snapshot for the latest close, plus one snapshot per
        missing trading day so each series stays contiguous;
      - append only the bars newer than each ticker's last cached date;
      - tickers that are uncached, too short, absent from the feed, or staler
        than max_backfill_days fall back to `per_ticker_fetch` (full-history
        refetch) when one is supplied.

    Does not touch fundamentals or factor logic. Returns a counts dict:
    {updated, fell_back, up_to_date, no_data}."""
    os.makedirs(price_dir, exist_ok=True)
    tickers = list(dict.fromkeys(tickers))      # de-dup, preserve order

    latest = bulk_last_day_prices()
    if not latest:
        return {"updated": 0, "fell_back": 0, "up_to_date": 0,
                "no_data": len(tickers)}
    latest_date = max(bar["date"] for bar in latest.values())

    stale = {}          # ticker -> last cached date (str), needs a few new bars
    fallbacks = []      # ticker -> needs full-history refetch
    up_to_date = 0
    no_data = 0

    for t in tickers:
        path = f"{price_dir}/{t}.csv"
        last_date, n = _cached_last_date(path)
        if last_date is None or n < MIN_HISTORY_ROWS:
            fallbacks.append(t)                 # no/short cache -> full history
            continue
        if last_date >= latest_date:
            up_to_date += 1                     # already current
            continue
        if t not in latest:
            no_data += 1                        # cached but delisted / absent
            continue
        if (pd.Timestamp(latest_date) - pd.Timestamp(last_date)).days > max_backfill_days:
            fallbacks.append(t)                 # too stale for day-by-day fill
            continue
        stale[t] = last_date

    # Collect the daily snapshots needed to fill gaps contiguously.
    snapshots = {latest_date: latest}
    if stale:
        floor = pd.Timestamp(min(stale.values()))
        day = pd.Timestamp(latest_date) - pd.Timedelta(days=1)
        while day > floor:
            snap = bulk_last_day_prices(date=day.strftime("%Y-%m-%d"))
            if snap:                            # empty on weekends/holidays
                real_date = max(bar["date"] for bar in snap.values())
                snapshots[real_date] = snap
            day -= pd.Timedelta(days=1)

    # Append missing bars per stale ticker.
    updated = 0
    ordered_dates = sorted(snapshots.keys())
    total = len(stale)
    for i, (t, last_date) in enumerate(stale.items(), 1):
        new_rows = [(d, snapshots[d][t]["close"])
                    for d in ordered_dates
                    if d > last_date and t in snapshots[d]]
        if new_rows:
            path = f"{price_dir}/{t}.csv"
            try:
                # new_rows are strictly newer than the cached last date and in
                # ascending order, and the file is a sorted date,Close CSV, so
                # we can append raw lines instead of rewriting the whole file.
                with open(path, "a", newline="") as fh:
                    fh.write("".join(f"{d},{c}\n" for d, c in new_rows))
                updated += 1
            except Exception:
                fallbacks.append(t)
        if progress:
            progress(i, total)

    # Full-history fallback for everything the bulk path couldn't top up.
    fell_back = 0
    if per_ticker_fetch:
        for t in fallbacks:
            if per_ticker_fetch(t) is not None:
                fell_back += 1

    return {"updated": updated, "fell_back": fell_back,
            "up_to_date": up_to_date, "no_data": no_data}


# ---- Sector (cached per ticker) ----
SECTOR_CACHE = "data/sectors.csv"
# In-memory mirror of the CSV so a full-universe scan doesn't re-read the file
# once per ticker. Lazily loaded on first use; persists across Streamlit reruns.
_sector_mem = None


def _load_sector_cache():
    global _sector_mem
    if _sector_mem is None:
        _sector_mem = {}
        if os.path.exists(SECTOR_CACHE):
            try:
                df = pd.read_csv(SECTOR_CACHE)
                _sector_mem = {str(t).upper(): str(s)
                               for t, s in zip(df["ticker"], df["sector"])}
            except Exception:
                _sector_mem = {}
    return _sector_mem


def _save_sector_cache(cache):
    os.makedirs("data", exist_ok=True)
    (pd.DataFrame(sorted(cache.items()), columns=["ticker", "sector"])
     .to_csv(SECTOR_CACHE, index=False))


def get_sector(ticker):
    """Return the EODHD General.Sector for a ticker, cache-first.

    Reads the local sector cache; on a miss, pulls just the General.Sector field
    from EODHD and stores it. Anything missing or a failed request resolves to
    'Unknown' — and that result is cached too, so an unknown ticker isn't
    re-pulled on every scan."""
    ticker = ticker.upper()
    cache = _load_sector_cache()
    if ticker in cache:
        return cache[ticker]

    sector = "Unknown"
    try:
        url = (f"https://eodhd.com/api/fundamentals/{ticker}.US"
               f"?api_token={EODHD_API_KEY}&fmt=json&filter=General::Sector")
        resp = requests.get(url, timeout=30)
        if resp.status_code == 200:
            val = resp.json()
            if isinstance(val, str) and val.strip():
                sector = val.strip()
    except Exception:
        pass

    cache[ticker] = sector
    _save_sector_cache(cache)
    return sector


# ---- Test (runs only when this file is run directly) ----
if __name__ == "__main__":
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", None)

    prices = get_prices("COST")
    print()
    fundamentals = get_fundamentals("COST")
    print(fundamentals)