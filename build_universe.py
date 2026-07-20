import os
import sys
import time
import requests
import pandas as pd
from config import EODHD_API_KEY

# Import the membership machinery we already built
from explore_constituents import get_sp500_membership
from data_layer import refresh_universe_prices

PRICE_DIR = "data/universe_prices"


def get_eodhd_prices(ticker):
    """Pull daily adjusted close history from EODHD for one ticker."""
    url = (f"https://eodhd.com/api/eod/{ticker}.US"
           f"?api_token={EODHD_API_KEY}&fmt=json&period=d")
    try:
        resp = requests.get(url, timeout=30)
        if resp.status_code != 200:
            return None
        data = resp.json()
        if not data:
            return None
        df = pd.DataFrame(data)
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date")
        return df[["adjusted_close"]].rename(columns={"adjusted_close": "Close"})
    except Exception:
        return None


def build_universe():
    """Pull and cache prices for every stock ever in the S&P 500."""
    os.makedirs(PRICE_DIR, exist_ok=True)

    membership = get_sp500_membership()
    tickers = sorted(membership["ticker"].dropna().unique())
    print(f"Universe: {len(tickers)} unique tickers ever in S&P 500\n")

    pulled, skipped, failed = 0, 0, []

    for i, ticker in enumerate(tickers, 1):
        filepath = f"{PRICE_DIR}/{ticker}.csv"

        # Skip if already cached
        if os.path.exists(filepath):
            skipped += 1
            continue

        df = get_eodhd_prices(ticker)
        if df is not None and len(df) > 0:
            df.to_csv(filepath)
            pulled += 1
        else:
            failed.append(ticker)

        # Progress every 25 stocks
        if i % 25 == 0:
            print(f"[{i}/{len(tickers)}] pulled={pulled} "
                  f"skipped={skipped} failed={len(failed)}")

        time.sleep(0.05)  # gentle pause to be polite to the API

    print(f"\n=== DONE ===")
    print(f"Pulled: {pulled} | Skipped (cached): {skipped} | Failed: {len(failed)}")
    if failed:
        print(f"Failed tickers: {failed[:30]}{'...' if len(failed) > 30 else ''}")


def refresh_universe():
    """Top up already-cached tickers to the latest close via the bulk endpoint.

    Unlike build_universe (which pulls full history one ticker at a time for
    NEW names — the bulk last-day feed can't provide back-history), this only
    appends the most recent bars and does it in a handful of batched calls."""
    membership = get_sp500_membership()
    tickers = sorted(membership["ticker"].dropna().unique())
    print(f"Refreshing {len(tickers)} tickers via bulk last-day endpoint...\n")

    def progress(done, total):
        if total and (done % 25 == 0 or done == total):
            print(f"  backfilled {done}/{total} stale tickers")

    def fetch_and_save(ticker):
        """Full-history fallback that also persists the CSV (get_eodhd_prices
        only returns the frame; the build loop normally writes it)."""
        df = get_eodhd_prices(ticker)
        if df is not None and len(df) > 0:
            df.to_csv(f"{PRICE_DIR}/{ticker}.csv")
            return df
        return None

    counts = refresh_universe_prices(
        tickers, price_dir=PRICE_DIR, progress=progress,
        per_ticker_fetch=fetch_and_save)
    print(f"\n=== REFRESH DONE ===")
    print(f"updated={counts['updated']} | fallback(full refetch)={counts['fell_back']} | "
          f"already-current={counts['up_to_date']} | no-data={counts['no_data']}")


if __name__ == "__main__":
    # `python build_universe.py refresh` -> fast bulk top-up of cached tickers.
    # `python build_universe.py`         -> initial full-history build.
    if len(sys.argv) > 1 and sys.argv[1] == "refresh":
        refresh_universe()
    else:
        build_universe()