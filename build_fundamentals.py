import os
import time
from point_in_time import _fetch_raw
from explore_constituents import get_sp500_membership

CACHE_DIR = "data/pit_fundamentals"


def build_fundamentals():
    os.makedirs(CACHE_DIR, exist_ok=True)
    membership = get_sp500_membership()
    tickers = sorted(membership["ticker"].dropna().unique())
    print(f"Pulling fundamentals for {len(tickers)} tickers\n")

    pulled, skipped, failed = 0, 0, []
    for i, ticker in enumerate(tickers, 1):
        path = f"{CACHE_DIR}/{ticker}.json"
        if os.path.exists(path):
            skipped += 1
            continue
        data = _fetch_raw(ticker)
        if data is not None:
            pulled += 1
        else:
            failed.append(ticker)
        if i % 25 == 0:
            print(f"[{i}/{len(tickers)}] pulled={pulled} skipped={skipped} failed={len(failed)}")
        time.sleep(0.05)

    print(f"\n=== DONE === pulled={pulled} skipped={skipped} failed={len(failed)}")
    if failed:
        print(f"Failed: {failed[:20]}")


if __name__ == "__main__":
    build_fundamentals()