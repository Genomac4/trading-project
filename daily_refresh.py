"""
daily_refresh.py — headless daily refresh: prices, then the IV snapshot.

Step 1 runs the fast bulk EOD top-up (data_layer.refresh_universe_prices) over
the current S&P 500 members. Step 2 runs the options satellite's IV snapshotter
(options/snapshotter.py), which appends today's ATM-IV30 readings to an
append-only history that can never be backfilled. Each step logs its own line
to data/refresh_log.txt and neither can take the other down. No user
interaction — intended to be run once a day by Windows Task Scheduler.
NOTE: step 2 needs a valid Schwab token (7-day refresh cycle) — when it starts
failing with auth errors, run `python schwab_client.py login`.

One bulk call tops up the whole list to the latest close; recently-listed or
very stale names fall back to a per-ticker full-history fetch automatically.

Stock splits are rare (a handful of S&P names a year). This job does NOT try to
re-adjust for them. If a split-related price glitch is ever noticed by hand,
the fix is simply to re-run build_universe.py manually — nothing to automate.
"""

import os
import sys
import traceback
from datetime import datetime

# Run from this script's own directory so relative data/ paths resolve no
# matter what working directory Task Scheduler launches us in.
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from explore_constituents import get_sp500_membership
from build_universe import get_eodhd_prices, PRICE_DIR
from data_layer import refresh_universe_prices

LOG_PATH = "data/refresh_log.txt"


def _log(line):
    os.makedirs("data", exist_ok=True)
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(LOG_PATH, "a", encoding="utf-8") as fh:
        fh.write(f"[{stamp}] {line}\n")


def _fetch_and_save(ticker):
    """Full-history fallback that also persists the CSV (the bulk feed only
    returns one day, so it can't seed new listings or fill long gaps)."""
    df = get_eodhd_prices(ticker)
    if df is not None and len(df) > 0:
        df.to_csv(f"{PRICE_DIR}/{ticker}.csv")
        return df
    return None


def refresh_prices_step():
    start = datetime.now()
    try:
        members = get_sp500_membership()
        tickers = sorted(members[members["end"].isna()]["ticker"].dropna().unique())
        counts = refresh_universe_prices(
            tickers, price_dir=PRICE_DIR, per_ticker_fetch=_fetch_and_save)
        elapsed = (datetime.now() - start).total_seconds()
        _log(f"OK  tickers={len(tickers)} updated={counts['updated']} "
             f"fallback={counts['fell_back']} current={counts['up_to_date']} "
             f"no_data={counts['no_data']} elapsed={elapsed:.1f}s")
        return 0
    except Exception as e:
        _log(f"ERROR {type(e).__name__}: {e}")
        _log("TRACE " + traceback.format_exc().replace("\n", " | "))
        return 1


def snapshot_iv_step():
    """Options satellite: accrue today's IV history (it can never be
    backfilled, so a silently skipped day is permanently lost — hence its
    failure also fails the run so Task Scheduler shows it). Runs after the
    price refresh so the realized-vol proxy sees today's closes; kept
    independent so neither step can take the other down."""
    try:
        from options import snapshotter
        written = snapshotter.run()
        _log(f"OK  iv-snapshot rows={len(written)}")
        return 0
    except Exception as e:
        _log(f"ERROR iv-snapshot {type(e).__name__}: {e}")
        _log("TRACE " + traceback.format_exc().replace("\n", " | "))
        return 1


def main():
    rc_prices = refresh_prices_step()
    rc_iv = snapshot_iv_step()
    return 1 if (rc_prices or rc_iv) else 0


if __name__ == "__main__":
    sys.exit(main())
