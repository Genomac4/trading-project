"""
options/snapshotter.py — daily IV history writer for the options satellite.

Starts and feeds the IV-history clock. Runs unattended off daily_refresh.py.
For each name on a focused watchlist it records, per day:
  - ATM ~30-DTE implied volatility (from the live Schwab chain), plus the
    constant-maturity 30d interpolation — see interpolate_iv30 for why the
    single-contract reading alone is not a safe IV-rank input
  - the contract's Greeks, quote, OI at sample time
  - a 30-day realized-vol proxy (price-derived, the fallback comparison)

Why it exists: IV-rank (rubric category 3, trigger #3) needs "today's IV vs. its
own history." Schwab serves no options history, so we accrue it ourselves — and
that history NEVER backfills. Every day this doesn't run is a day permanently lost.

Storage: one append-only CSV (data/options/iv_history.csv), keyed (date, ticker).
Re-running the same day overwrites that day's rows cleanly; it never double-writes.
`date` is the TRADING SESSION the quotes came from, not the wall-clock day of the
run — see session_date_for for why that distinction protects the IV-rank history.

Repo wiring (the five seams, now live):
  - universe:   current S&P 500 members (this repo screens the 500 only — the
                original design said 500+400, but there is no S&P 400 here)
  - scores:     the latest data/scan_history/{date}.csv snapshot — the same
                composite the dashboard scan saved, never recomputed here.
                The watchlist is therefore only as fresh as the last scan.
  - earnings:   EODHD earnings calendar (same endpoint the dashboard uses)
  - prices:     data/universe_prices/{ticker}.csv, kept current by daily_refresh
  - chain:      schwab_client.get_option_chain (read-only market-data GET)
"""

from __future__ import annotations

import logging
import os
import sys
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import requests

# Repo-root anchoring: data paths and flat root modules (config, schwab_client,
# explore_constituents) must resolve whether we're imported from daily_refresh
# or run directly as `python options/snapshotter.py`.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

log = logging.getLogger("options.snapshotter")

# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #

IV_HISTORY_PATH = ROOT / "data/options/iv_history.csv"
RUN_LOG_PATH = ROOT / "data/options/snapshotter_log.txt"
SCAN_HISTORY_DIR = ROOT / "data/scan_history"
PRICE_DIR = ROOT / "data/universe_prices"

TARGET_DTE = 30          # sample ATM IV at ~30 days to expiry (IV30 convention)
MIN_DTE = 10             # don't sample the dying front week
MAX_DTE = 60             # bound the chain request; nothing past this is needed
STRIKE_COUNT = 8         # strikes around ATM per expiry — keeps payloads small
TOP_DECILE_FRACTION = 0.10
EARNINGS_WINDOW_DAYS = 45
EARNINGS_SCORE_FRACTION = 0.33   # catalyst names must rank in the top third
SCAN_STALE_WARN_DAYS = 7         # warn when the scores snapshot is older than this
RV_WINDOW = 30           # trading days for the realized-vol proxy
REQUEST_SLEEP_SEC = 0.55 # throttle: ~110 chain calls/min, under Schwab's ~120 limit

# Upper sanity bound on a normalized IV, in DECIMAL (5.0 = 500%). Schwab's -999
# sentinel is caught by the >0 test, but nothing bounded the top end, so a thin
# or stale strike's garbage model mark (observed shapes: volatility=1609.739 ->
# 16.09, volatility=99999 -> 999.99) would land in iv_history.csv as a real
# reading. That history can never be regenerated, so one junk row permanently
# skews that name's IV-rank percentiles. 500% is far above any genuine ATM
# 30-DTE quote — even a biotech binary or a squeeze prints well under it — so
# the band only ever catches unambiguous junk.
IV_SANITY_MAX = 5.0

# Count of contracts dropped for out-of-band IV, per run. Module-level so
# fetch_chain can report without changing its documented return contract;
# run() resets it and folds the total into the run-log line.
_iv_out_of_band = 0


# --------------------------------------------------------------------------- #
# Normalized contract — the ONLY Schwab-specific parsing lives in the adapter
# that produces these. Everything downstream is vendor-agnostic and testable.
# --------------------------------------------------------------------------- #

@dataclass
class Contract:
    expiry: date
    dte: int
    strike: float
    iv: float            # implied vol, decimal (0.34 = 34%)
    delta: float
    gamma: Optional[float]
    theta: float
    vega: float
    bid: float
    ask: float
    open_interest: int
    option_type: str = "call"
    quote_time: Optional[datetime] = None   # when the vendor last priced it


def session_date_for(contract: Contract, fallback: date) -> date:
    """The trading session a contract's quote belongs to.

    Schwab keeps serving the last session's quotes when the market is shut, so
    an unattended weekend/holiday run would otherwise stamp Friday's closing IV
    with Saturday's and Sunday's dates — three identical readings that would
    quietly bias every IV-rank percentile computed off this history. Keying on
    the quote's own session instead makes those runs idempotent rewrites of the
    real session rather than phantom observations."""
    if contract.quote_time is None:
        return fallback
    return contract.quote_time.date()


# --------------------------------------------------------------------------- #
# PURE LOGIC — no repo or network dependency
# --------------------------------------------------------------------------- #

def normalize_iv(volatility_pct, iv_max: float = IV_SANITY_MAX) -> Optional[float]:
    """Schwab's `volatility` (a PERCENT, e.g. 29.062) -> decimal IV (0.29062),
    or None when the value is unusable and the contract must be dropped.

    Rejects, all treated as missing rather than recorded:
      - None / non-numeric
      - <= 0, which is Schwab's -999.0 missing-value sentinel
      - > iv_max, i.e. a garbage model mark (see IV_SANITY_MAX)

    Dropping is the right call rather than clamping: a clamped 500% would be a
    fabricated observation in a history that can never be corrected, whereas a
    missing contract simply falls through to the next strike or leaves the name
    out for the day — visible in the drop log either way."""
    if volatility_pct is None:
        return None
    try:
        iv = float(volatility_pct) / 100.0
    except (TypeError, ValueError):
        return None
    if iv <= 0 or iv > iv_max:
        return None
    return iv


def realized_vol(closes: list[float], window: int = RV_WINDOW,
                 trading_days: int = 252) -> Optional[float]:
    """Annualized close-to-close realized vol over the trailing `window` days.
    Returns None if there isn't enough history."""
    if closes is None or len(closes) < window + 1:
        return None
    arr = np.asarray(closes[-(window + 1):], dtype=float)
    rets = np.diff(np.log(arr))
    if rets.size < 2:
        return None
    return float(np.std(rets, ddof=1) * np.sqrt(trading_days))


def select_atm_iv30(spot: float, contracts: list[Contract],
                    target_dte: int = TARGET_DTE, min_dte: int = MIN_DTE) -> Optional[Contract]:
    """Pick the ATM call nearest `target_dte`. Expiry = closest DTE to target among
    expiries at/after min_dte; strike = closest to spot within that expiry.
    Returns None if nothing qualifies."""
    calls = [c for c in contracts if c.option_type == "call" and c.dte >= min_dte]
    if not calls:
        return None
    # choose the expiry (by dte) closest to target; tie → the longer one
    dtes = sorted({c.dte for c in calls})
    chosen_dte = min(dtes, key=lambda d: (abs(d - target_dte), -d))
    at_expiry = [c for c in calls if c.dte == chosen_dte]
    return min(at_expiry, key=lambda c: abs(c.strike - spot))


def atm_contract_at_each_expiry(spot: float, contracts: list[Contract],
                                min_dte: int = MIN_DTE) -> list[tuple[int, Contract]]:
    """[(dte, atm_contract), ...] — the strike-nearest-spot CONTRACT for every
    expiry, sorted by DTE. Carries the whole contract (not just its IV) so the
    interpolation and its open-interest provenance are read off the same object
    and cannot describe different strikes."""
    calls = [c for c in contracts if c.option_type == "call" and c.dte >= min_dte]
    by_dte: dict[int, list[Contract]] = {}
    for c in calls:
        by_dte.setdefault(c.dte, []).append(c)
    return [(dte, min(by_dte[dte], key=lambda c: abs(c.strike - spot)))
            for dte in sorted(by_dte)]


def atm_iv_at_each_expiry(spot: float, contracts: list[Contract],
                          min_dte: int = MIN_DTE) -> list[tuple[int, float]]:
    """[(dte, atm_iv), ...] — the strike-nearest-spot IV for every expiry,
    sorted by DTE. The per-expiry input to constant-maturity interpolation."""
    return [(dte, c.iv)
            for dte, c in atm_contract_at_each_expiry(spot, contracts, min_dte)]


def interpolate_iv30(spot: float, contracts: list[Contract],
                     target_dte: int = TARGET_DTE,
                     min_dte: int = MIN_DTE) -> Optional[float]:
    """Constant-maturity target_dte IV, interpolated across the two expiries
    that bracket it. Returns None when the chain doesn't bracket the target.

    Why this exists alongside select_atm_iv30: names with only monthly expiries
    have no contract near 30 DTE, so single-contract sampling sawtooths through
    roughly 18->39->32->25 DTE as expiries roll. Term structure then moves the
    recorded IV for calendar reasons rather than volatility reasons, which would
    corrupt any IV-rank percentile computed off the series. Interpolating to a
    fixed tenor removes that artifact.

    Interpolates TOTAL VARIANCE linearly in time (sigma^2 * t), the standard
    convention — variance is additive in time, volatility is not, so
    interpolating vol directly would bias the result."""
    points = atm_iv_at_each_expiry(spot, contracts, min_dte)
    if len(points) < 2:
        return None
    below = [p for p in points if p[0] <= target_dte]
    above = [p for p in points if p[0] >= target_dte]
    if not below or not above:
        return None          # target outside the listed range — don't extrapolate
    t1, iv1 = below[-1]
    t2, iv2 = above[0]
    if t1 == t2:
        return float(iv1)
    v1, v2 = iv1 ** 2 * t1, iv2 ** 2 * t2
    v_target = v1 + (v2 - v1) * (target_dte - t1) / (t2 - t1)
    if v_target <= 0:
        return None
    return float(np.sqrt(v_target / target_dte))


def constant_maturity_iv30(spot: float, contracts: list[Contract],
                           target_dte: int = TARGET_DTE, min_dte: int = MIN_DTE
                           ) -> tuple[Optional[float], Optional[str], Optional[int]]:
    """(iv, method, min_oi) for the constant-maturity reading — the value written
    to the history, an honest label for how it was obtained, and the open
    interest backing it.

    min_oi is the LOWER open interest of the two bracketing contracts that fed
    the interpolation (or, on a clamped row, the OI of the single contract
    clamped to). It exists because the row's `open_interest` field describes the
    select_atm_iv30 contract behind `atm_iv` — a DIFFERENT contract — so
    down-weighting iv30_interp by that field would weight the wrong quote. The
    lower of the pair is the honest figure: an interpolation is only as trusted
    as its thinnest input.

    method is one of:
      'interpolated' — 30 DTE sat between two listed expiries; clean variance
                       interpolation via interpolate_iv30.
      'clamped_high' — every usable expiry is LONGER than 30 DTE (the usual
                       case: the front monthly rolled inside min_dte). Value is
                       the nearest available tenor, NOT extrapolated.
      'clamped_low'  — every usable expiry is SHORTER than 30 DTE. Same clamp.
      None           — no usable contracts at all; value is None.

    Clamping rather than extrapolating is deliberate. Extending the variance
    line past the last listed expiry is unbounded — a steep near-term skew can
    drive it to absurd or negative variance — whereas the nearest listed tenor
    is always a real traded quote. It is biased (a 37-DTE vol recorded as the
    30-day reading) but bounded, and the method flag means downstream can drop
    or down-weight those rows instead of silently trusting them. For a
    monthly-only name this fires on roughly a third of trading days, so the
    flag is load-bearing, not a rare-path nicety."""
    points = atm_contract_at_each_expiry(spot, contracts, min_dte)
    if not points:
        return None, None, None

    exact = interpolate_iv30(spot, contracts, target_dte, min_dte)
    if exact is not None:
        # Same filter and ordering interpolate_iv30 used, so this reselects the
        # identical bracketing pair — the OI provenance cannot drift from the
        # contracts that actually produced the value.
        below = [p for p in points if p[0] <= target_dte]
        above = [p for p in points if p[0] >= target_dte]
        c1, c2 = below[-1][1], above[0][1]
        return exact, "interpolated", min(c1.open_interest, c2.open_interest)

    nearest_dte, nearest = min(points, key=lambda p: abs(p[0] - target_dte))
    method = "clamped_low" if nearest_dte < target_dte else "clamped_high"
    return float(nearest.iv), method, nearest.open_interest


def build_watchlist(universe: list[str], scores: dict[str, float],
                    earnings: dict[str, date],
                    top_decile_fraction: float = TOP_DECILE_FRACTION,
                    earnings_score_fraction: float = EARNINGS_SCORE_FRACTION) -> list[str]:
    """Union of (a) top-decile composite names and (b) names with earnings inside
    the window that also rank in the top third by composite. Pure function of its
    inputs so it can be tested and audited."""
    scored = [t for t in universe if t in scores]
    ranked = sorted(scored, key=lambda t: scores[t], reverse=True)

    n_decile = max(1, int(len(ranked) * top_decile_fraction))
    top_decile = set(ranked[:n_decile])

    n_third = max(1, int(len(ranked) * earnings_score_fraction))
    top_third = set(ranked[:n_third])
    catalyst = {t for t in earnings if t in top_third}

    return sorted(top_decile | catalyst)


def _dedup_append(existing: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    """Append `new`, dropping any existing rows sharing (date, ticker) with it —
    so a same-day re-run overwrites that day cleanly instead of duplicating."""
    if existing.empty:
        return new
    keys = set(zip(new["date"], new["ticker"]))
    mask = [(d, t) not in keys for d, t in zip(existing["date"], existing["ticker"])]
    return pd.concat([existing[mask], new], ignore_index=True)


# --------------------------------------------------------------------------- #
# ADAPTERS — wired to this repo
# --------------------------------------------------------------------------- #

def get_universe() -> list[str]:
    """Current S&P 500 members (no end date in the membership table)."""
    from explore_constituents import get_sp500_membership
    members = get_sp500_membership()
    return sorted(members[members["end"].isna()]["ticker"].dropna().unique())


def latest_scan_path() -> Optional[Path]:
    if not SCAN_HISTORY_DIR.is_dir():
        return None
    snaps = sorted(SCAN_HISTORY_DIR.glob("*.csv"))
    return snaps[-1] if snaps else None


def get_composite_scores() -> dict[str, float]:
    """{ticker: composite} from the most recent dashboard scan snapshot.

    Deliberately NOT recomputed here: the snapshot is the exact composite the
    dashboard saved (same engine, same regime weights). The cost is staleness —
    if no scan has been saved recently the watchlist drifts, so we warn past
    SCAN_STALE_WARN_DAYS."""
    path = latest_scan_path()
    if path is None:
        log.warning("no scan_history snapshots found — watchlist will be empty")
        return {}
    age = (date.today() - date.fromisoformat(path.stem)).days
    if age > SCAN_STALE_WARN_DAYS:
        log.warning("scores snapshot %s is %d days old — run a dashboard scan "
                    "to refresh the watchlist", path.name, age)
    df = pd.read_csv(path)
    return {str(t): float(c) for t, c in zip(df["Ticker"], df["composite"])
            if pd.notna(c)}


def get_earnings_calendar(within_days: int = EARNINGS_WINDOW_DAYS) -> dict[str, date]:
    """{ticker: next_earnings_date} for scored names reporting within
    `within_days`, from the EODHD earnings calendar (same endpoint the
    dashboard's Market Awareness tab uses, batched 30 symbols per call)."""
    from config import EODHD_API_KEY
    tickers = sorted(get_composite_scores())
    start = date.today()
    end = start + timedelta(days=within_days)
    out: dict[str, date] = {}
    for i in range(0, len(tickers), 30):
        chunk = tickers[i:i + 30]
        url = (f"https://eodhd.com/api/calendar/earnings?api_token={EODHD_API_KEY}"
               f"&symbols={','.join(t + '.US' for t in chunk)}"
               f"&from={start}&to={end}&fmt=json")
        try:
            resp = requests.get(url, timeout=30)
            if resp.status_code != 200:
                continue
            for e in resp.json().get("earnings", []):
                tkr = str(e.get("code", "")).replace(".US", "")
                rd = str(e.get("report_date", ""))
                try:
                    d = date.fromisoformat(rd)
                except ValueError:
                    continue
                if tkr and (tkr not in out or d < out[tkr]):
                    out[tkr] = d
        except requests.RequestException:
            continue
    return out


def get_price_history(ticker: str) -> Optional[list[float]]:
    """Trailing daily closes (oldest→newest) from the universe price store that
    daily_refresh keeps current. None if the ticker has no cached prices."""
    path = PRICE_DIR / f"{ticker}.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    if "Close" not in df.columns:
        return None
    closes = df["Close"].dropna()
    return list(closes.values) if len(closes) else None


def fetch_chain(ticker: str) -> tuple[float, list[Contract]]:
    """(spot_price, [Contract, ...]) from the live Schwab chain — calls only,
    expiries bounded to [MIN_DTE, MAX_DTE], STRIKE_COUNT strikes around ATM.

    The ONLY Schwab-specific parsing in the module. Field names VERIFIED against
    a live AAPL chain on 2026-08-03: top level has 'underlyingPrice' and
    'callExpDateMap' keyed "YYYY-MM-DD:DTE"; each contract dict carries bid,
    ask, delta, gamma, theta, vega, volatility (IV in PERCENT — 29.062, divided
    by 100 here), openInterest, daysToExpiration, strikePrice. Expiry is parsed
    from the map key, not the contract's 'expirationDate' (an ISO timestamp).
    IV is run through normalize_iv, which drops both Schwab's -999.0 sentinel
    and out-of-band garbage marks; the out-of-band count is logged per ticker
    and accumulated into the run total."""
    global _iv_out_of_band
    import schwab_client
    today = date.today()
    data = schwab_client.get_option_chain(
        ticker,
        from_date=today + timedelta(days=MIN_DTE),
        to_date=today + timedelta(days=MAX_DTE),
        strike_count=STRIKE_COUNT,
    )
    spot = float(data.get("underlyingPrice") or 0.0)
    contracts: list[Contract] = []
    dropped_out_of_band = 0
    for exp_key, strikes in (data.get("callExpDateMap") or {}).items():
        exp_str, _, dte_str = exp_key.partition(":")
        try:
            expiry = date.fromisoformat(exp_str)
            key_dte = int(dte_str)
        except ValueError:
            continue
        for _strike, quotes in strikes.items():
            for q in quotes:
                iv = normalize_iv(q.get("volatility"))
                if iv is None:
                    # Separate the junk-mark case from the ordinary -999 sentinel
                    # so a name vanishing from the history is visible, not silent.
                    raw = q.get("volatility")
                    if raw is not None:
                        try:
                            if float(raw) / 100.0 > IV_SANITY_MAX:
                                dropped_out_of_band += 1
                        except (TypeError, ValueError):
                            pass
                    continue
                try:
                    contracts.append(Contract(
                        expiry=expiry,
                        dte=int(q.get("daysToExpiration", key_dte)),
                        strike=float(q["strikePrice"]),
                        iv=iv,
                        delta=float(q.get("delta", 0.0)),
                        gamma=(float(q["gamma"]) if q.get("gamma") is not None else None),
                        theta=float(q.get("theta", 0.0)),
                        vega=float(q.get("vega", 0.0)),
                        bid=float(q.get("bid", 0.0)),
                        ask=float(q.get("ask", 0.0)),
                        open_interest=int(q.get("openInterest", 0)),
                        option_type="call",
                        quote_time=(datetime.fromtimestamp(q["quoteTimeInLong"] / 1000)
                                    if q.get("quoteTimeInLong") else None),
                    ))
                except (KeyError, TypeError, ValueError):
                    continue
    if dropped_out_of_band:
        _iv_out_of_band += dropped_out_of_band
        log.warning("%s: dropped %d contract(s) with out-of-band IV "
                    "(>%.0f%% — garbage model mark, not recorded)",
                    ticker, dropped_out_of_band, IV_SANITY_MAX * 100)
    return spot, contracts


# --------------------------------------------------------------------------- #
# ORCHESTRATION
# --------------------------------------------------------------------------- #

def _row_for(ticker: str, as_of: date) -> Optional[dict]:
    spot, contracts = fetch_chain(ticker)
    atm = select_atm_iv30(spot, contracts)
    if atm is None:
        log.warning("%s: no qualifying ATM contract, skipping", ticker)
        return None
    rv = realized_vol(get_price_history(ticker))
    session = session_date_for(atm, as_of)
    iv30, iv30_method, iv30_min_oi = constant_maturity_iv30(spot, contracts)
    return {
        "date": session.isoformat(),
        "ticker": ticker,
        "spot": round(spot, 4),
        "atm_iv": round(atm.iv, 4),
        # Constant-maturity 30d IV — the tenor-stable series IV-rank should use.
        # ALWAYS read alongside iv30_method: only 'interpolated' rows are clean
        # 30-day readings; 'clamped_*' rows are the nearest listed tenor instead.
        "iv30_interp": round(iv30, 4) if iv30 is not None else None,
        "iv30_method": iv30_method,
        # OI of the contracts behind iv30_interp (the lower of the bracketing
        # pair) — NOT the same contract as `open_interest` below, which backs
        # atm_iv. Use this one to down-weight model-mark iv30 readings.
        "iv30_min_oi": iv30_min_oi,
        "sampled_dte": atm.dte,           # the tenor atm_iv actually came from
        "sampled_strike": atm.strike,
        "delta": atm.delta,
        "gamma": atm.gamma,
        "theta": atm.theta,
        "vega": atm.vega,
        "bid": atm.bid,
        "ask": atm.ask,
        "open_interest": atm.open_interest,
        "realized_vol_30": round(rv, 4) if rv is not None else None,
    }


def run(as_of: Optional[date] = None) -> pd.DataFrame:
    """Build the watchlist, snapshot each name, append to history. One name failing
    never kills the run. Returns the rows written (for logging/inspection)."""
    global _iv_out_of_band
    as_of = as_of or date.today()
    IV_HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    _iv_out_of_band = 0

    watchlist = build_watchlist(get_universe(), get_composite_scores(),
                                get_earnings_calendar())
    log.info("snapshotter: %d names on watchlist for %s", len(watchlist), as_of)

    rows, failures = [], 0
    for ticker in watchlist:
        try:
            row = _row_for(ticker, as_of)
            if row is not None:
                rows.append(row)
        except Exception as e:            # noqa: BLE001 — one bad name must not stop the run
            failures += 1
            log.warning("%s: snapshot failed (%s)", ticker, e)
        time.sleep(REQUEST_SLEEP_SEC)     # respect Schwab's rate limit

    new = pd.DataFrame(rows)
    if not new.empty:
        stamped = sorted(new["date"].unique())
        if stamped != [as_of.isoformat()]:
            log.warning("quotes belong to session(s) %s, not %s — recording "
                        "under the session date (market closed?)",
                        ", ".join(stamped), as_of)
        existing = (pd.read_csv(IV_HISTORY_PATH)
                    if IV_HISTORY_PATH.exists() else pd.DataFrame(columns=new.columns))
        combined = _dedup_append(existing, new)
        combined.to_csv(IV_HISTORY_PATH, index=False)

    msg = (f"{datetime.now().isoformat(timespec='seconds')} | {as_of} | "
           f"watchlist={len(watchlist)} written={len(rows)} failures={failures} "
           f"iv_out_of_band={_iv_out_of_band}")
    with open(RUN_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(msg + "\n")
    log.info(msg)
    return new


if __name__ == "__main__":
    os.chdir(ROOT)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    run()
