"""
options/triggers/pead.py — Trigger #1: Post-Earnings Announcement Drift.

The flagship trigger. Documented anomaly (Bernard & Thomas 1989): after a large
earnings surprise, the stock keeps drifting the SAME direction for ~60 trading days
as the market finishes repricing a resolved fact. We trade the drift with LONG options:
  - beat  → drift up   → long CALL candidate
  - miss  → drift down → long PUT  candidate
Both are long premium (capped-debit), both legal at Schwab Level 1. A long put is NOT
shorting — it's a bought option.

This module ONLY generates candidates (ticker + direction + surprise metrics). It does
not pick a contract or judge tradeability — data-assembly pulls the chain, the gauntlet
filters, the agents write the memo. Candidates carry a TriggerInfo (from the schema) so
they slot straight into an OptionsTicket downstream.

--------------------------------------------------------------------------------
FILTERING (hardened — a flat % EPS beat is too noisy to use raw):
  - denominator floor: |consensus EPS| >= $0.20, so a $0.04→$0.06 "50% beat" (rounding
    noise on a tiny base) can't qualify
  - real coverage: >= 3 analysts, so "consensus" isn't one stale estimate
  - surprise bar: |surprise| >= 8% (PROVISIONAL — raw magnitude is logged on every
    candidate so the Bookkeeper can move this from data; SUE is the first upgrade)
  - entry window: 3–15 trading days post-report (past the IV-crush chaos, most of the
    ~60-day drift still ahead)
  - "already drifted" guard: skip if the stock has already moved >8% in the drift
    direction since the report (crude proxy for "the move already happened")

FIRST UPGRADE (near-term, not v1): replace %-beat with SUE (surprise / its own
historical stdev) — needs a per-name surprise-history table, hence deferred.
--------------------------------------------------------------------------------

REPO WIRING (both adapters live, verified against real EODHD responses 2026-08-03):
  - earnings: EODHD calendar/earnings, the endpoint the Market Awareness tab uses,
    scoped to the snapshotter's universe. Rows with a null `actual` (reported but
    not yet posted) are dropped rather than scored.
  - coverage: EODHD fundamentals Earnings::Trend, joined on the calendar's `date`
    (fiscal period end) so the count belongs to the REPORTED quarter, not a
    forward estimate.
  - drift: data/universe_prices/{ticker}.csv, the store daily_refresh keeps current.
"""

from __future__ import annotations

import logging
import sys
from collections import Counter
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import requests

# Repo-root anchoring so flat root modules (config, explore_constituents) and
# data/ paths resolve however this module is invoked.
ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# from the satellite schema — candidates become ticket TriggerInfo downstream
from options_ticket import OptionType, TriggerInfo, TriggerType

log = logging.getLogger("options.triggers.pead")

PRICE_DIR = ROOT / "data/universe_prices"

# --------------------------------------------------------------------------- #
# Config (pre-committed, revisable from Bookkeeper data — not sacred numbers)
# --------------------------------------------------------------------------- #

# Provisional bar. Raised 8.0 -> 18.0 after the first live funnel: 8% passed
# 91/224 in-window names, which is the base RATE of beating, not a surprise.
# SUE (surprise / its own historical stdev) is the next upgrade and is now
# near-term rather than deferred — precisely because a flat %-beat does not
# discriminate. Raw magnitude stays logged on every candidate so the Bookkeeper
# can move this from data.
BEAT_THRESHOLD_PCT = 18.0
MIN_EPS_DENOMINATOR = 0.20      # floor |consensus| so tiny bases can't fake huge %
MIN_ANALYSTS = 3                # real coverage; applied only when analyst count is known
ENTRY_WINDOW_MIN_DAYS = 3       # trading days post-report (skip immediate crush/noise)
ENTRY_WINDOW_MAX_DAYS = 15      # still leaves most of the ~60-day drift ahead

# Two-sided drift guard, both measured from the POST-announcement baseline.
# WITH-thesis: the drift we wanted has already happened, little left to capture.
# AGAINST-thesis: the market moved hard the other way, i.e. it repudiated the
# surprise (good news sold off / bad news rallied) — the signal is void, not
# merely spent. The against-bar is deliberately WIDER: small counter-moves are
# ordinary noise, so only a large one should veto.
MAX_DRIFT_WITH_PCT = 8.0
MAX_DRIFT_AGAINST_PCT = 12.0

# EODHD calendar `before_after_market` values. Anything else (including a
# missing value or an intraday "During") is treated as AfterMarket, the
# conservative choice — see drift_baseline_date.
BEFORE_MARKET = "BeforeMarket"
AFTER_MARKET = "AfterMarket"


# --------------------------------------------------------------------------- #
# Candidate
# --------------------------------------------------------------------------- #

@dataclass
class PEADCandidate:
    ticker: str
    direction: OptionType          # CALL on a beat, PUT on a miss
    surprise_pct: float            # signed; + = beat, - = miss
    days_since_report: int         # trading days
    report_date: date
    actual_eps: float
    consensus_eps: float
    num_analysts: Optional[int]
    drift_so_far_pct: float        # signed stock move since report
    fired_at: datetime = field(default_factory=datetime.now)

    def to_trigger_info(self) -> TriggerInfo:
        """Convert to the schema's TriggerInfo so it can populate an OptionsTicket."""
        beat_miss = "beat" if self.surprise_pct >= 0 else "miss"
        cov = f"{self.num_analysts} analysts" if self.num_analysts is not None else "coverage n/a"
        cond = (f"EPS {beat_miss} {self.surprise_pct:+.1f}% "
                f"(actual {self.actual_eps:.2f} vs cons {self.consensus_eps:.2f}, {cov}); "
                f"{self.days_since_report}td post-report; drift {self.drift_so_far_pct:+.1f}% so far "
                f"→ long {self.direction.value}")
        return TriggerInfo(
            trigger=TriggerType.PEAD,
            fired_at=self.fired_at,
            condition=cond,
            metrics={
                "surprise_pct": round(self.surprise_pct, 2),
                "days_since_report": self.days_since_report,
                "report_date": self.report_date.isoformat(),
                "actual_eps": self.actual_eps,
                "consensus_eps": self.consensus_eps,
                "num_analysts": self.num_analysts,
                "drift_so_far_pct": round(self.drift_so_far_pct, 2),
            },
        )


# --------------------------------------------------------------------------- #
# PURE LOGIC — no repo/network dependency; unit-tested below
# --------------------------------------------------------------------------- #

def surprise_pct(actual_eps: float, consensus_eps: float) -> float:
    """Signed surprise as a % of the absolute consensus. + = beat, - = miss.
    abs() in the denominator so a miss vs. a negative consensus keeps the right sign."""
    if consensus_eps == 0:
        return 0.0
    return (actual_eps - consensus_eps) / abs(consensus_eps) * 100.0


def trading_days_since(report_date: date, as_of: date) -> int:
    """Weekday count between report and as_of. NOTE: excludes weekends, NOT holidays —
    a v1 approximation. Wire trading_days_between to a real market calendar to be exact."""
    return int(np.busday_count(report_date, as_of))


def evaluate(actual_eps: float, consensus_eps: float, num_analysts: Optional[int],
             report_date: date, drift_so_far_pct: float, ticker: str,
             as_of: date) -> tuple[Optional[PEADCandidate], str]:
    """Decide if one name is a PEAD candidate. Returns (candidate|None, reason).
    The reason string is for logging WHY something was rejected — useful diagnostics."""
    surp = surprise_pct(actual_eps, consensus_eps)
    direction = OptionType.CALL if surp >= 0 else OptionType.PUT

    # denominator floor — reject tiny-base false positives before anything else
    if abs(consensus_eps) < MIN_EPS_DENOMINATOR:
        return None, f"consensus |{consensus_eps:.2f}| below ${MIN_EPS_DENOMINATOR} floor"

    # real coverage — only enforced when the count is known
    if num_analysts is not None and num_analysts < MIN_ANALYSTS:
        return None, f"only {num_analysts} analysts (< {MIN_ANALYSTS})"

    # surprise bar
    if abs(surp) < BEAT_THRESHOLD_PCT:
        return None, f"surprise {surp:+.1f}% below {BEAT_THRESHOLD_PCT}% bar"

    # entry window
    dsr = trading_days_since(report_date, as_of)
    if dsr < ENTRY_WINDOW_MIN_DAYS:
        return None, f"{dsr}td post-report — too fresh (< {ENTRY_WINDOW_MIN_DAYS})"
    if dsr > ENTRY_WINDOW_MAX_DAYS:
        return None, f"{dsr}td post-report — window closed (> {ENTRY_WINDOW_MAX_DAYS})"

    # Two-sided drift guard, on post-announcement drift (the gap is excluded).
    # WITH-thesis exhaustion: the move we wanted has largely happened already.
    if direction is OptionType.CALL and drift_so_far_pct >= MAX_DRIFT_WITH_PCT:
        return None, (f"drift exhausted: already {drift_so_far_pct:+.1f}% since "
                      f"announcement (>= +{MAX_DRIFT_WITH_PCT}%)")
    if direction is OptionType.PUT and drift_so_far_pct <= -MAX_DRIFT_WITH_PCT:
        return None, (f"drift exhausted: already {drift_so_far_pct:+.1f}% since "
                      f"announcement (<= -{MAX_DRIFT_WITH_PCT}%)")
    # AGAINST-thesis repudiation: the market moved hard the OTHER way, so the
    # surprise was rejected rather than under-reacted to. Buying the drift here
    # is the "bad news rallied / good news sold off" trap.
    if direction is OptionType.CALL and drift_so_far_pct <= -MAX_DRIFT_AGAINST_PCT:
        return None, (f"move against thesis: {drift_so_far_pct:+.1f}% since "
                      f"announcement on a beat (<= -{MAX_DRIFT_AGAINST_PCT}%)")
    if direction is OptionType.PUT and drift_so_far_pct >= MAX_DRIFT_AGAINST_PCT:
        return None, (f"move against thesis: {drift_so_far_pct:+.1f}% since "
                      f"announcement on a miss (>= +{MAX_DRIFT_AGAINST_PCT}%)")

    return PEADCandidate(
        ticker=ticker, direction=direction, surprise_pct=surp, days_since_report=dsr,
        report_date=report_date, actual_eps=actual_eps, consensus_eps=consensus_eps,
        num_analysts=num_analysts, drift_so_far_pct=drift_so_far_pct,
    ), "candidate"


# --------------------------------------------------------------------------- #
# ADAPTERS — wired to this repo
# --------------------------------------------------------------------------- #

_analyst_cache: dict[str, dict[str, int]] = {}


def _analyst_counts(ticker: str) -> dict[str, int]:
    """{fiscal_period_end: analyst_count} from EODHD fundamentals Earnings::Trend.

    Keyed by fiscal-period-end so it joins to the earnings calendar's `date`
    field (which is the period end, NOT the announcement date). Using the
    historical 0q rows means the count belongs to the quarter that was actually
    reported rather than a forward-looking estimate. Failures cache as empty so
    a broken name isn't re-fetched every scan."""
    if ticker in _analyst_cache:
        return _analyst_cache[ticker]
    from config import EODHD_API_KEY
    out: dict[str, int] = {}
    try:
        url = (f"https://eodhd.com/api/fundamentals/{ticker}.US"
               f"?api_token={EODHD_API_KEY}&fmt=json&filter=Earnings::Trend")
        resp = requests.get(url, timeout=30)
        if resp.status_code == 200:
            data = resp.json()
            if isinstance(data, dict):
                for period_end, blk in data.items():
                    if not isinstance(blk, dict):
                        continue
                    n = blk.get("earningsEstimateNumberOfAnalysts")
                    try:
                        if n is not None and float(n) > 0:
                            out[str(period_end)] = int(float(n))
                    except (TypeError, ValueError):
                        continue
    except requests.RequestException:
        pass
    _analyst_cache[ticker] = out
    return out


def get_recent_earnings(lookback_trading_days: int = ENTRY_WINDOW_MAX_DAYS + 2) -> list[dict]:
    """One dict per universe name that reported within the lookback:
    {ticker, report_date: date, actual_eps, consensus_eps, num_analysts,
     report_timing}.

    report_timing is the calendar's raw before_after_market value and drives the
    drift baseline (see drift_baseline_date); it is passed through unnormalized
    so an unexpected value shows up as "defaulted" rather than being silently
    coerced into one of the known cases.

    Sourced from the EODHD earnings calendar, batched 30 symbols per call.
    Rows without a posted `actual` are dropped — EODHD lists a scheduled report
    before the number lands, and scoring a null actual as a miss would fabricate
    candidates. Where a name reported more than once in the window (restatement
    or a stray duplicate), the most recent report_date wins."""
    from options.snapshotter import get_universe
    universe = get_universe()

    end = date.today()
    # trading-day lookback -> calendar span, with slack for holidays
    start = pd.Timestamp(np.busday_offset(end, -lookback_trading_days,
                                          roll="backward")).date() - timedelta(days=4)

    raw: list[dict] = []
    for i in range(0, len(universe), 30):
        chunk = universe[i:i + 30]
        url = (f"https://eodhd.com/api/calendar/earnings?api_token={_key()}"
               f"&symbols={','.join(t + '.US' for t in chunk)}"
               f"&from={start}&to={end}&fmt=json")
        try:
            resp = requests.get(url, timeout=30)
            if resp.status_code == 200:
                raw.extend(resp.json().get("earnings", []))
        except requests.RequestException:
            continue

    best: dict[str, dict] = {}
    for e in raw:
        ticker = str(e.get("code", "")).replace(".US", "").strip().upper()
        actual, est = e.get("actual"), e.get("estimate")
        if not ticker or not isinstance(actual, (int, float)) \
                or not isinstance(est, (int, float)):
            continue                      # scheduled-but-unreported, or no consensus
        try:
            rd = date.fromisoformat(str(e.get("report_date")))
        except (TypeError, ValueError):
            continue
        if ticker not in best or rd > best[ticker]["report_date"]:
            best[ticker] = {
                "ticker": ticker,
                "report_date": rd,
                "actual_eps": float(actual),
                "consensus_eps": float(est),
                # `date` is the fiscal period end — the Earnings::Trend join key
                "num_analysts": _analyst_counts(ticker).get(str(e.get("date"))),
                "report_timing": (str(e["before_after_market"])
                                  if e.get("before_after_market") else None),
            }
    return sorted(best.values(), key=lambda r: r["ticker"])


def _key() -> str:
    from config import EODHD_API_KEY
    return EODHD_API_KEY


def timing_defaulted(report_timing: Optional[str]) -> bool:
    """True when the calendar gave no usable timing, so the baseline had to fall
    back to the conservative AfterMarket rule."""
    return report_timing not in (BEFORE_MARKET, AFTER_MARKET)


def drift_baseline_date(report_date: date, report_timing: Optional[str]) -> date:
    """The first calendar date whose close POSTDATES the announcement.

    This is the correction that matters for PEAD. Drift is the market's slow
    re-rating AFTER the news is public — it is not the announcement gap itself.
    So the baseline must sit on the far side of the announcement:

      BeforeMarket  -> the news is out before the open, so that same day's close
                       already embeds the reaction. Baseline = report-date close.
      AfterMarket   -> the news lands after the close, so the report-date close
                       is a PRE-announcement price. Baseline = next session.
      unknown/other -> treated as AfterMarket. Conservative: it can only skip
                       the gap, never fold a gap into the drift.

    Measuring an AfterMarket report from its report-date close counts the
    overnight reaction as drift, which then trips the exhaustion guard and
    discards exactly the largest surprises PEAD says drift hardest."""
    if report_timing == BEFORE_MARKET:
        return report_date
    return report_date + timedelta(days=1)


def get_drift_since_report(ticker: str, report_date: date,
                           report_timing: Optional[str] = None) -> float:
    """Signed % move from the first post-announcement close to the latest close.

    report_timing is the calendar's before_after_market value; omitting it takes
    the conservative AfterMarket path. Where the computed baseline date is not
    itself a trading day (weekend/holiday), the next available session close is
    used and logged, so the substitution is visible rather than silent."""
    path = PRICE_DIR / f"{ticker}.csv"
    if not path.exists():
        raise FileNotFoundError(f"no price history for {ticker}")
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    if "Close" not in df.columns:
        raise ValueError(f"{ticker}: price file has no Close column")
    closes = df["Close"].dropna().sort_index()
    if closes.empty:
        raise ValueError(f"{ticker}: no closes")

    want = drift_baseline_date(report_date, report_timing)
    on_or_after = closes.loc[pd.Timestamp(want):]
    if len(on_or_after) < 1:
        raise ValueError(f"{ticker}: no closes on/after baseline {want}")
    baseline_date = on_or_after.index[0]
    if baseline_date.date() != want:
        log.debug("%s: no session on %s, using %s as drift baseline",
                  ticker, want, baseline_date.date())
    baseline = float(on_or_after.iloc[0])
    latest = float(closes.iloc[-1])
    if baseline <= 0:
        raise ValueError(f"{ticker}: non-positive baseline close")
    return (latest / baseline - 1.0) * 100.0


# --------------------------------------------------------------------------- #
# ORCHESTRATION
# --------------------------------------------------------------------------- #

def scan(as_of: Optional[date] = None) -> list[PEADCandidate]:
    """Run PEAD over recent earnings; return surviving candidates. Logs the reject reason
    for every name that doesn't make it, so the funnel is auditable."""
    as_of = as_of or date.today()
    rows = get_recent_earnings()
    n_defaulted = sum(1 for r in rows if timing_defaulted(r.get("report_timing")))
    if n_defaulted:
        log.info("PEAD: %d/%d rows had no usable report timing — drift baseline "
                 "defaulted to the AfterMarket rule", n_defaulted, len(rows))
    candidates, rejected = [], 0
    for r in rows:
        try:
            drift = get_drift_since_report(r["ticker"], r["report_date"],
                                           r.get("report_timing"))
            cand, reason = evaluate(
                actual_eps=r["actual_eps"], consensus_eps=r["consensus_eps"],
                num_analysts=r.get("num_analysts"), report_date=r["report_date"],
                drift_so_far_pct=drift, ticker=r["ticker"], as_of=as_of,
            )
            if cand is not None:
                candidates.append(cand)
                log.info("PEAD candidate: %s %s", r["ticker"], cand.to_trigger_info().condition)
            else:
                rejected += 1
                log.debug("PEAD reject %s: %s", r["ticker"], reason)
        except Exception as e:  # noqa: BLE001 — one bad name must not stop the scan
            rejected += 1
            log.warning("PEAD %s: evaluation failed (%s)", r.get("ticker"), e)
    log.info("PEAD scan %s: %d candidates, %d rejected of %d earnings rows",
             as_of, len(candidates), rejected, len(rows))
    return candidates
