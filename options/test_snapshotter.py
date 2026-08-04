"""Unit tests for options/snapshotter.py pure logic.

Pure functions only — no network, no Schwab token, no API keys, no reads of
data/. The adapters (fetch_chain, get_universe, get_price_history, ...) are
exercised by an actual scan, not here.

Several of these guard fixes to an UNBACKFILLABLE artifact: data/options/
iv_history.csv accrues one row per name per session and can never be
regenerated, so a defect that silently corrupts a reading is permanent. Those
tests are marked CORRUPTION GUARD below — treat a failure there as data loss,
not merely a red test.

Run:  python options/test_snapshotter.py
"""
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from options.snapshotter import (Contract, realized_vol, select_atm_iv30,
                                 build_watchlist, _dedup_append,
                                 session_date_for, interpolate_iv30,
                                 atm_iv_at_each_expiry,
                                 atm_contract_at_each_expiry,
                                 constant_maturity_iv30)

fails = []


def check(name, cond):
    print(("PASS " if cond else "FAIL ") + name)
    if not cond:
        fails.append(name)


def mk(dte, strike, iv=0.30, oi=100):
    return Contract(expiry=date.today() + timedelta(days=dte), dte=dte,
                    strike=strike, iv=iv, delta=0.5, gamma=0.01, theta=-0.05,
                    vega=0.10, bid=1.0, ask=1.2, open_interest=oi)


# --------------------------------------------------------------------------- #
# realized_vol
# --------------------------------------------------------------------------- #
# Deterministic series: alternating +/-2% log returns, so the annualization is
# checkable to the digit rather than to a noisy range.
logrets = np.array([0.02 if i % 2 == 0 else -0.02 for i in range(200)])
closes = list(100 * np.exp(np.cumsum(logrets)))
rv = realized_vol(closes)
expected = float(np.std(logrets[-30:], ddof=1) * np.sqrt(252))
check("rv: matches direct computation on last 30 returns",
      rv is not None and abs(rv - expected) < 1e-12)
check("rv: annualizes to ~0.32 for 2%/day", 0.30 < rv < 0.34)
check("rv: too-short history -> None", realized_vol(closes[:20]) is None)
check("rv: None input -> None", realized_vol(None) is None)

# --------------------------------------------------------------------------- #
# select_atm_iv30
# --------------------------------------------------------------------------- #
chain = ([mk(7, s) for s in (140, 145, 150)]        # front week — must be excluded
         + [mk(28, s) for s in (140, 145, 150, 155)]
         + [mk(56, s) for s in (140, 145, 150)])
pick = select_atm_iv30(147.0, chain)
check("atm: picks 28-DTE expiry", pick.dte == 28)
check("atm: picks 145 strike nearest spot 147", pick.strike == 145)
check("atm: empty chain -> None", select_atm_iv30(147.0, []) is None)
check("atm: only front week -> None", select_atm_iv30(147.0, [mk(7, 145)]) is None)
tie = [mk(20, 100), mk(40, 100)]     # both 10 from target 30 -> longer wins
check("atm: DTE tie -> longer expiry", select_atm_iv30(100.0, tie).dte == 40)

# --------------------------------------------------------------------------- #
# build_watchlist
# --------------------------------------------------------------------------- #
universe = [f"T{i:03d}" for i in range(100)]
scores = {t: 100.0 - i for i, t in enumerate(universe)}   # T000 best ... T099 worst
earnings = {"T020": date.today() + timedelta(days=10),    # rank 21 — top third
            "T050": date.today() + timedelta(days=10),    # rank 51 — NOT top third
            "T005": date.today() + timedelta(days=10)}    # rank 6 — already top decile
wl = build_watchlist(universe, scores, earnings)
check("wl: top decile present", all(f"T{i:03d}" in wl for i in range(10)))
check("wl: catalyst in top third included", "T020" in wl)
check("wl: catalyst below top third excluded", "T050" not in wl)
check("wl: union has no dupes", len(wl) == len(set(wl)))
check("wl: exactly decile + T020",
      sorted(wl) == sorted([f"T{i:03d}" for i in range(10)] + ["T020"]))
check("wl: unscored universe names ignored",
      "ZZZZ" not in build_watchlist(universe + ["ZZZZ"], scores, {}))

# --------------------------------------------------------------------------- #
# _dedup_append  — CORRUPTION GUARD: a same-day rerun must overwrite, never
# duplicate, or the history gains phantom observations.
# --------------------------------------------------------------------------- #
existing = pd.DataFrame([
    {"date": "2026-08-01", "ticker": "AAPL", "atm_iv": 0.28},
    {"date": "2026-08-02", "ticker": "AAPL", "atm_iv": 0.30},
    {"date": "2026-08-02", "ticker": "MSFT", "atm_iv": 0.25},
])
new = pd.DataFrame([{"date": "2026-08-02", "ticker": "AAPL", "atm_iv": 0.35}])
out = _dedup_append(existing, new)
check("dedup: same-day rerun overwrites 0.30 -> 0.35",
      out[(out["date"] == "2026-08-02") & (out["ticker"] == "AAPL")]["atm_iv"].tolist() == [0.35])
check("dedup: prior day preserved",
      out[(out["date"] == "2026-08-01") & (out["ticker"] == "AAPL")]["atm_iv"].tolist() == [0.28])
check("dedup: other ticker same day preserved",
      out[(out["date"] == "2026-08-02") & (out["ticker"] == "MSFT")]["atm_iv"].tolist() == [0.25])
check("dedup: row count 3 (one replaced, none duplicated)", len(out) == 3)
check("dedup: empty existing -> new returned", _dedup_append(pd.DataFrame(), new).equals(new))

# --------------------------------------------------------------------------- #
# session_date_for — CORRUPTION GUARD: Schwab keeps serving the last session's
# quotes when the market is shut, so an unattended weekend run would otherwise
# stamp Friday's close with Saturday's and Sunday's dates: three identical
# readings biasing every future IV-rank percentile.
# --------------------------------------------------------------------------- #
friday = datetime(2026, 7, 31, 15, 0, 0)
saturday = date(2026, 8, 1)
c_fri = mk(30, 100)
c_fri.quote_time = friday
check("session: weekend run stamps Friday's session, not Saturday",
      session_date_for(c_fri, saturday) == date(2026, 7, 31))
c_live = mk(30, 100)
c_live.quote_time = datetime(2026, 8, 3, 14, 59, 59)
check("session: same-day quote stamps today",
      session_date_for(c_live, date(2026, 8, 3)) == date(2026, 8, 3))
check("session: missing quote_time falls back to as_of",
      session_date_for(mk(30, 100), saturday) == saturday)
wk = pd.DataFrame([{"date": "2026-07-31", "ticker": "AAPL", "atm_iv": 0.28}])
after_sat = _dedup_append(wk, pd.DataFrame(
    [{"date": "2026-07-31", "ticker": "AAPL", "atm_iv": 0.28}]))
check("session: weekend rerun is an idempotent rewrite, not a new row",
      len(after_sat) == 1)

# --------------------------------------------------------------------------- #
# interpolate_iv30 — CORRUPTION GUARD: monthly-only names sawtooth through
# ~18->39->32->25 DTE as expiries roll, so a single-contract reading moves for
# calendar reasons rather than volatility reasons. Interpolation must be in
# TOTAL VARIANCE (sigma^2 * t), which is additive in time; interpolating raw
# vol is subtly wrong and would bias every reading.
# --------------------------------------------------------------------------- #
flat = [mk(18, 100, iv=0.40), mk(46, 100, iv=0.40)]
check("interp: flat term structure -> unchanged vol",
      abs(interpolate_iv30(100.0, flat) - 0.40) < 1e-12)

# hand-computed: v1=.09*18=1.62 ; v2=.16*46=7.36 ; w=(30-18)/(46-18)=12/28
# v30 = 1.62 + (7.36-1.62)*12/28 = 4.08 ; sigma30 = sqrt(4.08/30) = 0.36878...
sloped = [mk(18, 100, iv=0.30), mk(46, 100, iv=0.40)]
expected30 = ((0.09 * 18 + (0.16 * 46 - 0.09 * 18) * (30 - 18) / (46 - 18)) / 30) ** 0.5
got30 = interpolate_iv30(100.0, sloped)
check("interp: matches hand-computed variance interpolation",
      abs(got30 - expected30) < 1e-12)
check("interp: result lies between the two bracketing vols", 0.30 < got30 < 0.40)
check("interp: variance-space differs from naive vol-average (the subtle bug)",
      abs(got30 - (0.30 + (0.40 - 0.30) * 12 / 28)) > 1e-4)

check("interp: exact 30-DTE expiry returns its own vol",
      abs(interpolate_iv30(100.0, [mk(30, 100, iv=0.33), mk(60, 100, iv=0.45)]) - 0.33) < 1e-12)
check("interp: all expiries above target -> None (never extrapolate)",
      interpolate_iv30(100.0, [mk(40, 100), mk(60, 100)]) is None)
check("interp: all expiries below target -> None (never extrapolate)",
      interpolate_iv30(100.0, [mk(12, 100), mk(20, 100)]) is None)
check("interp: single expiry -> None", interpolate_iv30(100.0, [mk(30, 100)]) is None)
check("interp: empty chain -> None", interpolate_iv30(100.0, []) is None)

multi = [mk(18, 90, iv=0.55), mk(18, 101, iv=0.31),
         mk(46, 90, iv=0.60), mk(46, 99, iv=0.36)]
check("interp: per-expiry ATM is strike nearest spot",
      atm_iv_at_each_expiry(100.0, multi) == [(18, 0.31), (46, 0.36)])

# the defect this exists to fix: the tenor roll must not move the reading
sept = [mk(18, 100, iv=0.30), mk(46, 100, iv=0.40)]
next_wk = [mk(11, 100, iv=0.28), mk(39, 100, iv=0.38)]   # same curve, rolled forward
a, b = interpolate_iv30(100.0, sept), interpolate_iv30(100.0, next_wk)
raw_a = select_atm_iv30(100.0, sept).iv
raw_b = select_atm_iv30(100.0, next_wk).iv
check("interp: single-contract reading jumps on the roll (the defect)",
      abs(raw_a - raw_b) > 0.05)
check("interp: constant-maturity reading is far steadier across the roll",
      abs(a - b) < abs(raw_a - raw_b) / 2)

# --------------------------------------------------------------------------- #
# constant_maturity_iv30 — clamp + method flag. CORRUPTION GUARD: a monthly-only
# name has no bracket around 30 DTE on ~34% of trading days. Clamping to the
# nearest REAL quote (never extrapolating an unbounded variance line) plus an
# honest method flag is what lets downstream drop or down-weight those rows.
# --------------------------------------------------------------------------- #
br = [mk(18, 100, iv=0.30), mk(46, 100, iv=0.40)]
v, m, oi = constant_maturity_iv30(100.0, br)
check("clamp: bracketed still interpolates", m == "interpolated")
check("clamp: bracketed value identical to interpolate_iv30",
      abs(v - interpolate_iv30(100.0, br)) < 1e-12)

v, m, oi = constant_maturity_iv30(100.0, [mk(37, 100, iv=0.42)])
check("clamp: single far expiry -> clamped_high", m == "clamped_high")
check("clamp: value is that expiry's real vol, not an extrapolation",
      abs(v - 0.42) < 1e-12)

v, m, oi = constant_maturity_iv30(100.0, [mk(32, 100, iv=0.35), mk(60, 100, iv=0.45)])
check("clamp: all-above picks NEAREST tenor (32 not 60)", abs(v - 0.35) < 1e-12)
check("clamp: all-above flagged clamped_high", m == "clamped_high")

v, m, oi = constant_maturity_iv30(100.0, [mk(18, 100, iv=0.50), mk(25, 100, iv=0.46)])
check("clamp: all-below picks nearest (25)", abs(v - 0.46) < 1e-12)
check("clamp: all-below flagged clamped_low", m == "clamped_low")

v, m, oi = constant_maturity_iv30(100.0, [])
check("clamp: empty chain -> (None, None, None)",
      v is None and m is None and oi is None)
v, m, oi = constant_maturity_iv30(100.0, [mk(5, 100)])
check("clamp: all contracts under min_dte -> (None, None, None)",
      v is None and m is None and oi is None)

steep = [mk(39, 100, iv=0.20), mk(46, 100, iv=0.60)]
v, m, oi = constant_maturity_iv30(100.0, steep)
check("clamp: steep curve does not extrapolate below the front vol",
      abs(v - 0.20) < 1e-12 and m == "clamped_high")
v, m, oi = constant_maturity_iv30(100.0, [mk(37, 80, iv=0.90), mk(37, 101, iv=0.33)])
check("clamp: clamp path uses strike nearest spot", abs(v - 0.33) < 1e-12)

# --------------------------------------------------------------------------- #
# iv30_min_oi — CORRUPTION GUARD: the row's `open_interest` describes the
# select_atm_iv30 contract behind atm_iv, a DIFFERENT contract from the pair
# feeding iv30_interp. Down-weighting the interpolated series by the wrong
# quote's OI would hide model marks. (Live AIZ: atm OI 4, but a bracketing leg
# was 0 — the interpolation rested partly on a pure model price.)
# --------------------------------------------------------------------------- #
v, m, oi = constant_maturity_iv30(
    100.0, [mk(18, 100, iv=0.30, oi=4), mk(46, 100, iv=0.40, oi=900)])
check("min_oi: interpolated pair OI 4 and 900 records 4", oi == 4)
check("min_oi: that row is flagged interpolated", m == "interpolated")
v, m, oi = constant_maturity_iv30(
    100.0, [mk(18, 100, iv=0.30, oi=900), mk(46, 100, iv=0.40, oi=4)])
check("min_oi: thin FAR leg also records 4", oi == 4)

v, m, oi = constant_maturity_iv30(100.0, [mk(37, 100, iv=0.42, oi=57)])
check("min_oi: clamped row records the clamped contract's OI", oi == 57)
v, m, oi = constant_maturity_iv30(
    100.0, [mk(32, 100, iv=0.35, oi=11), mk(60, 100, iv=0.45, oi=999)])
check("min_oi: clamped row uses the tenor it clamped to, not the far one", oi == 11)

mixed = [mk(18, 100, iv=0.30, oi=50), mk(46, 100, iv=0.40, oi=60),
         mk(11, 100, iv=0.28, oi=1)]      # thin 11-DTE leg is NOT a bracket input
v, m, oi = constant_maturity_iv30(100.0, mixed)
check("min_oi: ignores non-bracketing expiries (thin 11-DTE leg excluded)", oi == 50)

v, m, oi = constant_maturity_iv30(
    100.0, [mk(18, 100, iv=0.30, oi=0), mk(46, 100, iv=0.40, oi=900)])
check("min_oi: zero OI recorded as 0, not dropped or coerced", oi == 0)

v, m, oi = constant_maturity_iv30(
    100.0, [mk(18, 70, iv=0.9, oi=5), mk(18, 101, iv=0.30, oi=300),
            mk(46, 100, iv=0.40, oi=900)])
check("min_oi: uses the ATM strike's OI (300), not a far strike's (5)", oi == 300)

# the refactor that introduced atm_contract_at_each_expiry must not have changed
# the tested helper it now delegates from
check("refactor: atm_iv_at_each_expiry output unchanged",
      atm_iv_at_each_expiry(100.0, multi) == [(18, 0.31), (46, 0.36)])
cpts = atm_contract_at_each_expiry(100.0, [mk(18, 90, iv=0.55), mk(18, 101, iv=0.31)])
check("refactor: contract variant returns the same ATM pick",
      len(cpts) == 1 and cpts[0][0] == 18 and cpts[0][1].strike == 101)

print(f"\n{len(fails)} FAILURES" if fails else "\nALL PASS (0 failures)")
sys.exit(1 if fails else 0)
