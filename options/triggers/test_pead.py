"""Unit tests for the PEAD trigger's pure logic.

Pure functions only — no network, no repo data, no API keys. The two adapters
(get_recent_earnings, get_drift_since_report) are exercised by the live scan,
not here.

Run:  python options/triggers/test_pead.py
"""
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import options_ticket
from options_ticket import OptionType, TriggerInfo, TriggerType
from options.triggers import pead

fails = []


def check(name, cond):
    print(("PASS " if cond else "FAIL ") + name)
    if not cond:
        fails.append(name)


def ev(direction, drift):
    """An evaluate() call that clears every filter except the drift guard, so
    the guard is the only thing under test. +50% beat / -50% miss."""
    actual, cons = (3.0, 2.0) if direction == "beat" else (1.0, 2.0)
    return pead.evaluate(actual_eps=actual, consensus_eps=cons, num_analysts=10,
                         report_date=date(2026, 7, 24), drift_so_far_pct=drift,
                         ticker="TEST", as_of=date(2026, 8, 3))


# --------------------------------------------------------------------------- #
# schema wiring
# --------------------------------------------------------------------------- #
check("schema: loads from the repo root, not a stand-in",
      options_ticket.__file__.startswith(str(ROOT)))
check("schema: TriggerType has exactly the three live triggers",
      {m.value for m in TriggerType} == {"pead", "insider_cluster", "iv_catalyst"})
check("schema: Grade is A/B/C/pass (no A+)",
      {m.value for m in options_ticket.Grade} == {"A", "B", "C", "pass"})

# --------------------------------------------------------------------------- #
# surprise_pct — sign and denominator handling (verified against real prints:
# CNC +182.0%, TRV +89.1%, BA -123.5% on a NEGATIVE consensus, META -13.0%)
# --------------------------------------------------------------------------- #
check("surprise: beat is positive",
      abs(pead.surprise_pct(3.0, 2.0) - 50.0) < 1e-9)
check("surprise: miss is negative",
      abs(pead.surprise_pct(1.0, 2.0) + 50.0) < 1e-9)
check("surprise: BA case — wider loss vs negative consensus stays negative",
      abs(pead.surprise_pct(-0.76, -0.34) + 123.5294117647) < 1e-6)
check("surprise: smaller loss than a negative consensus is a BEAT",
      pead.surprise_pct(-0.10, -0.34) > 0)
check("surprise: zero consensus returns 0 rather than dividing by zero",
      pead.surprise_pct(1.0, 0.0) == 0.0)

# --------------------------------------------------------------------------- #
# drift baseline — the AfterMarket correction
# --------------------------------------------------------------------------- #
RD = date(2026, 7, 28)
check("baseline: BeforeMarket uses the report-date close",
      pead.drift_baseline_date(RD, pead.BEFORE_MARKET) == RD)
check("baseline: AfterMarket uses the NEXT day",
      pead.drift_baseline_date(RD, pead.AFTER_MARKET) == date(2026, 7, 29))
check("baseline: unknown timing defaults to next day (conservative)",
      pead.drift_baseline_date(RD, None) == date(2026, 7, 29))
check("baseline: an unexpected value also defaults to next day",
      pead.drift_baseline_date(RD, "During") == date(2026, 7, 29))

check("defaulted: None is flagged", pead.timing_defaulted(None) is True)
check("defaulted: 'During' is flagged", pead.timing_defaulted("During") is True)
check("defaulted: BeforeMarket is not flagged",
      pead.timing_defaulted(pead.BEFORE_MARKET) is False)
check("defaulted: AfterMarket is not flagged",
      pead.timing_defaulted(pead.AFTER_MARKET) is False)

# --------------------------------------------------------------------------- #
# two-sided drift guard
# --------------------------------------------------------------------------- #
c, _ = ev("beat", 2.0)
check("guard: beat with mild drift survives",
      c is not None and c.direction is OptionType.CALL)
c, _ = ev("miss", -2.0)
check("guard: miss with mild drift survives",
      c is not None and c.direction is OptionType.PUT)

c, r = ev("beat", 8.5)
check("guard: beat +8.5% rejected as exhausted", c is None and "drift exhausted" in r)
c, r = ev("miss", -8.5)
check("guard: miss -8.5% rejected as exhausted", c is None and "drift exhausted" in r)
c, _ = ev("beat", 7.9)
check("guard: beat +7.9% survives (just under the with-bar)", c is not None)

# the NOW case: a miss whose stock ripped upward — market repudiated the signal
c, r = ev("miss", 19.6)
check("guard: miss +19.6% rejected as move-against-thesis (the NOW case)",
      c is None and "move against thesis" in r)
c, r = ev("beat", -15.0)
check("guard: beat -15% rejected as move-against-thesis",
      c is None and "move against thesis" in r)

# the against-bar is deliberately WIDER than the with-bar: a counter-move
# between the two thresholds is ordinary noise and must NOT veto
check("guard: against-bar is wider than with-bar",
      pead.MAX_DRIFT_AGAINST_PCT > pead.MAX_DRIFT_WITH_PCT)
c, _ = ev("miss", 9.0)
check("guard: miss +9% (between the bars) survives — counter-noise tolerated",
      c is not None)
c, _ = ev("beat", -9.0)
check("guard: beat -9% (between the bars) survives", c is not None)
c, r = ev("miss", 12.5)
check("guard: miss +12.5% rejected (just over the against-bar)",
      c is None and "move against thesis" in r)

_, r_exh = ev("beat", 9.0)
_, r_agn = ev("beat", -13.0)
check("guard: the two reject reasons are distinguishable in a tally",
      r_exh != r_agn and "exhausted" in r_exh and "against" in r_agn)

# --------------------------------------------------------------------------- #
# remaining filters
# --------------------------------------------------------------------------- #
check("threshold: bar is the provisional 18%", pead.BEAT_THRESHOLD_PCT == 18.0)
c, r = pead.evaluate(actual_eps=2.30, consensus_eps=2.00, num_analysts=10,
                     report_date=date(2026, 7, 24), drift_so_far_pct=1.0,
                     ticker="T", as_of=date(2026, 8, 3))            # +15%
check("threshold: +15% rejected by the 18% bar", c is None and "bar" in r)
c, _ = pead.evaluate(actual_eps=2.40, consensus_eps=2.00, num_analysts=10,
                     report_date=date(2026, 7, 24), drift_so_far_pct=1.0,
                     ticker="T", as_of=date(2026, 8, 3))            # +20%
check("threshold: +20% passes", c is not None)

# denominator floor — the COIN case: a huge % on a tiny base is an artifact
c, r = pead.evaluate(actual_eps=-0.39, consensus_eps=0.14, num_analysts=10,
                     report_date=date(2026, 7, 24), drift_so_far_pct=0.2,
                     ticker="COIN", as_of=date(2026, 8, 3))
check("floor: -378% on a $0.14 base rejected by the denominator floor (COIN)",
      c is None and "floor" in r)

# coverage is enforced only when the count is known
c, r = pead.evaluate(actual_eps=3.0, consensus_eps=2.0, num_analysts=2,
                     report_date=date(2026, 7, 24), drift_so_far_pct=1.0,
                     ticker="T", as_of=date(2026, 8, 3))
check("coverage: 2 analysts rejected", c is None and "analysts" in r)
c, _ = pead.evaluate(actual_eps=3.0, consensus_eps=2.0, num_analysts=None,
                     report_date=date(2026, 7, 24), drift_so_far_pct=1.0,
                     ticker="T", as_of=date(2026, 8, 3))
check("coverage: unknown count does NOT reject (it is flagged, not fatal)",
      c is not None)

# entry window
c, r = pead.evaluate(actual_eps=3.0, consensus_eps=2.0, num_analysts=10,
                     report_date=date(2026, 8, 1), drift_so_far_pct=1.0,
                     ticker="T", as_of=date(2026, 8, 3))
check("window: 0td post-report rejected as too fresh", c is None and "too fresh" in r)
c, r = pead.evaluate(actual_eps=3.0, consensus_eps=2.0, num_analysts=10,
                     report_date=date(2026, 6, 1), drift_so_far_pct=1.0,
                     ticker="T", as_of=date(2026, 8, 3))
check("window: long-past report rejected as window closed",
      c is None and "window closed" in r)

# --------------------------------------------------------------------------- #
# TriggerInfo round-trips through real pydantic validation
# --------------------------------------------------------------------------- #
cand, _ = ev("beat", 2.0)
ti = cand.to_trigger_info()
check("ticket: trigger is the TriggerType enum member, not a bare string",
      ti.trigger is TriggerType.PEAD)
check("ticket: metrics carry the raw magnitude for later tuning",
      ti.metrics["surprise_pct"] == 50.0)
check("ticket: model_dump -> reconstruct round-trips",
      TriggerInfo(**ti.model_dump()).trigger is TriggerType.PEAD)
check("ticket: json round-trips",
      TriggerInfo.model_validate_json(ti.model_dump_json()).condition == ti.condition)

cand_none, _ = pead.evaluate(actual_eps=3.0, consensus_eps=2.0, num_analysts=None,
                             report_date=date(2026, 7, 24), drift_so_far_pct=2.0,
                             ticker="T", as_of=date(2026, 8, 3))
ti_none = cand_none.to_trigger_info()
check("ticket: unknown coverage validates as None and reads 'coverage n/a'",
      ti_none.metrics["num_analysts"] is None and "coverage n/a" in ti_none.condition)

cand_put, _ = ev("miss", -2.0)
check("ticket: a miss renders as a long put",
      cand_put.to_trigger_info().condition.endswith("long put"))

print(f"\n{len(fails)} FAILURES" if fails else f"\nALL PASS ({len(fails)} failures)")
sys.exit(1 if fails else 0)
