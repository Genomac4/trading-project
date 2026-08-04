"""
options_ticket.py — Data-block schema for the options satellite.

The single object every ticket carries end-to-end. Each pipeline stage reads
and writes its own slice, leaving downstream slices as None until they run:

    trigger scan → data assembly → gauntlet → Analyst agent → Skeptic agent
    → human decision → Bookkeeper(entry) → Bookkeeper(resolution)

Design rules baked in (from the master doc):
- Fields populate by stage. Anything downstream of the current stage is Optional/None.
- Every quantitative claim in a memo must trace to a DataBlock field or be flagged
  ASSUMPTION. TracedClaim is where that lives; the validator enforces it.
- P&L attribution (delta/vega/theta/spread) needs Greeks + IV at BOTH entry and
  exit, so both snapshots are captured.
- IV context that isn't real snapshot history is labelled PROXY, everywhere.
- This is a RECORD, not an execution path. Nothing here can place an order.

Pydantic v2.
"""

from __future__ import annotations

from datetime import date, datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, computed_field


# --------------------------------------------------------------------------- #
# Enums
# --------------------------------------------------------------------------- #

class OptionType(str, Enum):
    CALL = "call"
    PUT = "put"


class TriggerType(str, Enum):
    PEAD = "pead"                       # earnings surprise, drift unclosed (trades AFTER print)
    INSIDER_CLUSTER = "insider_cluster" # >=2 opportunistic buyers / 30d
    IV_CATALYST = "iv_catalyst"         # IV-rank low + dated catalyst; F-score inflection = optional edge input
    # (old CATALYST_FSCORE killed 2026-07-22: pre-earnings long premium = hold through the print =
    #  hard fail #6's retail donation. Its only value, surprise-direction hint, folds into IV_CATALYST.)


class IVSource(str, Enum):
    PROXY_REALIZED_VOL = "proxy_realized_vol"  # crude proxy until snapshots mature
    SNAPSHOT_HISTORY = "snapshot_history"      # real IV history from the snapshotter


class Grade(str, Enum):
    A = "A"        # 85+ AND no broken legs
    B = "B"        # 70-84
    C = "C"        # 55-69
    PASS = "pass"  # < 55, or a broken leg that drops it below A — not traded


class TicketStatus(str, Enum):
    CANDIDATE = "candidate"                 # trigger fired
    REJECTED_GAUNTLET = "rejected_gauntlet" # died at a hard fail
    MEMO_WRITTEN = "memo_written"           # Analyst done
    SKEPTIC_REVIEWED = "skeptic_reviewed"   # Skeptic done
    QUEUED = "queued"                       # awaiting human verdict
    PASSED = "passed"                       # human declined
    ENTERED = "entered"                     # paper position opened
    OPEN = "open"                           # live paper position
    CLOSED = "closed"                       # resolved


# --------------------------------------------------------------------------- #
# Shared building blocks
# --------------------------------------------------------------------------- #

class Greeks(BaseModel):
    """A Greeks + IV snapshot at a point in time. Reused at candidate, entry, exit."""
    snapshot_at: datetime
    delta: float
    gamma: Optional[float] = None
    theta: float
    vega: float
    rho: Optional[float] = None
    iv: float = Field(description="Implied vol at this snapshot (BSM output from Schwab)")


class TracedClaim(BaseModel):
    """One quantitative claim in a memo and where it came from. The traceability
    validator rejects any memo containing a number with no TracedClaim."""
    statement: str
    value: Optional[float] = None
    source_field: str = Field(
        description="Dot-path into DataBlock (e.g. 'chain.open_interest'), or 'ASSUMPTION'"
    )


# --------------------------------------------------------------------------- #
# Stage 1 — trigger scan
# --------------------------------------------------------------------------- #

class Identity(BaseModel):
    ticker: str
    company_name: Optional[str] = None
    underlying_price: float
    occ_symbol: str = Field(description="Full OCC contract symbol, e.g. NVDA260116C00150000")
    option_type: OptionType
    strike: float
    expiration: date
    dte: int = Field(description="Calendar days to expiry at candidate time")


class TriggerInfo(BaseModel):
    trigger: TriggerType
    fired_at: datetime
    condition: str = Field(description="Human-readable measured condition that fired")
    metrics: dict = Field(
        default_factory=dict,
        description="Raw numbers behind the fire, e.g. {'surprise_pct': 8.2, 'days_since': 3}",
    )


# --------------------------------------------------------------------------- #
# Stage 2 — data assembly (EODHD factors + Schwab chain) → the block agents read
# --------------------------------------------------------------------------- #

class ChainQuote(BaseModel):
    bid: float
    ask: float
    mid: float
    last: Optional[float] = None
    open_interest: int
    volume: int

    @computed_field
    @property
    def spread_pct_of_mid(self) -> float:
        return (self.ask - self.bid) / self.mid if self.mid else float("inf")


class FactorScores(BaseModel):
    """Cross-referenced from the core screener at candidate time."""
    composite_score: Optional[float] = None
    composite_rank: Optional[int] = None
    f_score: Optional[int] = None
    momentum_score: Optional[float] = None
    above_200dma: Optional[bool] = None
    in_sp500: bool = False
    in_sp400: bool = False
    market_cap: Optional[float] = None
    sector: Optional[str] = None


class VolContext(BaseModel):
    iv_current: float
    iv_rank: Optional[float] = Field(default=None, description="0-100; None until computable")
    iv_rank_source: IVSource = IVSource.PROXY_REALIZED_VOL
    realized_vol: Optional[float] = Field(default=None, description="Proxy input, annualized")
    post_event_iv_norm: Optional[float] = Field(
        default=None, description="Expected IV after the catalyst (for crush modeling)"
    )
    post_event_iv_source: IVSource = IVSource.PROXY_REALIZED_VOL


class CatalystInfo(BaseModel):
    catalyst_type: str = Field(description="earnings / product / regulatory / macro / ...")
    catalyst_date: Optional[date] = None
    days_to_catalyst: Optional[int] = None
    next_earnings_date: Optional[date] = None
    earnings_in_holding_window: bool = False


class DataBlock(BaseModel):
    """Everything the Analyst and Skeptic are allowed to reason from. If a number
    isn't here, it's an ASSUMPTION and must be flagged as one."""
    assembled_at: datetime
    factors: FactorScores
    chain: ChainQuote
    greeks: Greeks
    vol: VolContext
    catalyst: CatalystInfo
    implied_move: Optional[float] = Field(default=None, description="From ATM straddle, %")
    breakeven_move_needed: Optional[float] = Field(default=None, description="(strike+debit) vs spot, %")
    # NOTE: no generic edge_signals dict. Edge inputs are trigger-specific and typed on the
    # trigger that owns them, so a signal can't leak onto tickets where it means nothing.
    # (F-score inflection attaches only to IV_CATALYST tickets as an optional edge input
    #  (per TriggerType), never a universal signal.)
    fundamentals: dict = Field(
        default_factory=dict,
        description="Extensible catch-all for EODHD fundamentals the agent may cite",
    )


# --------------------------------------------------------------------------- #
# Stage 3 — hard-fail gauntlet (code, deterministic)
# --------------------------------------------------------------------------- #

class HardFailCheck(BaseModel):
    name: str
    passed: bool
    value: Optional[float] = None
    threshold: Optional[float] = None
    note: Optional[str] = None


class GauntletResult(BaseModel):
    ran_at: datetime
    passed: bool = Field(description="True only if every check passed")
    checks: list[HardFailCheck] = Field(default_factory=list)

    @computed_field
    @property
    def rejections(self) -> list[str]:
        return [c.name for c in self.checks if not c.passed]


# --------------------------------------------------------------------------- #
# Stage 4 — Analyst agent
# --------------------------------------------------------------------------- #

class Memo(BaseModel):
    written_at: datetime
    thesis: str
    mechanism: str = Field(description="Why the price moves — the causal story")
    catalyst_desc: str
    structure_justification: str = Field(description="Why this contract / DTE / delta")
    price_target: float
    target_rationale: str = Field(description="Derived from stated multiples")
    pnl_at_target_post_crush: Optional[float] = Field(
        default=None, description="Required for earnings tickets: P&L at target using POST-event IV"
    )
    risks: list[str] = Field(default_factory=list)
    claims: list[TracedClaim] = Field(default_factory=list)
    web_sources: list[str] = Field(default_factory=list, description="Cited catalyst research URLs")


# --------------------------------------------------------------------------- #
# Stage 5 — Skeptic agent + grading (rubric Part 2)
# --------------------------------------------------------------------------- #

class GradeScores(BaseModel):
    edge: int = Field(ge=0, le=25)
    catalyst_quality: int = Field(ge=0, le=20)
    volatility_pricing: int = Field(ge=0, le=15)
    risk_reward: int = Field(ge=0, le=15)
    timing: int = Field(ge=0, le=15)
    liquidity_execution: int = Field(ge=0, le=10)

    @computed_field
    @property
    def total(self) -> int:
        return (self.edge + self.catalyst_quality + self.volatility_pricing
                + self.risk_reward + self.timing + self.liquidity_execution)

    @computed_field
    @property
    def no_broken_legs(self) -> bool:
        """True if no category scores below 60% of its max."""
        pairs = [(self.edge, 25), (self.catalyst_quality, 20), (self.volatility_pricing, 15),
                 (self.risk_reward, 15), (self.timing, 15), (self.liquidity_execution, 10)]
        return all(score >= 0.6 * mx for score, mx in pairs)

    @computed_field
    @property
    def grade(self) -> Grade:
        t = self.total
        if t >= 85 and self.no_broken_legs:
            return Grade.A
        if t >= 70:
            return Grade.B
        if t >= 55:
            return Grade.C
        return Grade.PASS


class SkepticReview(BaseModel):
    reviewed_at: datetime
    base_rate: str = Field(description="How often this kind of setup actually works")
    other_side: str = Field(description="Who sells this to you and why they're willing")
    kill_conditions: list[str] = Field(description="Specific prints that falsify the thesis")
    scores: GradeScores


# --------------------------------------------------------------------------- #
# Stage 6-7 — human decision + Bookkeeper (entry)
# --------------------------------------------------------------------------- #

class ExitPlan(BaseModel):
    """The four-part ticket's exit half, set before entry."""
    target_price: float               # exit-up
    max_loss_note: str = Field(
        description="Stop rule or explicit max-loss acceptance. Must be CONCRETE and checkable "
                    "at a glance ('exit below $8.00', not 'exit if weak'). Immutable once set: "
                    "may trail in your favor, never widen against you."
    )
    thesis_expiry: date = Field(description="Date the idea is wrong even if price hasn't moved")


class EntryRecord(BaseModel):
    entered_at: datetime
    contracts: int
    fill_price: float = Field(description="ACTUAL fill, not mid")
    mid_at_entry: float
    underlying_at_entry: float
    entry_snapshot: Greeks

    exit_plan: ExitPlan

    @computed_field
    @property
    def debit_total(self) -> float:
        return self.fill_price * self.contracts * 100

    @computed_field
    @property
    def half_spread_paid(self) -> float:
        """Explicit execution cost line: how far above mid we filled, per contract-share."""
        return (self.fill_price - self.mid_at_entry) * self.contracts * 100


# --------------------------------------------------------------------------- #
# Stage 8 — Bookkeeper (resolution) + P&L attribution
# --------------------------------------------------------------------------- #

class PnLAttribution(BaseModel):
    """Decomposition of realized P&L. delta+vega+theta+spread+residual ≈ realized_pnl.
    This table is the point of tracking all the Greeks: it says whether losses came
    from bad direction calls or bad Greek management."""
    delta_pnl: float
    vega_pnl: float
    theta_pnl: float
    spread_cost: float
    residual: float = Field(description="Unexplained remainder (gamma, rho, model error)")


class ExitRecord(BaseModel):
    closed_at: datetime
    exit_reason: str = Field(description="target / max-loss / thesis-expiry / manual")
    fill_price: float
    mid_at_exit: float
    underlying_at_exit: float
    exit_snapshot: Greeks
    realized_pnl: float
    realized_pnl_pct: float
    attribution: PnLAttribution


# --------------------------------------------------------------------------- #
# Root object
# --------------------------------------------------------------------------- #

class OptionsTicket(BaseModel):
    ticket_id: str
    status: TicketStatus = TicketStatus.CANDIDATE
    created_at: datetime

    # populated by stage; None until that stage runs
    identity: Identity
    trigger: TriggerInfo
    data_block: Optional[DataBlock] = None
    gauntlet: Optional[GauntletResult] = None
    memo: Optional[Memo] = None
    skeptic: Optional[SkepticReview] = None
    entry: Optional[EntryRecord] = None
    exit: Optional[ExitRecord] = None

    notes: Optional[str] = None
