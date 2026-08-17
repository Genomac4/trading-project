import os
import re
import requests
import numpy as np
import pandas as pd
import streamlit as st
from datetime import datetime, timedelta
from urllib.parse import urlparse
from config import EODHD_API_KEY
from data_layer import get_valuation, get_sector, refresh_universe_prices
from explore_constituents import get_sp500_membership
from backtest_composite import cached_fscore, cached_value, zscore
from pit_factors import pit_fscore, pit_value_inputs
from screener import get_regime_weights
# Read-only Schwab access (GET endpoints only — see schwab_client.py docstring).
from schwab_client import SchwabAuthError, get_balances, get_positions

PRICE_DIR = "data/universe_prices"
SCAN_HISTORY_DIR = "data/scan_history"
TRADE_LOG = "trade_log.csv"
YEAR = datetime.now().year

# Canonical trade-log schema (idea tracker v2). Older logs are migrated to this
# on read/write (see load_trade_log).
LOG_COLUMNS = [
    "date", "ticker", "entry_price", "composite",
    "momentum", "fscore", "trend", "earn_yield",
    "above_200d", "regime", "status",
    "target_price", "stop_price", "notes",
    "close_date", "close_price", "outcome_note",
]

st.set_page_config(page_title="Stock Screener", layout="wide")
st.title("Systematic Stock Screener")


# ---------- Data loaders ----------
@st.cache_data(show_spinner=False)
def load_prices():
    all_series = {}
    for fname in os.listdir(PRICE_DIR):
        if fname.endswith(".csv"):
            t = fname[:-4]
            df = pd.read_csv(f"{PRICE_DIR}/{fname}", index_col=0, parse_dates=True)
            if "Close" in df.columns and len(df) > 0:
                all_series[t] = df["Close"]
    return pd.DataFrame(all_series).sort_index()


@st.cache_data(show_spinner=False)
def current_members():
    m = get_sp500_membership()
    return sorted(m[m["end"].isna()]["ticker"].dropna().unique())


@st.cache_data(show_spinner=False, ttl=3600)
def spy_prices():
    url = f"https://eodhd.com/api/eod/SPY.US?api_token={EODHD_API_KEY}&fmt=json&period=d"
    data = requests.get(url, timeout=30).json()
    df = pd.DataFrame(data)
    df["date"] = pd.to_datetime(df["date"])
    return df.set_index("date")["adjusted_close"]


def live_prices(ticker):
    """Pull any ticker's price history live from EODHD; cache it for next time."""
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
        s = df["adjusted_close"].rename("Close")
        os.makedirs(PRICE_DIR, exist_ok=True)
        s.to_frame().to_csv(f"{PRICE_DIR}/{ticker}.csv")
        return s
    except Exception:
        return None


def refresh_prices(tickers, status):
    """Batch-refresh a list of tickers via EODHD's bulk last-day endpoint
    (one call for the whole market, plus one per missing trading day) instead
    of one call per ticker. live_prices is the per-ticker full-history fallback
    for anything uncached, too short, or very stale."""
    def progress(done, total):
        if total and (done % 25 == 0 or done == total):
            status.write(f"Backfilling stale tickers {done}/{total}...")

    counts = refresh_universe_prices(
        list(tickers), price_dir=PRICE_DIR, progress=progress,
        per_ticker_fetch=live_prices)
    load_prices.clear()
    status.write(
        f"updated={counts['updated']}  fallback={counts['fell_back']}  "
        f"already-current={counts['up_to_date']}  no-data={counts['no_data']}")
    return counts["updated"] + counts["fell_back"]


def get_price_series(ticker, prices_all):
    """Cache first, live fallback."""
    if ticker in prices_all.columns:
        s = prices_all[ticker].dropna()
        if len(s) > 0:
            return s
    return live_prices(ticker)


# ---------- Factor computation ----------
def compute_factors(t, prices_all):
    s = get_price_series(t, prices_all)
    if s is None or len(s) < 260:
        return None
    p_now = s.iloc[-1]
    mom = (s.iloc[-21] / s.iloc[-252]) - 1
    ma200 = s.rolling(200).mean()
    tr = (p_now / ma200.iloc[-1]) - 1 if pd.notna(ma200.iloc[-1]) and ma200.iloc[-1] > 0 else None

    fs = cached_fscore(t, YEAR)
    if fs is None:
        try:
            fs = pit_fscore(t, f"{YEAR}-06-30")
        except Exception:
            fs = None
    vi = cached_value(t, YEAR)
    if vi is None:
        try:
            vi = pit_value_inputs(t, f"{YEAR}-06-30")
        except Exception:
            vi = None
    ey = None
    if vi and p_now > 0:
        mc = p_now * vi["shares"]
        if mc > 0:
            ey = vi["net_income"] / mc
    return {"Ticker": t, "price": p_now, "momentum": mom,
            "trend": tr, "fscore": fs, "earn_yield": ey,
            "sector": get_sector(t), "_series": s}


def winsorize(s, lower=0.01, upper=0.99):
    """Cap a factor series at its 1st/99th percentiles before z-scoring so a
    genuine extreme outlier (e.g. SNDK's real +3,982% momentum) can't dominate
    the composite and crush everything else's relative score. Values are capped
    to the bounds, never dropped, and NaNs pass through untouched — so the
    outlier stays pinned at the top of its factor, just no longer at an absurd
    z-distance from the field."""
    valid = s.dropna()
    if len(valid) < 2:
        return s
    lo, hi = valid.quantile(lower), valid.quantile(upper)
    return s.clip(lower=lo, upper=hi)


def add_zscores(df, w_mom, w_qual, w_val, w_trend):
    n = len(df)
    def z(col):
        if n > 1:
            return zscore(winsorize(df[col])).fillna(0)
        return pd.Series(0.0, index=df.index)
    df["z_mom"] = z("momentum")
    df["z_qual"] = z("fscore")
    df["z_val"] = z("earn_yield")
    df["z_trend"] = z("trend")
    df["composite"] = (w_mom * df["z_mom"] + w_qual * df["z_qual"]
                       + w_val * df["z_val"] + w_trend * df["z_trend"])
    return df.sort_values("composite", ascending=False).reset_index(drop=True)


# ---------- Post-ranking selection ----------
def select_top(ranked, top_n, diversify=False, cap=2):
    """Trim a composite-sorted frame to the top_n selection.

    diversify=False -> pure composite rank: the head of the frame.
    diversify=True  -> walk the composite-sorted list top-down and keep each
    name unless its sector already has `cap` picks, yielding the highest-
    composite names spread across sectors instead of clustered in one. This is a
    pure post-ranking filter — it never touches the composite or its ordering.

    Assumes `ranked` is already sorted by composite descending (as returned by
    add_zscores / scan_universe). A missing sector counts as 'Unknown'."""
    if not diversify:
        return ranked.head(top_n)
    counts, keep = {}, []
    for idx, row in ranked.iterrows():
        sec = row.get("sector")
        sec = sec if isinstance(sec, str) and sec else "Unknown"
        if counts.get(sec, 0) >= cap:
            continue
        counts[sec] = counts.get(sec, 0) + 1
        keep.append(idx)
        if len(keep) >= top_n:
            break
    return ranked.loc[keep]


# ---------- Regime ----------
def current_regime():
    """Single source of truth for the regime + its weights. TRENDING weights are
    backtest-derived; CHOPPY weights are not — see screener.REGIME_WEIGHTS."""
    return get_regime_weights(spy_prices())


def regime_banner(reg):
    pct = reg["pct_vs_200d"]
    w = reg["weights"]
    weight_str = (f"momentum {w['momentum']:.2f} · quality {w['quality']:.2f} · "
                  f"value {w['value']:.2f} · trend {w['trend']:.2f}")
    if reg["trending"]:
        st.success(f"MARKET REGIME: TRENDING — SPY {pct:+.1%} vs 200d MA. "
                   f"Active weight set (momentum-heavy, from backtest): {weight_str}.")
    else:
        # Not "from backtest": the live CHOPPY vector was never backtested.
        # See the note at screener.REGIME_WEIGHTS. Say so here rather than
        # imply support that does not exist — this banner is the only place a
        # first-ever CHOPPY regime would announce itself.
        st.warning(f"MARKET REGIME: CHOPPY/DEFENSIVE — SPY {pct:+.1%} vs 200d MA. "
                   f"Active weight set (quality/trend-heavy): {weight_str}. "
                   f"⚠ These CHOPPY weights are UNVALIDATED — never backtested "
                   f"(see screener.REGIME_WEIGHTS). Treat rankings with extra caution.")


# ---------- Thesis ----------
def make_thesis(row):
    parts = []
    t = row["Ticker"]
    if row["z_mom"] > 0.5:
        parts.append(f"strong 12-1 momentum ({row['momentum']:+.0%}, top tier of universe)")
    elif row["z_mom"] < -0.5:
        parts.append(f"weak momentum ({row['momentum']:+.0%}) — ranked despite the tape")
    elif pd.notna(row["momentum"]):
        parts.append(f"momentum {row['momentum']:+.0%}")
    if pd.notna(row["fscore"]):
        if row["fscore"] >= 7:
            parts.append(f"high quality (F-score {row['fscore']:.0f}/9 — improving fundamentals)")
        elif row["fscore"] <= 3:
            parts.append(f"weak fundamentals (F-score {row['fscore']:.0f}/9)")
        else:
            parts.append(f"middling quality (F-score {row['fscore']:.0f}/9)")
    if pd.notna(row.get("earn_yield")):
        parts.append(f"earnings yield {row['earn_yield']:.1%}")
    if pd.notna(row.get("trend")):
        if row["trend"] > 0:
            parts.append(f"above its 200d MA ({row['trend']:+.0%})")
        else:
            parts.append(f"below its 200d MA ({row['trend']:+.0%})")
    why = "; ".join(parts) if parts else "insufficient factor data"
    return f"{t}: {why}. Composite {row['composite']:.2f}."


# ---------- Full S&P scan ----------
def scan_universe(w_mom, w_qual, w_val, w_trend):
    """Score and rank EVERY current S&P member by the given weights.
    Returns the full ranked frame (rank 1 = best) — callers trim as needed."""
    prices_all = load_prices()
    members = current_members()
    rows = []
    for t in members:
        if t not in prices_all.columns:
            continue
        r = compute_factors(t, prices_all)
        if r is not None:
            rows.append(r)
    df = pd.DataFrame(rows).dropna(subset=["momentum"])
    df = add_zscores(df, w_mom, w_qual, w_val, w_trend)
    df["rank"] = range(1, len(df) + 1)
    return df


def full_scan(w_mom, w_qual, w_val, w_trend, top_n, diversify=False, cap=2):
    ranked = scan_universe(w_mom, w_qual, w_val, w_trend)
    return select_top(ranked, top_n, diversify, cap)


# ---------- Flagging engine (scan history + diffs) ----------
SNAPSHOT_COLS = ["date", "Ticker", "sector", "price", "momentum", "trend",
                 "fscore", "earn_yield", "composite", "rank", "regime"]


def save_snapshot(scored, date_str, regime):
    """Persist a ranked scan to data/scan_history/{date}.csv (overwrites same-day)."""
    os.makedirs(SCAN_HISTORY_DIR, exist_ok=True)
    snap = scored[["Ticker", "sector", "price", "momentum", "trend",
                   "fscore", "earn_yield", "composite", "rank"]].copy()
    snap.insert(0, "date", date_str)
    snap["regime"] = regime
    snap[SNAPSHOT_COLS].to_csv(f"{SCAN_HISTORY_DIR}/{date_str}.csv", index=False)


def prior_snapshot(before_date):
    """Most recent saved snapshot strictly before `before_date`.
    Returns (date_str, frame) or (None, None) if there is no earlier scan."""
    if not os.path.isdir(SCAN_HISTORY_DIR):
        return None, None
    dates = sorted(f[:-4] for f in os.listdir(SCAN_HISTORY_DIR)
                   if f.endswith(".csv"))
    earlier = [d for d in dates if d < before_date]
    if not earlier:
        return None, None
    d = earlier[-1]
    return d, pd.read_csv(f"{SCAN_HISTORY_DIR}/{d}.csv")


def latest_snapshot_date():
    """Date string of the most recent saved scan snapshot, or None if none."""
    if not os.path.isdir(SCAN_HISTORY_DIR):
        return None
    dates = sorted(f[:-4] for f in os.listdir(SCAN_HISTORY_DIR)
                   if f.endswith(".csv"))
    return dates[-1] if dates else None


def days_since(date_str):
    """Whole days from date_str (YYYY-MM-DD) to today, or None if unparseable."""
    try:
        return (datetime.now().date()
                - datetime.strptime(date_str, "%Y-%m-%d").date()).days
    except (ValueError, TypeError):
        return None


def compute_flags(scored, prior_df, prior_date, top_n=20, diversify=False, cap=2):
    """Diff today's ranked scan against the prior snapshot.

    Returns a dict with four lists. New/trend/movers items are (row, reason)
    where row is the live scored Series (has z-scores for make_thesis and every
    field log_trade needs). Dropped items are (ticker, row_or_None, reason).

    When diversify is on, "top N" means the sector-diversified selection (max
    `cap` per sector) on BOTH sides of the diff, so new entrants / drop-outs are
    computed against the same filter the user is looking at."""
    by_tkr = scored.set_index("Ticker", drop=False)
    p = prior_df.set_index("Ticker")

    # Older snapshots predate the sector column; backfill it (cache-first) so the
    # prior top-N is diversified the same way as today's.
    prior_sorted = prior_df.sort_values("composite", ascending=False)
    if "sector" not in prior_sorted.columns:
        prior_sorted = prior_sorted.copy()
        prior_sorted["sector"] = prior_sorted["Ticker"].map(get_sector)

    today_sel = select_top(scored, top_n, diversify, cap)
    prior_sel = select_top(prior_sorted, top_n, diversify, cap)
    today_top = set(today_sel["Ticker"])
    prior_top = set(prior_sel["Ticker"])

    # 1. NEW ENTRANTS — in today's top N, not in the prior top N.
    new_entrants = []
    for tk in today_sel["Ticker"]:
        if tk not in prior_top:
            row = by_tkr.loc[tk]
            was = f"was #{int(p.loc[tk, 'rank'])}" if tk in p.index else "not in prior scan"
            new_entrants.append((row, f"Entered top {top_n} at #{int(row['rank'])} ({was})."))

    # 2. TREND CONFIRMATIONS — F-score >= 7 AND crossed above 200d MA since last scan.
    trend_conf = []
    for tk in scored["Ticker"]:
        row = by_tkr.loc[tk]
        if (tk in p.index and pd.notna(row["fscore"]) and row["fscore"] >= 7
                and pd.notna(row["trend"]) and row["trend"] > 0):
            prev = p.loc[tk, "trend"]
            if pd.notna(prev) and prev <= 0:
                trend_conf.append((row, f"F-score {row['fscore']:.0f}/9 and crossed "
                                        f"above its 200d MA ({prev:+.0%} → {row['trend']:+.0%}) "
                                        f"since {prior_date}."))

    # 3. BIG MOVERS — top 5 composite-score increases vs the prior scan.
    deltas = []
    for tk in scored["Ticker"]:
        if tk in p.index:
            d = float(by_tkr.loc[tk, "composite"] - p.loc[tk, "composite"])
            deltas.append((tk, d))
    deltas.sort(key=lambda x: x[1], reverse=True)
    big_movers = []
    for tk, d in deltas[:5]:
        if d <= 0:
            break  # only surface genuine increases
        row = by_tkr.loc[tk]
        big_movers.append((row, f"Composite {d:+.2f} ({p.loc[tk, 'composite']:.2f} → "
                                f"{row['composite']:.2f}) since {prior_date}."))

    # 4. DROPPED OUT — were in the prior top N, not in today's top N.
    dropped = []
    for tk in sorted(prior_top - today_top):
        if tk in by_tkr.index:
            row = by_tkr.loc[tk]
            reason = (f"Fell out of top {top_n}: was #{int(p.loc[tk, 'rank'])}, "
                      f"now #{int(row['rank'])}.")
        else:
            row = None
            reason = (f"Was #{int(p.loc[tk, 'rank'])} in the prior top {top_n}; "
                      f"no longer in the scan.")
        dropped.append((tk, row, reason))

    return {"new": new_entrants, "trend": trend_conf,
            "movers": big_movers, "dropped": dropped}


def render_flags(title, items, reg, key):
    """Render one flag category: ticker, composite now, reason, thesis, log button."""
    st.markdown(f"#### {title}")
    if not items:
        st.caption("Nothing flagged.")
        return
    widths = [0.8, 1.4, 1.0, 3.0, 4.0, 1.4]
    for i, (row, reason) in enumerate(items):
        c = st.columns(widths)
        c[0].markdown(f"**{row['Ticker']}**")
        c[1].caption(str(row.get("sector") or "Unknown"))
        c[2].write(f"{row['composite']:.2f}")
        c[3].write(reason)
        c[4].caption(make_thesis(row))
        if c[5].button("Log this idea", key=f"{key}_{i}_{row['Ticker']}"):
            log_trade(row, reg, notes=f"Flagged ({title})")
            st.success(f"Logged {row['Ticker']} — see the Ideas tab.")


def render_dropped(items):
    """Dropped-out flag: ticker, composite now (if still scanned), reason. No log button."""
    st.markdown("#### Dropped Out")
    if not items:
        st.caption("Nothing flagged.")
        return
    widths = [0.8, 1.4, 1.0, 5.0]
    for tk, row, reason in items:
        c = st.columns(widths)
        c[0].markdown(f"**{tk}**")
        sec = str(row.get("sector") or "Unknown") if row is not None else get_sector(tk)
        c[1].caption(sec)
        c[2].write(f"{row['composite']:.2f}" if row is not None else "—")
        c[3].write(reason)


# ---------- Trade log ----------
def load_trade_log():
    """Read trade_log.csv, tolerating the old schema. Always returns a frame
    with exactly LOG_COLUMNS so the rest of the app can rely on it."""
    if not os.path.exists(TRADE_LOG):
        return pd.DataFrame(columns=LOG_COLUMNS)
    try:
        df = pd.read_csv(TRADE_LOG)
    except Exception:
        return pd.DataFrame(columns=LOG_COLUMNS)
    # migrate v1 columns
    if "price" in df.columns and "entry_price" not in df.columns:
        df = df.rename(columns={"price": "entry_price"})
    if "thesis" in df.columns:
        # preserve the old auto-thesis as notes where notes is empty
        notes = df.get("notes", pd.Series("", index=df.index)).astype(str)
        blank = notes.str.strip().isin(["", "nan"])
        df["notes"] = notes.where(~blank, df["thesis"].astype(str))
    for col in LOG_COLUMNS:
        if col not in df.columns:
            df[col] = ""
    # default a missing/blank status to "open" so legacy rows still surface
    status = df["status"].astype(str).str.strip().str.lower()
    df["status"] = status.where(status.isin(["open", "closed"]), "open")
    return df[LOG_COLUMNS].reset_index(drop=True)


def log_trade(row, reg, target_price="", stop_price="", notes=""):
    """Record a factor snapshot for an idea as a new OPEN row."""
    trend = row.get("trend")
    above = bool(trend > 0) if pd.notna(trend) else ""

    def num(x, nd=4):
        return round(float(x), nd) if pd.notna(x) else ""

    entry = {
        "date": datetime.now().strftime("%Y-%m-%d"),
        "ticker": row["Ticker"],
        "entry_price": num(row.get("price"), 2),
        "composite": num(row.get("composite"), 2),
        "momentum": num(row.get("momentum")),
        "fscore": num(row.get("fscore"), 0),
        "trend": num(trend),
        "earn_yield": num(row.get("earn_yield")),
        "above_200d": above,
        "regime": reg.get("regime", ""),
        "status": "open",
        "target_price": target_price,
        "stop_price": stop_price,
        "notes": notes,
        "close_date": "",
        "close_price": "",
        "outcome_note": "",
    }
    out = pd.concat([load_trade_log(), pd.DataFrame([entry], columns=LOG_COLUMNS)],
                    ignore_index=True)
    out.to_csv(TRADE_LOG, index=False)


def close_idea(idx, close_price, note):
    """Mark the idea at position idx closed, stamping close date/price/note."""
    df = load_trade_log()
    if idx not in df.index:
        return
    # blank columns read back as float64 (all-NaN); cast to object so writing
    # date/note strings into them doesn't raise on strict pandas.
    df = df.astype(object)
    df.loc[idx, "status"] = "closed"
    df.loc[idx, "close_date"] = datetime.now().strftime("%Y-%m-%d")
    df.loc[idx, "close_price"] = round(float(close_price), 2) if close_price is not None else ""
    df.loc[idx, "outcome_note"] = note
    df.to_csv(TRADE_LOG, index=False)


# ---------- Live scoring for open ideas ----------
@st.cache_data(show_spinner=False, ttl=1800)
def universe_factor_frame():
    """Current factor snapshot for the whole S&P universe — the z-score
    reference used to recompute an idea's composite 'now'."""
    prices_all = load_prices()
    rows = []
    for t in current_members():
        if t not in prices_all.columns:
            continue
        r = compute_factors(t, prices_all)
        if r is not None:
            rows.append({k: r[k] for k in
                         ("Ticker", "price", "momentum", "trend", "fscore", "earn_yield")})
    return pd.DataFrame(rows).dropna(subset=["momentum"])


def score_against_universe(r, weights, uni_raw):
    """Composite for one idea's current factors, standardized vs the universe."""
    if r is None or pd.isna(r.get("momentum")):
        return None
    rr = {k: r.get(k) for k in
          ("Ticker", "price", "momentum", "trend", "fscore", "earn_yield")}
    frame = uni_raw
    if len(uni_raw) == 0 or rr["Ticker"] not in set(uni_raw["Ticker"]):
        frame = pd.concat([uni_raw, pd.DataFrame([rr])], ignore_index=True)
    if len(frame) < 2:
        return None
    scored = add_zscores(frame.copy(), *weights)
    hit = scored[scored["Ticker"] == rr["Ticker"]]
    return float(hit.iloc[0]["composite"]) if len(hit) else None


def idea_live(ticker, weights, uni_raw):
    """Cache-first current price + recomputed composite for an open idea.
    Defensive: a bad/short-history ticker yields Nones, never raises."""
    price, comp = None, None
    try:
        s = get_price_series(ticker, load_prices())
        if s is not None and len(s) > 0:
            price = float(s.iloc[-1])
    except Exception:
        pass
    try:
        r = compute_factors(ticker, load_prices())
        if r is not None:
            comp = score_against_universe(r, weights, uni_raw)
    except Exception:
        pass
    return {"price": price, "comp_now": comp}


# ---------- Deep dive ----------
def deep_dive(df, key_prefix, reg):
    st.subheader("Deep Dive")
    pick = st.selectbox("Stock", df["Ticker"].tolist(), key=f"{key_prefix}_pick")
    row = df[df["Ticker"] == pick].iloc[0]

    thesis = make_thesis(row)
    st.info(thesis)

    c1, c2, c3 = st.columns(3)
    c1.metric("Price", f"${row['price']:.2f}")
    c1.metric("Composite", f"{row['composite']:.2f}")
    c2.metric("Momentum 12-1", f"{row['momentum']:+.1%}")
    c2.metric("vs 200d MA", f"{row['trend']:+.1%}" if pd.notna(row['trend']) else "—")
    c3.metric("F-Score", f"{row['fscore']:.0f}/9" if pd.notna(row['fscore']) else "—")
    c3.metric("Earnings Yield", f"{row['earn_yield']:.1%}" if pd.notna(row['earn_yield']) else "—")

    st.markdown("**Log this idea**")
    li1, li2 = st.columns(2)
    tgt = li1.number_input("Target price ($, optional)", min_value=0.0, value=0.0,
                           step=1.0, key=f"{key_prefix}_logtgt")
    stp = li2.number_input("Stop price ($, optional)", min_value=0.0, value=0.0,
                           step=1.0, key=f"{key_prefix}_logstop")
    note = st.text_input("Notes (optional)", key=f"{key_prefix}_lognote")
    if st.button("Log this idea", key=f"{key_prefix}_log"):
        log_trade(row, reg,
                  target_price=round(tgt, 2) if tgt > 0 else "",
                  stop_price=round(stp, 2) if stp > 0 else "",
                  notes=note.strip())
        st.success(f"Logged {pick} as an open idea — see the Ideas tab.")

    st.markdown("**Valuation & Analyst Reference** (live from EODHD)")
    with st.spinner("Pulling valuation..."):
        try:
            val = get_valuation(pick)
        except Exception:
            val = None
    if val:
        v1, v2, v3, v4 = st.columns(4)
        v1.metric("Trailing P/E", f"{val['trailing_pe']:.1f}x" if val.get('trailing_pe') else "—")
        v1.metric("Forward P/E", f"{val['forward_pe']:.1f}x" if val.get('forward_pe') else "—")
        v2.metric("EV/EBITDA", f"{val['ev_ebitda']:.1f}x" if val.get('ev_ebitda') else "—")
        v2.metric("EV/Revenue", f"{val['ev_revenue']:.1f}x" if val.get('ev_revenue') else "—")
        v3.metric("PEG", f"{val['peg']:.2f}" if val.get('peg') else "—")
        v3.metric("Price/Sales", f"{val['price_sales']:.1f}x" if val.get('price_sales') else "—")
        v4.metric("EPS est (this yr)", f"${val['eps_est_current_yr']:.2f}" if val.get('eps_est_current_yr') else "—")
        v4.metric("EPS est (next yr)", f"${val['eps_est_next_yr']:.2f}" if val.get('eps_est_next_yr') else "—")

        eps_now, eps_next = val.get("eps_est_current_yr"), val.get("eps_est_next_yr")
        if eps_now and eps_next and eps_now > 0:
            st.caption(f"Implied EPS growth next year: {(eps_next/eps_now - 1):+.1%}")

        target = val.get("analyst_target")
        rating = val.get("analyst_rating")
        if target and row["price"]:
            upside = (target / row["price"]) - 1
            st.markdown("### 🎯 Analyst Consensus")
            a1, a2, a3 = st.columns(3)
            a1.metric("Consensus target", f"${target:.0f}",
                      f"{upside:+.1%} vs current", delta_color="normal")
            a2.metric("Rating", f"{rating:.2f}/5" if rating else "—",
                      help="EODHD consensus: 1 = Strong Buy … 5 = Strong Sell")
            sb, b, h, s = (val.get("strong_buy"), val.get("buy"),
                           val.get("hold"), val.get("sell"))
            n_analysts = sum(int(x) for x in (sb, b, h, s)
                             if isinstance(x, (int, float)))
            a3.metric("Analysts covering", str(n_analysts) if n_analysts else "—")
            st.markdown(
                f"**Ratings breakdown:**   "
                f":green[**Strong Buy {sb}**] · :green[**Buy {b}**] · "
                f"**Hold {h}** · :red[**Sell {s}**]")

        your_target = st.number_input("Your price target ($)", min_value=0.0,
                                      value=0.0, step=5.0, key=f"{key_prefix}_tgt")
        if your_target > 0 and target:
            your_up = (your_target / row["price"]) - 1
            gap = (your_target / target) - 1
            st.write(f"Your target: **${your_target:.0f}** ({your_up:+.1%} vs current) — "
                     f"{gap:+.1%} vs the Street ({'more bullish' if gap > 0 else 'more bearish'})")
    else:
        st.caption("Valuation data unavailable for this ticker.")

    s = row.get("_series")
    if s is not None and len(s) > 0:
        s2 = s.iloc[-504:]
        chart = pd.DataFrame({
            "Price": s2,
            "MA50": s2.rolling(50).mean(),
            "MA200": s2.rolling(200).mean(),
        })
        st.line_chart(chart)

    if os.path.exists(TRADE_LOG):
        st.subheader("Trade Log")
        st.dataframe(pd.read_csv(TRADE_LOG), use_container_width=True)


def show_table(df):
    cols = ["Ticker", "sector", "price", "composite",
            "momentum", "trend", "fscore", "earn_yield"]
    show = df[[c for c in cols if c in df.columns]]
    st.dataframe(show.style.format({
        "price": "${:.2f}", "composite": "{:.2f}", "momentum": "{:+.1%}",
        "trend": "{:+.1%}", "fscore": "{:.0f}", "earn_yield": "{:.1%}",
    }, na_rep="—"), use_container_width=True)


# ---------- Paper portfolios (two named tracks vs SPY over time) ----------
PAPER_DIR = "data/paper_portfolio"
REBAL_COLS = ["date", "regime", "tickers", "entry_prices", "weights"]
PORTFOLIO_BASKET_N = 30
START_CAPITAL = 10_000.0
REBALANCE_MIN_DAYS = 25  # monthly cadence guard (block re-rebalancing sooner)
REBALANCE_DUE_DAYS = 28  # status banner: flag as due for the monthly rebalance
SCAN_STALE_DAYS = 7      # flags banner: suggest a fresh scan past this age

# Two tracks run head-to-head from the same inception: same universe scan, same
# regime weights, different post-ranking selection filter. Each has its own file.
TRACKS = {
    "concentrated": {"label": "Concentrated", "diversify": False, "cap": 2},
    "diversified": {"label": "Diversified", "diversify": True, "cap": 2},
}


def rebalances_path(track):
    return f"{PAPER_DIR}/{track}.csv"


def load_rebalances(track):
    """Raw rebalance ledger for one track as a frame (one row per rebalance).
    Always the REBAL_COLS schema; empty frame if the track hasn't been started."""
    path = rebalances_path(track)
    if not os.path.exists(path):
        return pd.DataFrame(columns=REBAL_COLS)
    try:
        df = pd.read_csv(path, dtype=str)
    except Exception:
        return pd.DataFrame(columns=REBAL_COLS)
    for col in REBAL_COLS:
        if col not in df.columns:
            df[col] = ""
    return df[REBAL_COLS]


def parse_rebalances(track):
    """One track's rebalances as a date-sorted list of dicts with parsed holding
    lists: {date, regime, tickers:[...], entry_prices:[float], weights:[float]}."""
    df = load_rebalances(track)
    out = []
    for _, r in df.iterrows():
        tickers = [t for t in str(r["tickers"]).split("|") if t]
        if not tickers:
            continue

        def nums(field):
            vals = []
            for x in str(r[field]).split("|"):
                try:
                    vals.append(float(x))
                except (ValueError, TypeError):
                    vals.append(None)
            return vals

        out.append({"date": str(r["date"]), "regime": str(r["regime"]),
                    "tickers": tickers, "entry_prices": nums("entry_prices"),
                    "weights": nums("weights")})
    out.sort(key=lambda x: x["date"])
    return out


def save_rebalance(track, date_str, selected, regime):
    """Append (or same-day overwrite) a rebalance snapshot to a track's file.
    `selected` is the ranked top-N frame with Ticker + price; equal-weighted."""
    os.makedirs(PAPER_DIR, exist_ok=True)
    tickers = list(selected["Ticker"])
    prices = [round(float(p), 4) for p in selected["price"]]
    w = round(1.0 / len(tickers), 6)
    row = {
        "date": date_str,
        "regime": regime,
        "tickers": "|".join(tickers),
        "entry_prices": "|".join(str(p) for p in prices),
        "weights": "|".join(str(w) for _ in tickers),
    }
    df = load_rebalances(track)
    df = df[df["date"] != date_str]  # same-day rebalance replaces
    df = pd.concat([df, pd.DataFrame([row], columns=REBAL_COLS)], ignore_index=True)
    df.sort_values("date").to_csv(rebalances_path(track), index=False)


def _asof(index, ts):
    """Last trading date in `index` on or before ts (rebalances are stamped with
    a calendar date that may fall on a weekend/holiday)."""
    prior = index[index <= ts]
    return prior[-1] if len(prior) else index[0]


def build_strategy_curve(rebs, prices_all, start_capital=START_CAPITAL):
    """Chain equal-weight basket returns across holding periods into one daily
    equity curve starting at $start_capital. Within a period the basket value is
    the mean of each held name's price / its entry price (missing names drop out
    of the mean — defensive), and the level carries across rebalance boundaries."""
    if not rebs or prices_all is None or prices_all.shape[1] == 0:
        return None
    idx = prices_all.index
    if len(idx) == 0:
        return None
    today = idx.max()
    reb_asof = [_asof(idx, pd.Timestamp(r["date"])) for r in rebs]
    if reb_asof[0] > today:
        return None

    curve = {}
    level = float(start_capital)
    for k, reb in enumerate(rebs):
        start = reb_asof[k]
        end = reb_asof[k + 1] if k + 1 < len(rebs) else today
        span = idx[(idx >= start) & (idx <= end)]
        if len(span) == 0:
            continue
        held = [t for t in reb["tickers"] if t in prices_all.columns]
        if not held:
            for d in span:
                curve[d] = level
            continue
        sub = prices_all[held].loc[start:end].ffill()
        base = sub.loc[start]
        valid = base[base.notna() & (base > 0)].index.tolist()
        if not valid:
            for d in span:
                curve[d] = level
            continue
        # equal-weight basket value vs entry; mean skips NaN (dropped names)
        growth = sub[valid].div(base[valid]).mean(axis=1)
        period = level * growth
        for d, v in period.items():
            curve[d] = float(v)
        level = float(period.iloc[-1])  # carry the boundary level forward
    if not curve:
        return None
    return pd.Series(curve).sort_index()


def build_spy_curve(index, start_capital=START_CAPITAL):
    """SPY buy-and-hold rebased to $start_capital over the strategy's dates."""
    try:
        spy = spy_prices()
    except Exception:
        return None
    spy = spy.reindex(index).ffill().dropna()
    if len(spy) == 0:
        return None
    return start_capital * spy / spy.iloc[0]


def portfolio_metrics(equity):
    """Same five metrics as the backtest's performance_metrics (total,
    annualized, vol, Sharpe, max drawdown), adapted from a daily live equity
    curve: calendar-time annualization and daily vol × sqrt(252)."""
    if equity is None:
        return None
    e = equity.dropna()
    if len(e) < 2:
        return None
    total = e.iloc[-1] / e.iloc[0] - 1
    days = (e.index[-1] - e.index[0]).days
    years = days / 365.25 if days > 0 else None
    annual = (1 + total) ** (1 / years) - 1 if years and years > 0 else None
    r = e.pct_change().dropna()
    vol = float(r.std() * np.sqrt(252)) if len(r) > 1 else None
    sharpe = (annual / vol) if (annual is not None and vol) else None
    max_dd = float(((e - e.cummax()) / e.cummax()).min())
    return {"total": total, "annual": annual, "vol": vol,
            "sharpe": sharpe, "max_dd": max_dd}


def metrics_table(pairs):
    """Format a {label: metrics_dict} mapping into a display frame."""
    def pct(x):
        return f"{x:+.1%}" if isinstance(x, (int, float)) else "—"

    rows = []
    for label, m in pairs:
        if m is None:
            rows.append({"": label, "Total": "—", "Annualized": "—",
                         "Volatility": "—", "Sharpe": "—", "Max DD": "—"})
            continue
        rows.append({
            "": label,
            "Total": pct(m["total"]),
            "Annualized": pct(m["annual"]),
            "Volatility": f"{m['vol']:.1%}" if m["vol"] is not None else "—",
            "Sharpe": f"{m['sharpe']:.2f}" if m["sharpe"] is not None else "—",
            "Max DD": pct(m["max_dd"]),
        })
    return pd.DataFrame(rows).set_index("")


# ---------- Market Awareness (display-only) ----------
# FIREWALL: everything in this section is read-only context for discretionary
# judgment. It must never feed the screener composite, the flags diff, or the
# paper-portfolio ledgers — no writes, no factor inputs, display only.

# EODHD sector names (General.Sector, cached in data/sectors.csv) -> SPDR ETF.
SECTOR_ETF = {
    "Technology": "XLK", "Energy": "XLE", "Utilities": "XLU",
    "Financial Services": "XLF", "Healthcare": "XLV", "Industrials": "XLI",
    "Basic Materials": "XLB", "Consumer Cyclical": "XLY",
    "Consumer Defensive": "XLP", "Real Estate": "XLRE",
    "Communication Services": "XLC",
}

# The economic-events endpoint has no importance field on our plan, so "high
# importance" is a name filter. EODHD calls CPI "Inflation Rate".
HIGH_IMPORTANCE_EVENTS = {
    "Fed Interest Rate Decision": "FOMC",
    "Inflation Rate": "CPI",
    "Non Farm Payrolls": "Nonfarm Payrolls",
    "GDP Growth Rate": "GDP",
    "Gross Domestic Product": "GDP",
}

# Hyperscalers / AI-capex drivers — their earnings (capex guidance) are the
# thesis catalysts for the whole book. Hardcoded for Stage 1.
THESIS_WATCHLIST = ["MSFT", "GOOGL", "AMZN", "META", "NVDA"]

# Within this many pct-points of SPY = a market move; of the sector ETF = a
# sector move; otherwise the move is stock-specific.
MOVE_MATCH_PP = 1.0
EARNINGS_SOON_DAYS = 14


@st.cache_data(show_spinner=False, ttl=6 * 3600)
def macro_calendar():
    """High-importance US economic events over the next 14 days as a
    date-sorted list of (label, date). None = endpoint unavailable."""
    start = datetime.now().date()
    end = start + timedelta(days=14)
    url = (f"https://eodhd.com/api/economic-events?api_token={EODHD_API_KEY}"
           f"&country=US&from={start}&to={end}&limit=1000&fmt=json")
    try:
        resp = requests.get(url, timeout=30)
        if resp.status_code != 200:
            return None
        seen, out = set(), []
        for e in resp.json():
            label = HIGH_IMPORTANCE_EVENTS.get(str(e.get("type", "")))
            if not label:
                continue
            try:
                d = pd.Timestamp(e["date"]).date()
            except (ValueError, TypeError, KeyError):
                continue
            # mom/yoy variants and GDP aliases arrive as separate rows
            if (label, d) in seen:
                continue
            seen.add((label, d))
            out.append((label, d))
        return sorted(out, key=lambda x: x[1])
    except Exception:
        return None


@st.cache_data(show_spinner=False, ttl=3600)
def treasury_10y():
    """10-year Treasury yield level + 1-day change in bp, from US10Y.GBOND EOD
    closes (the real-time endpoint returns NA for bonds on our plan)."""
    start = datetime.now().date() - timedelta(days=14)
    url = (f"https://eodhd.com/api/eod/US10Y.GBOND?api_token={EODHD_API_KEY}"
           f"&fmt=json&period=d&from={start}")
    try:
        data = requests.get(url, timeout=30).json()
        closes = [float(d["close"]) for d in data
                  if isinstance(d.get("close"), (int, float))]
        if len(closes) < 2:
            return None
        return {"level": closes[-1], "chg_bp": (closes[-1] - closes[-2]) * 100}
    except Exception:
        return None


@st.cache_data(show_spinner=False, ttl=300)
def live_quotes(symbols):
    """Delayed intraday quotes for a tuple of full EODHD symbols (e.g.
    'SPY.US', 'VIX.INDX'): {symbol: {close, chg_pct}}. chg_pct is today's %
    change vs previous close. Failed batches just drop out of the dict."""
    out = {}
    syms = list(symbols)
    for i in range(0, len(syms), 15):  # batch limit on the s= parameter
        chunk = syms[i:i + 15]
        url = (f"https://eodhd.com/api/real-time/{chunk[0]}"
               f"?api_token={EODHD_API_KEY}&fmt=json")
        if len(chunk) > 1:
            url += "&s=" + ",".join(chunk[1:])
        try:
            resp = requests.get(url, timeout=30)
            if resp.status_code != 200:
                continue
            data = resp.json()
            for q in [data] if isinstance(data, dict) else data:
                code, close, chg = q.get("code"), q.get("close"), q.get("change_p")
                if (code and isinstance(close, (int, float))
                        and isinstance(chg, (int, float))):
                    out[str(code)] = {"close": float(close), "chg_pct": float(chg)}
        except Exception:
            continue
    return out


@st.cache_data(show_spinner=False, ttl=12 * 3600)
def next_earnings(tickers):
    """Next scheduled report date per plain US ticker (tuple in), from the
    EODHD earnings calendar: {ticker: 'YYYY-MM-DD'}. Missing = unknown."""
    out = {}
    start = datetime.now().date()
    end = start + timedelta(days=120)
    syms = list(tickers)
    for i in range(0, len(syms), 30):
        chunk = syms[i:i + 30]
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
                if tkr and rd and (tkr not in out or rd < out[tkr]):
                    out[tkr] = rd
        except Exception:
            continue
    return out


# --- Recently-reported earnings (Stage 2.5). Same firewall: display only. ---
EARNINGS_LOOKBACK_DAYS = 5


@st.cache_data(show_spinner=False, ttl=3600)
def recent_earnings(tickers):
    """Prints from the last EARNINGS_LOOKBACK_DAYS days for plain US tickers,
    via the earnings calendar — the same structured facts as the fundamentals
    Earnings::History block, but one batched call. {ticker: {report_date,
    when, actual, estimate, surprise_pct}}. EODHD posts actuals with a lag,
    so a same-day report legitimately has actual=None."""
    out = {}
    end = datetime.now().date()
    start = end - timedelta(days=EARNINGS_LOOKBACK_DAYS)
    syms = list(tickers)
    for i in range(0, len(syms), 30):
        chunk = syms[i:i + 30]
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
                if not tkr or not rd:
                    continue
                actual, est, pct = e.get("actual"), e.get("estimate"), e.get("percent")
                actual = actual if isinstance(actual, (int, float)) else None
                est = est if isinstance(est, (int, float)) else None
                if not isinstance(pct, (int, float)):
                    pct = ((actual - est) / abs(est) * 100
                           if actual is not None and est else None)
                if tkr not in out or rd > out[tkr]["report_date"]:
                    out[tkr] = {"report_date": rd,
                                "when": str(e.get("before_after_market") or ""),
                                "actual": actual, "estimate": est,
                                "surprise_pct": pct}
        except Exception:
            continue
    return out


def _edgar_8k_url(tkr):
    """EDGAR's company browse accepts a ticker in the CIK param — links to the
    live 8-K filing list without needing a CIK mapping or SEC API call."""
    return ("https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany"
            f"&CIK={tkr}&type=8-K&dateb=&owner=include&count=10")


def render_recent_earnings(reported, news):
    """The just-reported digest: structured EPS facts, the release on EDGAR,
    and the ticker's pulled coverage under one earnings header."""
    st.markdown("##### Recently reported (last 5 days)")
    st.caption("EPS facts are EODHD's structured earnings history — actuals "
               "can lag a same-day print. **Guidance is the part that moves "
               "the stock, and it lives in the release text, not in "
               "structured data** — read the linked release rather than a "
               "headline summary; nothing here auto-extracts guidance.")
    when_txt = {"BeforeMarket": " before the open", "AfterMarket": " after the close"}
    for t, r in sorted(reported.items(),
                       key=lambda kv: kv[1]["report_date"], reverse=True):
        try:
            d = datetime.strptime(r["report_date"], "%Y-%m-%d").date()
            date_label = f"{d.strftime('%b')} {d.day}"
        except ValueError:
            date_label = r["report_date"]
        label = f"{t} — reported {date_label}{when_txt.get(r['when'], '')}"
        with st.expander(label, expanded=True):
            if r["actual"] is not None:
                est_s = (f" vs est ${r['estimate']:.2f}"
                         if r["estimate"] is not None else "")
                line = f"EPS **${r['actual']:.2f}**{est_s}"
                if r["surprise_pct"] is not None:
                    pc = ("green" if r["surprise_pct"] > 0
                          else "red" if r["surprise_pct"] < 0 else "gray")
                    line += f" · surprise :{pc}[**{r['surprise_pct']:+.1f}%**]"
                st.markdown(line)
            elif r["estimate"] is not None:
                st.markdown(f"EPS est ${r['estimate']:.2f} — actual not "
                            f"posted to EODHD yet")
            else:
                st.markdown("EPS figures unavailable")
            st.markdown(f"[8-K / press release filings on EDGAR]"
                        f"({_edgar_8k_url(t)}) — guidance is in there")
            st.markdown("**Earnings coverage**")
            items = news.get(t) if news else None
            if items:
                render_news_items(items)
            elif items is None:
                st.caption("news unavailable")
            else:
                st.caption("No ticker-matched coverage pulled.")


# --- News layer (Stage 2). Same firewall: read-and-display only. ---
NEWS_DAYS = 3               # look-back for position news
NEWS_PER_POSITION = 5       # anti-noise cap per held/idea ticker
NEWS_PER_THESIS = 3         # anti-noise cap per thesis ticker
THESIS_NEWS_DAYS = 7        # wider window — capex stories are less frequent
NEWS_WRAP_SYMBOLS = 12      # > this many tagged tickers = market-wrap spam
NEWS_DUP_JACCARD = 0.7      # title token overlap that counts as syndication
AUTO_EXPAND_MOVE = 3.0      # |move| % on a SPECIFIC verdict that auto-opens news

# Headline/content relevance filter for the thesis-break layer.
THESIS_KEYWORDS = [
    "capex", "capital expenditure", "capital spending", "data center",
    "datacenter", "data-center", "ai infrastructure", "gpu",
    "cloud spending", "hyperscaler",
]

# EODHD tags stories with every mentioned/related symbol, so a Corning story
# that discusses data centers gets tagged NVDA and would pass the keyword
# filter. The watchlist is hardcoded, so require the company itself in the
# headline to keep the thesis layer about the thesis names.
THESIS_TITLE_NAMES = {
    "MSFT": ["microsoft", "msft"],
    "GOOGL": ["google", "alphabet", "googl"],
    "AMZN": ["amazon", "amzn"],
    "META": ["meta"],
    "NVDA": ["nvidia", "nvda"],
}


@st.cache_data(show_spinner=False, ttl=900)
def _news_raw(symbol, days, limit):
    """Raw EODHD news items for one full symbol (e.g. 'SNDK.US'), newest
    first. None = endpoint unavailable (vs [] = genuinely no news)."""
    start = datetime.now().date() - timedelta(days=days)
    url = (f"https://eodhd.com/api/news?api_token={EODHD_API_KEY}"
           f"&s={symbol}&from={start}&to={datetime.now().date()}"
           f"&limit={limit}&fmt=json")
    try:
        resp = requests.get(url, timeout=30)
        if resp.status_code != 200:
            return None
        data = resp.json()
        return data if isinstance(data, list) else None
    except Exception:
        return None


def _title_tokens(title):
    return set(re.sub(r"[^a-z0-9 ]", " ", str(title).lower()).split())


def _domain(link):
    host = urlparse(str(link)).netloc.lower()
    return host[4:] if host.startswith("www.") else host


def _rel_time(date_str):
    """'4h ago' style age for an ISO news timestamp; '' if unparseable."""
    try:
        ts = pd.Timestamp(date_str)
        now = (pd.Timestamp.now(tz="UTC") if ts.tzinfo is not None
               else pd.Timestamp.now())
        mins = max(0, int((now - ts).total_seconds() // 60))
    except (ValueError, TypeError):
        return ""
    if mins < 60:
        return f"{mins}m ago"
    if mins < 48 * 60:
        return f"{mins // 60}h ago"
    return f"{mins // (24 * 60)}d ago"


def _clean_news(raw, tkr):
    """Apply the anti-noise rules to raw items for one ticker: must be tagged
    with the ticker, must link out, not a many-symbol market wrap, syndicated
    near-duplicates collapsed (newest kept). Returns display-ready dicts."""
    kept, seen_tokens = [], []
    for it in raw:
        title, link = str(it.get("title", "")).strip(), str(it.get("link", "")).strip()
        if not title or not link.startswith("http"):
            continue
        symbols = [str(s).upper() for s in (it.get("symbols") or [])]
        if not any(s == f"{tkr}.US" or s == tkr for s in symbols):
            continue
        if len(symbols) > NEWS_WRAP_SYMBOLS:
            continue
        toks = _title_tokens(title)
        if not toks or any(len(toks & prev) / len(toks | prev) >= NEWS_DUP_JACCARD
                           for prev in seen_tokens):
            continue
        seen_tokens.append(toks)
        pol = (it.get("sentiment") or {}).get("polarity")
        kept.append({
            "title": title, "link": link, "domain": _domain(link),
            "when": _rel_time(it.get("date")),
            "sent": f"{pol:+.2f}" if isinstance(pol, (int, float)) else "—",
            "content": str(it.get("content", "")),
        })
    return kept


def position_news(tkr):
    """Recent ticker-matched news for one position, capped and de-noised.
    None = news endpoint unavailable."""
    raw = _news_raw(f"{tkr}.US", NEWS_DAYS, 20)
    if raw is None:
        return None
    return _clean_news(raw, tkr)[:NEWS_PER_POSITION]


def thesis_news(tkr):
    """Capex/AI-infrastructure-relevant news only, for the thesis watchlist.
    Keyword-matched on title + content; None = unavailable."""
    raw = _news_raw(f"{tkr}.US", THESIS_NEWS_DAYS, 50)
    if raw is None:
        return None
    names = THESIS_TITLE_NAMES.get(tkr, [tkr.lower()])
    out = []
    for it in _clean_news(raw, tkr):
        title = it["title"].lower()
        text = f"{title} {it['content']}".lower()
        if (any(n in title for n in names)
                and any(k in text for k in THESIS_KEYWORDS)):
            out.append(it)
        if len(out) >= NEWS_PER_THESIS:
            break
    return out


def render_news_items(items):
    for it in items:
        st.markdown(f"[{it['title']}]({it['link']})")
        st.caption(f"{it['domain']} · {it['when']} · sent {it['sent']}")


def explain_move(tkr, quotes, spy_chg):
    """Today's move vs SPY vs the sector ETF, with a one-word verdict:
    MARKET (within ~1pp of SPY), SECTOR (tracks its ETF but not SPY),
    SPECIFIC (diverges from both). None = no quote for the ticker."""
    q = quotes.get(f"{tkr}.US")
    if q is None:
        return None
    etf = SECTOR_ETF.get(get_sector(tkr))
    etf_q = quotes.get(f"{etf}.US") if etf else None
    move = q["chg_pct"]
    if spy_chg is not None and abs(move - spy_chg) <= MOVE_MATCH_PP:
        verdict = "MARKET"
    elif etf_q is not None and abs(move - etf_q["chg_pct"]) <= MOVE_MATCH_PP:
        verdict = "SECTOR"
    else:
        verdict = "SPECIFIC"
    return {"move": move, "etf": etf,
            "etf_move": etf_q["chg_pct"] if etf_q else None, "verdict": verdict}


def _pct_md(v):
    """Red/green colored percent for markdown cells."""
    if v is None:
        return "—"
    color = "green" if v > 0 else "red" if v < 0 else "gray"
    return f":{color}[{v:+.2f}%]"


def _verdict_md(verdict):
    if verdict == "SPECIFIC":
        return ":violet[**SPECIFIC**]"
    if verdict == "SECTOR":
        return ":blue[SECTOR]"
    return "MARKET"


def _earnings_md(date_str):
    """Earnings cell: loud orange flag inside the soon-window, plain date
    otherwise, em-dash when the calendar has nothing scheduled."""
    if not date_str:
        return "—"
    try:
        d = datetime.strptime(date_str, "%Y-%m-%d").date()
    except (ValueError, TypeError):
        return "—"
    days = (d - datetime.now().date()).days
    label = f"{d.strftime('%b')} {d.day}"
    if 0 <= days <= EARNINGS_SOON_DAYS:
        when = "today" if days == 0 else f"in {days}d"
        return f":orange[**⚠ earnings {when} ({label})**]"
    return label if days > 0 else "—"


def render_awareness_rows(tickers, quotes, spy_chg, earn_dates, tags=None,
                          news=None, suppress_auto=None):
    """One row per ticker: move vs SPY vs sector ETF, verdict, next earnings.
    Sorted by absolute move, biggest first; quoteless tickers sink to the end.
    With `news` ({ticker: items|None}), each row gets an expander of headlines
    — auto-opened when the ticker is on a big SPECIFIC move, so the likely
    'why' is readable without a click. Tickers in `suppress_auto` stay
    collapsed (their news is already open in the recently-reported digest)."""
    rows = [(t, explain_move(t, quotes, spy_chg)) for t in tickers]
    rows.sort(key=lambda r: abs(r[1]["move"]) if r[1] else -1, reverse=True)

    widths = [1.0, 1.0, 1.0, 1.5, 1.2, 2.3]
    hdr = st.columns(widths)
    for col, name in zip(hdr, ["Ticker", "Move", "SPY", "Sector ETF",
                               "Verdict", "Next earnings"]):
        col.markdown(f"**{name}**")
    for t, ex in rows:
        c = st.columns(widths)
        c[0].markdown(f"**{t}**")
        if tags and tags.get(t):
            c[0].caption(tags[t])
        if ex is None:
            c[1].write("unavailable")
        else:
            c[1].markdown(_pct_md(ex["move"]))
            c[2].markdown(_pct_md(spy_chg))
            if ex["etf"]:
                c[3].markdown(f"{ex['etf']} {_pct_md(ex['etf_move'])}")
            else:
                c[3].write("—")
            c[4].markdown(_verdict_md(ex["verdict"]))
        c[5].markdown(_earnings_md(earn_dates.get(t)))

        if news is None:
            continue
        items = news.get(t)
        if items is None:
            st.caption("news unavailable")
        elif items:
            big_specific = (ex is not None and ex["verdict"] == "SPECIFIC"
                            and abs(ex["move"]) >= AUTO_EXPAND_MOVE
                            and not (suppress_auto and t in suppress_auto))
            label = f"{t} news ({len(items)})"
            if big_specific:
                label += f" — {ex['move']:+.1f}% SPECIFIC move"
            with st.expander(label, expanded=big_specific):
                render_news_items(items)


# ---------- UI ----------
reg = current_regime()
regime_banner(reg)

# --- Data freshness ---
prices_check = load_prices()
latest_bar = prices_check.index.max()
days_old = (pd.Timestamp.now() - latest_bar).days
if days_old > 3:
    st.error(f"Price data is {days_old} days old (latest bar: {latest_bar.date()}). "
             f"Refresh before trusting signals.")
else:
    st.caption(f"Price data current through {latest_bar.date()}.")

rc1, rc2 = st.columns(2)
if rc1.button("Refresh S&P 500 prices (~5 min)"):
    status = st.status("Refreshing full universe...")
    n = refresh_prices(current_members(), status)
    status.update(label=f"Refreshed {n} tickers. Rerun your scan.", state="complete")
if rc2.button("Quick refresh: my holdings"):
    mine = ["CEG", "TLN", "VRT", "SMR", "JPM", "SOFI", "GLD", "KLAR"]
    status = st.status("Refreshing holdings...")
    n = refresh_prices(mine, status)
    status.update(label=f"Refreshed {n} tickers.", state="complete")

st.sidebar.header("Factor Weights")
# Defaults are set automatically from the current regime (see get_regime_weights
# in screener.py). The sliders below are a manual override — move them and the
# scan uses your values instead of the regime defaults.
dw = reg["weights"]
st.sidebar.caption(f"Regime default: **{reg['regime']}** — weights auto-set from "
                   f"the multi-period backtest. Sliders override.")
w_mom = st.sidebar.slider("Momentum", 0.0, 1.0, dw["momentum"], 0.05)
w_qual = st.sidebar.slider("Quality (F-score)", 0.0, 1.0, dw["quality"], 0.05)
w_val = st.sidebar.slider("Value (earnings yield)", 0.0, 1.0, dw["value"], 0.05)
w_trend = st.sidebar.slider("Trend (vs 200d)", 0.0, 1.0, dw["trend"], 0.05)

# Tell the user which weight set is actually driving the scan.
active = {"momentum": w_mom, "quality": w_qual, "value": w_val, "trend": w_trend}
if active == dw:
    st.sidebar.success(f"Using {reg['regime']} regime defaults (unmodified).")
else:
    st.sidebar.info("MANUAL OVERRIDE active — sliders differ from regime defaults.")

top_n = st.sidebar.number_input("Show top N", 5, 50, 20)

# Post-ranking selection filter: spread the top N across sectors instead of
# letting one hot sector (e.g. semis) dominate. OFF = pure composite rank.
diversify = st.sidebar.checkbox("Diversify by sector (max N per sector)", value=False,
                                help="Walk the composite-ranked list and skip any "
                                     "stock whose sector already has N picks, so you "
                                     "get the best name per sector, not N of the same trade.")
sector_cap = st.sidebar.number_input("Max per sector", 1, 10, 2, disabled=not diversify)
if diversify:
    st.sidebar.info(f"Sector diversification ON — max {int(sector_cap)} per sector.")

(tab_scan, tab_flags, tab_portfolio, tab_myport, tab_search,
 tab_ideas, tab_market) = st.tabs(["Full S&P Scan", "Flags", "Portfolio",
                                   "My Portfolio", "Search Any Ticker", "Ideas",
                                   "Market Awareness"])

with tab_scan:
    if st.button("Scan Full S&P 500", type="primary"):
        with st.spinner("Scanning ~500 stocks from cached data..."):
            st.session_state["scan"] = full_scan(w_mom, w_qual, w_val, w_trend,
                                                 top_n, diversify, sector_cap)
    if "scan" in st.session_state:
        df = st.session_state["scan"]
        if diversify:
            st.subheader(f"Top {len(df)} — composite rank, diversified "
                         f"(max {int(sector_cap)} per sector)")
        else:
            st.subheader(f"Top {len(df)} — ranked by your composite")
        show_table(df)
        deep_dive(df, "scan", reg)
    else:
        st.info("Click Scan Full S&P 500 to rank the whole index by your weights.")

with tab_flags:
    st.subheader("Proactive Flags")
    st.caption("Each run refreshes prices, rescans the whole S&P 500 with the "
               "current regime weights, saves a dated snapshot, and diffs it "
               "against the most recent prior snapshot to surface opportunities.")

    # --- Last-scan indicator ---
    _snap_date = latest_snapshot_date()
    _snap_age = days_since(_snap_date) if _snap_date else None
    if _snap_date is None:
        st.info("No scan snapshots saved yet — run a scan to start the history.")
    elif _snap_age is not None and _snap_age > SCAN_STALE_DAYS:
        st.warning(f"Consider running a scan — last snapshot {_snap_date} "
                   f"({_snap_age} days ago).")
    else:
        st.caption(f"Last scanned {_snap_age} day(s) ago "
                   f"(snapshot {_snap_date}).")

    if st.button("Run flag scan (refreshes prices — ~5 min)", type="primary"):
        today_str = datetime.now().strftime("%Y-%m-%d")
        status = st.status("Refreshing S&P 500 prices before scanning...")
        refresh_prices(current_members(), status)
        status.update(label="Prices refreshed — scoring universe with regime weights...")
        # Always score with the CURRENT regime's weights (not the sliders).
        rw = reg["weights"]
        scored = scan_universe(rw["momentum"], rw["quality"], rw["value"], rw["trend"])
        # Capture the comparison snapshot BEFORE writing today's (strict date <).
        pdate, pdf = prior_snapshot(today_str)
        save_snapshot(scored, today_str, reg["regime"])
        status.update(label=f"Scan complete — {len(scored)} stocks ranked.",
                      state="complete")
        st.session_state["flag_scan"] = {
            "scored": scored, "prior_date": pdate, "prior_df": pdf,
            "date": today_str, "regime": reg["regime"], "weights": rw}

    if "flag_scan" not in st.session_state:
        st.info("Click Run flag scan to detect new entrants, trend confirmations, "
                "big movers, and drop-outs versus your last scan.")
    else:
        fs = st.session_state["flag_scan"]
        scored, pdf, pdate = fs["scored"], fs["prior_df"], fs["prior_date"]
        w = fs["weights"]
        weight_str = (f"momentum {w['momentum']:.2f} · quality {w['quality']:.2f} · "
                      f"value {w['value']:.2f} · trend {w['trend']:.2f}")

        h1, h2 = st.columns(2)
        h1.markdown(f"**Regime:** {fs['regime']}  \n**Active weights:** {weight_str}")
        cmp_txt = pdate if pdate else "— (no prior scan)"
        h2.markdown(f"**This scan:** {fs['date']}  \n**Compared against:** {cmp_txt}")

        if diversify:
            st.caption(f"Sector diversification ON — top-N membership below is the "
                       f"best {int(sector_cap)} name(s) per sector.")
        if pdf is None:
            st.success("Baseline saved — flags appear on next scan.")
            st.markdown("**Today's top 20 (baseline)**")
            show_table(select_top(scored, 20, diversify, sector_cap))
        else:
            prev_regime = str(pdf["regime"].iloc[0]) if "regime" in pdf.columns and len(pdf) else None
            if prev_regime and prev_regime != fs["regime"]:
                st.warning(f"Regime changed since last scan ({prev_regime} → {fs['regime']}). "
                           f"Composite deltas below span two different weight sets.")
            flags = compute_flags(scored, pdf, pdate, top_n=20,
                                  diversify=diversify, cap=sector_cap)
            render_flags("New Entrants", flags["new"], reg, "flag_new")
            render_flags("Trend Confirmations", flags["trend"], reg, "flag_trend")
            render_flags("Big Movers", flags["movers"], reg, "flag_mover")
            render_dropped(flags["dropped"])

def render_track_detail(rebs, prices_all):
    """Per-track holdings (P&L since last rebalance) + rebalance history."""
    ldate = rebs[-1]["date"]
    st.markdown(f"**Current holdings — since last rebalance ({ldate})**")
    last = rebs[-1]
    hrows = []
    for tk, entry, wt in zip(last["tickers"], last["entry_prices"], last["weights"]):
        cur = None
        if tk in prices_all.columns:
            s = prices_all[tk].dropna()
            if len(s) > 0:
                cur = float(s.iloc[-1])
        pnl = ((cur / entry) - 1) if (cur is not None and entry) else None
        hrows.append({
            "Ticker": tk,
            "Entry": entry if entry else None,
            "Current": cur,
            # percentage points, not fractions: NumberColumn's printf format
            # renders the % sign, so the stored value must already be scaled.
            "P&L": pnl * 100 if pnl is not None else None,
            "Weight": wt * 100 if wt is not None else None,
        })
    # Keep these columns numeric rather than preformatted strings: a string P&L
    # sorts lexicographically, which ranks "-3.8%" above "-2.5%" and pushes every
    # gain ("+…") below every loss ("-…"). Sorted best -> worst, unpriced last.
    hdf = pd.DataFrame(hrows).sort_values("P&L", ascending=False, na_position="last")
    st.dataframe(
        hdf, use_container_width=True, hide_index=True,
        column_config={
            "Entry": st.column_config.NumberColumn("Entry", format="$%.2f"),
            "Current": st.column_config.NumberColumn("Current", format="$%.2f"),
            "P&L": st.column_config.NumberColumn("P&L", format="%+.1f%%"),
            "Weight": st.column_config.NumberColumn("Weight", format="%.1f%%"),
        },
    )

    st.markdown("**Rebalance history**")
    hist = []
    for r in reversed(rebs):
        tickers = r["tickers"]
        preview = ", ".join(tickers[:8]) + (" …" if len(tickers) > 8 else "")
        hist.append({"Date": r["date"], "Regime": r["regime"],
                     "Holdings": len(tickers), "Tickers": preview})
    st.dataframe(pd.DataFrame(hist), use_container_width=True, hide_index=True)


with tab_portfolio:
    st.subheader("Paper Portfolios — concentrated vs diversified vs SPY")
    st.caption(f"Two equal-weight top-{PORTFOLIO_BASKET_N} baskets from the SAME "
               f"scan (current regime weights, winsorized), rebalanced monthly "
               f"from a shared inception at ${START_CAPITAL:,.0f}: "
               f"**Concentrated** (pure composite rank) vs **Diversified** "
               f"(max {TRACKS['diversified']['cap']} per sector). The live "
               f"head-to-head before real capital.")

    track_rebs = {t: parse_rebalances(t) for t in TRACKS}
    any_started = any(track_rebs[t] for t in TRACKS)

    # Both tracks rebalance together, so cadence is judged off the latest date
    # across either file.
    all_last = [r[-1]["date"] for r in track_rebs.values() if r]
    last_date = max(all_last) if all_last else None
    dsince = days_since(last_date) if last_date else None
    too_soon = (dsince is not None and dsince < REBALANCE_MIN_DAYS)

    # --- Status banner ---
    b1, b2, b3 = st.columns(3)
    b1.metric("Last rebalance", last_date if last_date else "—")
    b2.metric("Days since", str(dsince) if dsince is not None else "—")
    if last_date is None:
        b3.metric("Status", "No rebalance yet")
    elif dsince is not None and dsince >= REBALANCE_DUE_DAYS:
        b3.metric("Status", "Due for rebalance",
                  f"{dsince}d since last", delta_color="inverse")
    else:
        rem = REBALANCE_DUE_DAYS - dsince if dsince is not None else None
        b3.metric("Status", "Not yet due",
                  f"{rem} day(s) to go" if rem is not None else None,
                  delta_color="off")

    btn_label = "Rebalance Both" if any_started else "Start Both Portfolios"
    if st.button(btn_label, type="primary"):
        if too_soon:
            st.warning(f"Too soon — last rebalance was {dsince} days ago "
                       f"(monthly cadence expects ≥ {REBALANCE_MIN_DAYS} days). "
                       f"Wait {REBALANCE_MIN_DAYS - dsince} more day(s). "
                       f"Both portfolios unchanged.")
        else:
            with st.spinner("Scoring the universe once and selecting both baskets..."):
                rw = reg["weights"]
                scored = scan_universe(rw["momentum"], rw["quality"],
                                       rw["value"], rw["trend"])
                today_str = datetime.now().strftime("%Y-%m-%d")
                counts = {}
                for t, cfg in TRACKS.items():
                    sel = select_top(scored, PORTFOLIO_BASKET_N,
                                     cfg["diversify"], cfg["cap"])
                    save_rebalance(t, today_str, sel, reg["regime"])
                    counts[t] = len(sel)
            st.success(f"Rebalanced both tracks on {today_str}: "
                       + " · ".join(f"{TRACKS[t]['label']} {n}"
                                    for t, n in counts.items())
                       + " equal-weight holdings. Curves updated below.")
            track_rebs = {t: parse_rebalances(t) for t in TRACKS}
            any_started = True
            last_date = today_str

    # --- Empty state ---
    if not any_started:
        st.info("No rebalances yet. Click **Start Both Portfolios** to begin the "
                "head-to-head — both baskets stamp the same inception.")
    else:
        # A track with no ledger silently drops out of the chart, tiles and
        # metrics below — say so, rather than leaving a one-sided head-to-head
        # that looks like a rendering bug.
        no_ledger = [TRACKS[t]["label"] for t in TRACKS if not track_rebs[t]]
        if no_ledger:
            st.warning(f"No rebalance ledger yet for: {', '.join(no_ledger)} — "
                       f"that track is missing from everything below. Click "
                       f"**Rebalance Both** to bring it in line.")

        prices_all = load_prices()
        curves = {}
        for t in TRACKS:
            if track_rebs[t]:
                c = build_strategy_curve(track_rebs[t], prices_all)
                if c is not None:
                    curves[t] = c

        if not curves or all(len(c) < 2 for c in curves.values()):
            st.warning("Not enough price history yet to build the equity curves — "
                       "check back after the next trading day.")
        else:
            # Shared timeline: union of every track's dates, SPY rebased on it.
            combined = None
            for c in curves.values():
                combined = c.index if combined is None else combined.union(c.index)
            combined = combined.sort_values()
            spy = build_spy_curve(combined)

            # --- Equity curves: both tracks + SPY on one chart ---
            st.markdown("#### Equity Curves — from inception")
            chart = pd.DataFrame(index=combined)
            for t, c in curves.items():
                chart[TRACKS[t]["label"]] = c.reindex(combined)
            if spy is not None:
                chart["SPY"] = spy.reindex(combined)
            st.line_chart(chart)

            tiles = st.columns(len(curves) + (1 if spy is not None else 0))
            i = 0
            for t, c in curves.items():
                v = float(c.iloc[-1])
                tiles[i].metric(TRACKS[t]["label"], f"${v:,.0f}",
                                f"{v / START_CAPITAL - 1:+.1%}")
                i += 1
            if spy is not None:
                sv = float(spy.dropna().iloc[-1])
                tiles[i].metric("SPY", f"${sv:,.0f}",
                                f"{sv / START_CAPITAL - 1:+.1%}")

            # --- Metrics table: all three ---
            st.markdown("#### Performance Metrics")
            st.caption(f"Inception {combined[0].date()} → {combined[-1].date()}. "
                       f"Annualized by calendar time; volatility from daily returns. "
                       f"Short track records are noisy — read with care.")
            pairs = [(TRACKS[t]["label"], portfolio_metrics(curves[t]))
                     for t in curves]
            if spy is not None:
                pairs.append(("SPY", portfolio_metrics(spy)))
            st.table(metrics_table(pairs))

            # --- Per-track holdings + history ---
            for t in TRACKS:
                if track_rebs[t]:
                    with st.expander(f"{TRACKS[t]['label']} — holdings & history"):
                        render_track_detail(track_rebs[t], prices_all)


# ---------- My Portfolio (Schwab, read-only) ----------
def schwab_screener_take(tkr, weights, uni_raw, prices_all):
    """The screener's current take on one real holding: composite scored
    against today's universe, F-score, momentum, 200d-MA position. Returns
    None when the screener has nothing on it (non-equity, no/short history)."""
    try:
        r = compute_factors(tkr, prices_all)
    except Exception:
        return None
    if r is None:
        return None
    try:
        comp = score_against_universe(r, weights, uni_raw)
    except Exception:
        comp = None
    return {"comp": comp, "fscore": r.get("fscore"),
            "mom": r.get("momentum"), "trend": r.get("trend")}


def _pnl_css(v):
    """Red/green text for P&L cells — same palette as the Ideas tab."""
    if pd.isna(v):
        return "color: gray"
    return ("color: #16a34a" if v > 0            # green
            else "color: #dc2626" if v < 0       # red
            else "color: gray")


def _zebra_css(row):
    """Subtle alternate-row shading; theme-safe via low-alpha gray."""
    shade = "background-color: rgba(128,128,128,0.07)"
    return [shade if row.name % 2 else ""] * len(row)


with tab_myport:
    st.subheader("My Portfolio — live Schwab holdings (read-only)")
    st.caption("Positions and balances pulled from Schwab with GET-only calls "
               "(no order capability exists in this integration). Each holding "
               "is cross-referenced against the screener's CURRENT factor view "
               "using the sidebar weights — 'I own it; what does the model "
               "think of it today?'")

    if st.button("Load / refresh from Schwab", type="primary"):
        try:
            with st.spinner("Fetching accounts, balances, and positions..."):
                st.session_state["schwab"] = {
                    "balances": get_balances(),
                    "positions": get_positions(),
                    "fetched": datetime.now().strftime("%Y-%m-%d %H:%M"),
                }
            st.session_state.pop("schwab_err", None)
        except SchwabAuthError as e:
            st.session_state.pop("schwab", None)
            st.session_state["schwab_err"] = str(e)
        except Exception as e:
            st.session_state.pop("schwab", None)
            st.session_state["schwab_err"] = f"Unexpected error talking to Schwab: {e}"

    if "schwab_err" in st.session_state:
        st.error(f"**Reconnect to Schwab needed.** {st.session_state['schwab_err']}")
        st.markdown("In a terminal (not here — the login opens a browser):")
        st.code("python schwab_client.py login", language="bash")
    elif "schwab" not in st.session_state:
        st.info("Click **Load / refresh from Schwab** to pull your live "
                "positions. First time? Connect your account from a terminal "
                "first with `python schwab_client.py login`.")
    else:
        snap = st.session_state["schwab"]
        positions = snap["positions"]
        opt_pos = [p for p in positions if p["asset_type"] == "OPTION"]
        eq_pos = [p for p in positions if p["asset_type"] != "OPTION"]

        # --- Score stock/fund holdings against the screener ---
        eq_rows = []
        if eq_pos:
            weights = (w_mom, w_qual, w_val, w_trend)
            with st.spinner("Scoring holdings against today's universe..."):
                uni = universe_factor_frame()
                members = set(current_members())
                prices_all = load_prices()
                for p in eq_pos:
                    tkr = p["ticker"]
                    take = None
                    # Only chart-able security types get a screener view; cash
                    # sweeps, bonds etc. show as N/A.
                    if p["asset_type"] in ("EQUITY", "COLLECTIVE_INVESTMENT",
                                           "ETF", "INDEX"):
                        take = schwab_screener_take(tkr, weights, uni, prices_all)
                    trend = take.get("trend") if take else None
                    eq_rows.append({
                        "Ticker": tkr,
                        "Acct": p["account"],
                        "Qty": p["quantity"],
                        "Mkt Value": p["market_value"],
                        "Cost Basis": p["cost_basis"],
                        "Unrl P&L": p["unrealized_pnl"],
                        "Unrl P&L %": (p["unrealized_pnl_pct"] * 100
                                       if p["unrealized_pnl_pct"] is not None else None),
                        "Composite": take.get("comp") if take else None,
                        "F-Score": take.get("fscore") if take else None,
                        "Momentum %": (take["mom"] * 100
                                       if take and pd.notna(take.get("mom")) else None),
                        "200d MA": ("Above" if trend is not None and trend > 0
                                    else "Below" if trend is not None else "N/A"),
                        "In S&P 500": "Yes" if tkr in members else "No",
                    })

        # --- Summary header: totals across stocks AND options ---
        mv_sum = sum(p["market_value"] for p in positions
                     if p["market_value"] is not None)
        pnl_sum = sum(p["unrealized_pnl"] for p in positions
                      if p["unrealized_pnl"] is not None)
        basis_sum = sum(p["cost_basis"] for p in positions
                        if p["cost_basis"] is not None)
        pnl_pct = (pnl_sum / abs(basis_sum)) if basis_sum else None
        pcolor = ("green" if pnl_sum > 0 else "red" if pnl_sum < 0 else "gray")
        pnl_str = f"${pnl_sum:+,.2f}" + (f" ({pnl_pct:+.1%})"
                                         if pnl_pct is not None else "")
        s1, s2, s3 = st.columns([1.4, 1.8, 0.8])
        s1.caption("Total market value")
        s1.markdown(f"### ${mv_sum:,.2f}")
        s2.caption("Total unrealized P&L")
        s2.markdown(f"### :{pcolor}[{pnl_str}]")
        s3.caption("Positions")
        s3.markdown(f"### {len(eq_pos)}"
                    + (f" +{len(opt_pos)} opt" if opt_pos else ""))

        # One-line portfolio-vs-screener readout.
        scoreable = [r for r in eq_rows if r["200d MA"] in ("Above", "Below")]
        if scoreable:
            above = sum(1 for r in scoreable if r["200d MA"] == "Above")
            weak = sorted(r["Ticker"] for r in eq_rows
                          if r["Composite"] is not None and r["Composite"] < 0)
            msg = (f"**Screener check:** {above} of {len(scoreable)} scoreable "
                   f"holdings above their 200d MA, {len(scoreable) - above} below")
            msg += (f" · :red[screener-weak (composite < 0): {', '.join(weak)}]"
                    if weak else " · none score composite < 0")
            st.markdown(msg)
        st.caption(f"Fetched {snap['fetched']} — click the button above to refresh.")

        # --- Accounts: one compact table instead of stacked metric rows ---
        st.markdown("**Accounts**")
        bdf = pd.DataFrame([{
            "Account": f"...{b['account']}" + (f" ({b['type']})" if b["type"] else ""),
            "Total value": b["total_value"],
            "Cash": b["cash"],
            "Buying power": b["buying_power"],
        } for b in snap["balances"]])
        st.dataframe(
            bdf, use_container_width=True, hide_index=True,
            column_config={
                "Total value": st.column_config.NumberColumn("Total value", format="$%.2f"),
                "Cash": st.column_config.NumberColumn("Cash", format="$%.2f"),
                "Buying power": st.column_config.NumberColumn("Buying power", format="$%.2f"),
            })

        # --- Stock/fund holdings x screener ---
        st.markdown("**Holdings**")
        if not eq_rows:
            st.info("No stock/fund positions found in the connected account(s).")
        else:
            hdf = (pd.DataFrame(eq_rows)
                   .sort_values("Mkt Value", ascending=False, na_position="last")
                   .reset_index(drop=True))
            sty = (hdf.style
                   .map(_pnl_css, subset=["Unrl P&L", "Unrl P&L %"])
                   .apply(_zebra_css, axis=1))
            st.dataframe(
                sty, use_container_width=True, hide_index=True,
                column_config={
                    "Ticker": st.column_config.TextColumn("Ticker", width="small"),
                    "Acct": st.column_config.TextColumn("Acct", width="small"),
                    "Qty": st.column_config.NumberColumn("Qty", format="%.2f", width="small"),
                    "Mkt Value": st.column_config.NumberColumn("Mkt Value", format="$%.2f"),
                    "Cost Basis": st.column_config.NumberColumn("Cost Basis", format="$%.2f"),
                    "Unrl P&L": st.column_config.NumberColumn("Unrl P&L", format="$%.2f"),
                    "Unrl P&L %": st.column_config.NumberColumn("Unrl P&L %", format="%+.1f%%"),
                    "Composite": st.column_config.NumberColumn("Composite", format="%.2f"),
                    "F-Score": st.column_config.NumberColumn("F-Score", format="%.0f", width="small"),
                    "Momentum %": st.column_config.NumberColumn("Momentum 12-1", format="%+.1f%%"),
                    "200d MA": st.column_config.TextColumn("200d MA", width="small"),
                    "In S&P 500": st.column_config.TextColumn("In S&P 500", width="small"),
                },
            )
            st.caption("Composite / F-Score / Momentum / 200d MA are the "
                       "screener's current view under the sidebar weights "
                       "(blank or N/A = not scoreable: not enough history, "
                       "non-equity, or no fundamentals). 'In S&P 500' flags "
                       "whether the name is inside the scan universe.")

        # --- Options: separate section, never scored by the screener ---
        if opt_pos:
            st.markdown("**Options**")
            odf = (pd.DataFrame([{
                "Option": p.get("description") or p["ticker"],
                "Underlying": p.get("underlying", ""),
                "Acct": p["account"],
                "Qty": p["quantity"],
                "Mkt Value": p["market_value"],
                "Cost Basis": p["cost_basis"],
                "Unrl P&L": p["unrealized_pnl"],
                "Unrl P&L %": (p["unrealized_pnl_pct"] * 100
                               if p["unrealized_pnl_pct"] is not None else None),
            } for p in opt_pos])
                .sort_values("Mkt Value", ascending=False, na_position="last")
                .reset_index(drop=True))
            osty = (odf.style
                    .map(_pnl_css, subset=["Unrl P&L", "Unrl P&L %"])
                    .apply(_zebra_css, axis=1))
            st.dataframe(
                osty, use_container_width=True, hide_index=True,
                column_config={
                    "Qty": st.column_config.NumberColumn("Qty", format="%.0f", width="small"),
                    "Mkt Value": st.column_config.NumberColumn("Mkt Value", format="$%.2f"),
                    "Cost Basis": st.column_config.NumberColumn("Cost Basis", format="$%.2f"),
                    "Unrl P&L": st.column_config.NumberColumn("Unrl P&L", format="$%.2f"),
                    "Unrl P&L %": st.column_config.NumberColumn("Unrl P&L %", format="%+.1f%%"),
                },
            )
            st.caption("Cost basis and P&L include the 100× contract "
                       "multiplier (matched to Schwab's own open P&L). "
                       "Options aren't scoreable by the screener — no "
                       "composite/F-score/momentum columns apply.")


with tab_search:
    q = st.text_input("Enter any US ticker(s), comma-separated",
                      placeholder="CEG, PLTR, SMR, VRT")
    if st.button("Look up"):
        want = [t.strip().upper() for t in q.split(",") if t.strip()]
        prices_all = load_prices()
        rows, missing = [], []
        with st.spinner(f"Pulling {len(want)} ticker(s)..."):
            for t in want:
                r = compute_factors(t, prices_all)
                if r is not None:
                    rows.append(r)
                else:
                    missing.append(t)
        if missing:
            st.warning(f"No data found (bad ticker or <1yr history): {', '.join(missing)}")
        if rows:
            st.session_state["search"] = add_zscores(pd.DataFrame(rows),
                                                     w_mom, w_qual, w_val, w_trend)
    if "search" in st.session_state:
        sdf = st.session_state["search"]
        show_table(sdf)
        deep_dive(sdf, "search", reg)

with tab_ideas:
    st.subheader("Idea Tracker — live P&L")
    st.caption("Open ideas are priced cache-first from EODHD; the composite is "
               "recomputed against today's universe to check if the thesis held.")
    log = load_trade_log()
    if log.empty:
        st.info("No ideas logged yet. Log one from the Deep Dive on the other tabs.")
    else:
        entry_num = pd.to_numeric(log["entry_price"], errors="coerce")
        close_num = pd.to_numeric(log["close_price"], errors="coerce")
        is_open = log["status"].astype(str).str.lower() == "open"
        open_ideas = log[is_open]
        closed = log[~is_open]

        # ---- summary stats ----
        cpnl = ((close_num - entry_num) / entry_num)[~is_open].dropna()
        n_closed = int((~is_open).sum())
        s1, s2, s3, s4 = st.columns(4)
        s1.metric("Open ideas", int(is_open.sum()))
        s2.metric("Closed", n_closed)
        s3.metric("Win rate (closed)", f"{(cpnl > 0).mean():.0%}" if len(cpnl) else "—")
        s4.metric("Avg P&L (closed)", f"{cpnl.mean():+.1%}" if len(cpnl) else "—")

        # ---- open ideas ----
        if open_ideas.empty:
            st.caption("No open ideas.")
        else:
            weights = (w_mom, w_qual, w_val, w_trend)
            with st.spinner("Pricing open ideas and recomputing factors..."):
                uni = universe_factor_frame()
            widths = [0.7, 0.5, 1.0, 1.0, 0.8, 0.9, 1.4, 0.8, 1.3, 0.7]
            hdr = st.columns(widths)
            for col, name in zip(hdr, ["Ticker", "Days", "Entry", "Current", "P&L",
                                       "→Target", "Comp then→now", "Abv200d",
                                       "Outcome note", ""]):
                col.markdown(f"**{name}**")

            for idx, idea in open_ideas.iterrows():
                tkr = str(idea["ticker"]).strip()
                live = idea_live(tkr, weights, uni)
                cur, comp_now = live["price"], live["comp_now"]
                entry = entry_num.get(idx)

                try:
                    days = (pd.Timestamp.now().normalize()
                            - pd.to_datetime(idea["date"]).normalize()).days
                except Exception:
                    days = None
                pnl = ((cur - entry) / entry
                       if cur is not None and pd.notna(entry) and entry else None)
                tgt = pd.to_numeric(pd.Series([idea["target_price"]]),
                                    errors="coerce").iloc[0]
                dist = (tgt / cur - 1) if cur and pd.notna(tgt) and tgt > 0 else None
                comp_then = pd.to_numeric(pd.Series([idea["composite"]]),
                                          errors="coerce").iloc[0]

                rc = st.columns(widths)
                rc[0].write(tkr)
                rc[1].write(str(days) if days is not None else "—")
                rc[2].write(f"${entry:.2f}" if pd.notna(entry) else "—")
                rc[3].write(f"${cur:.2f}" if cur is not None else "—")
                if pnl is None:
                    rc[4].write("—")
                else:
                    pcolor = "green" if pnl > 0 else "red" if pnl < 0 else "gray"
                    rc[4].markdown(f":{pcolor}[**{pnl:+.1%}**]")
                rc[5].write(f"{dist:+.1%}" if dist is not None else "—")
                then_s = f"{comp_then:.2f}" if pd.notna(comp_then) else "—"
                now_s = f"{comp_now:.2f}" if comp_now is not None else "—"
                held = ""
                if pd.notna(comp_then) and comp_now is not None:
                    held = " ✅" if comp_now >= comp_then else " ⚠️"
                rc[6].write(f"{then_s} → {now_s}{held}")
                av = str(idea["above_200d"]).strip()
                rc[7].write(av if av else "—")
                note_val = rc[8].text_input("note", key=f"note_{idx}",
                                            label_visibility="collapsed")
                if rc[9].button("Close", key=f"close_{idx}"):
                    close_idea(idx, cur, note_val.strip())
                    st.success(f"Closed {tkr}.")
                    st.rerun()

        # ---- closed ideas ----
        if n_closed > 0:
            st.markdown("**Closed ideas**")
            sc = closed.copy()
            sc["pnl"] = ((close_num - entry_num) / entry_num)[~is_open].values

            def pnl_color(v):
                if pd.isna(v):
                    return "color: gray"
                return ("color: #16a34a" if v > 0            # green
                        else "color: #dc2626" if v < 0       # red
                        else "color: gray")

            st.dataframe(
                sc[["date", "ticker", "entry_price", "close_date", "close_price",
                    "pnl", "outcome_note"]].style.format(
                    {"entry_price": "${:.2f}", "close_price": "${:.2f}",
                     "pnl": "{:+.1%}"}, na_rep="—")
                    .map(pnl_color, subset=["pnl"]),
                use_container_width=True)


with tab_market:
    st.subheader("Market Awareness")
    st.caption("Context for discretionary judgment — dates, prices, yields, "
               "and ticker-matched headlines. **Display-only:** nothing on "
               "this tab feeds the screener, the flags, or the paper "
               "portfolios. Quotes delayed intraday, cached 5 min; news "
               "cached 15 min; calendars cached for hours. Headlines link to "
               "the original source; *sent* chips are EODHD's algorithmic "
               "polarity (−1…+1) — reference only, never a signal.")

    # --- Gather tickers first so one quote batch serves the whole tab ---
    _ideas_log = load_trade_log()
    idea_tickers = sorted(set(
        t for t in (str(x).strip().upper()
                    for x in _ideas_log[_ideas_log["status"] == "open"]["ticker"])
        if t))
    held_tickers = set()
    if "schwab" in st.session_state:
        for p in st.session_state["schwab"]["positions"]:
            if p["asset_type"] == "OPTION":
                tk = str(p.get("underlying") or "").strip().upper()
            elif p["asset_type"] in ("EQUITY", "ETF", "COLLECTIVE_INVESTMENT",
                                     "MUTUAL_FUND", "INDEX"):
                tk = str(p["ticker"]).strip().upper()
            else:
                tk = ""  # cash sweeps, bonds — no move to explain
            if tk:
                held_tickers.add(tk)
    position_tickers = sorted(held_tickers | set(idea_tickers))

    all_tickers = sorted(set(position_tickers) | set(THESIS_WATCHLIST))
    quote_syms = tuple(sorted(
        {f"{t}.US" for t in all_tickers}
        | {f"{e}.US" for e in SECTOR_ETF.values()}
        | {"SPY.US", "VIX.INDX"}))
    quotes = live_quotes(quote_syms)
    spy_q = quotes.get("SPY.US")
    spy_chg = spy_q["chg_pct"] if spy_q else None
    earn_dates = next_earnings(tuple(all_tickers))

    # --- 1. Macro weather strip ---
    mc1, mc2, mc3, mc4 = st.columns([3.4, 0.9, 0.9, 0.9])
    events = macro_calendar()
    if events is None:
        mc1.markdown("**Macro calendar:** unavailable")
    elif not events:
        mc1.markdown("**Macro calendar:** no FOMC / CPI / NFP / GDP releases "
                     "in the next 14 days")
    else:
        mc1.markdown("**Next 14d:** " + " · ".join(
            f"**{label}** {d.strftime('%b')} {d.day}" for label, d in events))
        mc1.caption("US high-importance releases (FOMC, CPI, Nonfarm "
                    "Payrolls, GDP) from the EODHD economic calendar.")
    y10 = treasury_10y()
    if y10:
        mc2.metric("10Y yield", f"{y10['level']:.2f}%",
                   f"{y10['chg_bp']:+.1f} bp", delta_color="off")
    else:
        mc2.metric("10Y yield", "—", "unavailable", delta_color="off")
    vix_q = quotes.get("VIX.INDX")
    if vix_q:
        mc3.metric("VIX", f"{vix_q['close']:.1f}",
                   f"{vix_q['chg_pct']:+.1f}%", delta_color="inverse")
    else:
        mc3.metric("VIX", "—", "unavailable", delta_color="off")
    if spy_q:
        mc4.metric("SPY", f"${spy_q['close']:.2f}",
                   f"{spy_chg:+.2f}%")
    else:
        mc4.metric("SPY", "—", "unavailable", delta_color="off")

    st.divider()

    # --- 2. My positions: catalysts & moves ---
    st.markdown("#### My Positions — catalysts & moves")
    if "schwab" not in st.session_state:
        bc1, bc2 = st.columns([1.3, 3.7])
        if bc1.button("Load Schwab positions", key="ma_schwab"):
            try:
                with st.spinner("Fetching positions..."):
                    st.session_state["schwab"] = {
                        "balances": get_balances(),
                        "positions": get_positions(),
                        "fetched": datetime.now().strftime("%Y-%m-%d %H:%M"),
                    }
                st.rerun()
            except SchwabAuthError as e:
                st.error(f"Schwab reconnect needed: {e} — run "
                         f"`python schwab_client.py login` in a terminal.")
            except Exception as e:
                st.error(f"Schwab unavailable: {e}")
        bc2.caption("Schwab not loaded — showing open Ideas tickers only. "
                    "Option positions count via their underlying.")
    if not position_tickers:
        st.info("No holdings or open ideas to track yet.")
    else:
        tags = {}
        for t in position_tickers:
            src = [s for s, ok in (("held", t in held_tickers),
                                   ("idea", t in idea_tickers)) if ok]
            tags[t] = " + ".join(src)
        with st.spinner("Pulling position news..."):
            pos_news = {t: position_news(t) for t in position_tickers}
        reported = recent_earnings(tuple(position_tickers))
        if reported:
            render_recent_earnings(reported, pos_news)
            st.markdown("")  # breathing room before the moves table
        render_awareness_rows(position_tickers, quotes, spy_chg,
                              earn_dates, tags, news=pos_news,
                              suppress_auto=set(reported))

    st.divider()

    # --- 3. Thesis watch ---
    st.markdown("#### Thesis Watch — hyperscalers / AI capex")
    st.caption("Hardcoded watchlist. These earnings calls (capex guidance) are "
               "the thesis catalysts for the whole book. News below is "
               "filtered to capex / data-center / AI-infrastructure "
               "relevance only — the thesis-break layer.")
    with st.spinner("Pulling thesis news..."):
        th_news = {t: thesis_news(t) for t in THESIS_WATCHLIST}
    render_awareness_rows(THESIS_WATCHLIST, quotes, spy_chg, earn_dates,
                          news=th_news)