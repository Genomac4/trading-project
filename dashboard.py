import os
import requests
import numpy as np
import pandas as pd
import streamlit as st
from datetime import datetime
from config import EODHD_API_KEY
from data_layer import get_valuation
from explore_constituents import get_sp500_membership
from backtest_composite import cached_fscore, cached_value, zscore
from pit_factors import pit_fscore, pit_value_inputs
from screener import get_regime_weights

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
    """Re-pull latest prices for a list of tickers from EODHD."""
    done = 0
    for i, t in enumerate(tickers, 1):
        s = live_prices(t)
        if s is not None:
            done += 1
        if i % 25 == 0:
            status.write(f"Refreshed {i}/{len(tickers)}...")
    load_prices.clear()
    return done


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
            "trend": tr, "fscore": fs, "earn_yield": ey, "_series": s}


def add_zscores(df, w_mom, w_qual, w_val, w_trend):
    n = len(df)
    def z(col):
        if n > 1:
            return zscore(df[col]).fillna(0)
        return pd.Series(0.0, index=df.index)
    df["z_mom"] = z("momentum")
    df["z_qual"] = z("fscore")
    df["z_val"] = z("earn_yield")
    df["z_trend"] = z("trend")
    df["composite"] = (w_mom * df["z_mom"] + w_qual * df["z_qual"]
                       + w_val * df["z_val"] + w_trend * df["z_trend"])
    return df.sort_values("composite", ascending=False).reset_index(drop=True)


# ---------- Regime ----------
def current_regime():
    """Single source of truth for the regime + its backtest-derived weights."""
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
        st.warning(f"MARKET REGIME: CHOPPY/DEFENSIVE — SPY {pct:+.1%} vs 200d MA. "
                   f"Active weight set (quality/trend-heavy, from backtest): {weight_str}.")


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


def full_scan(w_mom, w_qual, w_val, w_trend, top_n):
    return scan_universe(w_mom, w_qual, w_val, w_trend).head(top_n)


# ---------- Flagging engine (scan history + diffs) ----------
SNAPSHOT_COLS = ["date", "Ticker", "price", "momentum", "trend",
                 "fscore", "earn_yield", "composite", "rank", "regime"]


def save_snapshot(scored, date_str, regime):
    """Persist a ranked scan to data/scan_history/{date}.csv (overwrites same-day)."""
    os.makedirs(SCAN_HISTORY_DIR, exist_ok=True)
    snap = scored[["Ticker", "price", "momentum", "trend",
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


def compute_flags(scored, prior_df, prior_date, top_n=20):
    """Diff today's ranked scan against the prior snapshot.

    Returns a dict with four lists. New/trend/movers items are (row, reason)
    where row is the live scored Series (has z-scores for make_thesis and every
    field log_trade needs). Dropped items are (ticker, row_or_None, reason)."""
    by_tkr = scored.set_index("Ticker", drop=False)
    p = prior_df.set_index("Ticker")
    today_top = set(scored[scored["rank"] <= top_n]["Ticker"])
    prior_top = set(prior_df[prior_df["rank"] <= top_n]["Ticker"])

    # 1. NEW ENTRANTS — in today's top N, not in the prior top N.
    new_entrants = []
    for tk in scored[scored["rank"] <= top_n].sort_values("rank")["Ticker"]:
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
    widths = [0.8, 1.0, 3.0, 4.5, 1.4]
    for i, (row, reason) in enumerate(items):
        c = st.columns(widths)
        c[0].markdown(f"**{row['Ticker']}**")
        c[1].write(f"{row['composite']:.2f}")
        c[2].write(reason)
        c[3].caption(make_thesis(row))
        if c[4].button("Log this idea", key=f"{key}_{i}_{row['Ticker']}"):
            log_trade(row, reg, notes=f"Flagged ({title})")
            st.success(f"Logged {row['Ticker']} — see the Ideas tab.")


def render_dropped(items):
    """Dropped-out flag: ticker, composite now (if still scanned), reason. No log button."""
    st.markdown("#### Dropped Out")
    if not items:
        st.caption("Nothing flagged.")
        return
    widths = [0.8, 1.0, 6.0]
    for tk, row, reason in items:
        c = st.columns(widths)
        c[0].markdown(f"**{tk}**")
        c[1].write(f"{row['composite']:.2f}" if row is not None else "—")
        c[2].write(reason)


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
            st.caption(f"Analyst consensus: target ${target:.0f} ({upside:+.1%} vs current) | "
                       f"rating {rating:.2f}/5 "
                       f"(SB:{val.get('strong_buy')} B:{val.get('buy')} "
                       f"H:{val.get('hold')} S:{val.get('sell')})")

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
    show = df[["Ticker", "price", "composite", "momentum", "trend", "fscore", "earn_yield"]]
    st.dataframe(show.style.format({
        "price": "${:.2f}", "composite": "{:.2f}", "momentum": "{:+.1%}",
        "trend": "{:+.1%}", "fscore": "{:.0f}", "earn_yield": "{:.1%}",
    }, na_rep="—"), use_container_width=True)


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

tab_scan, tab_flags, tab_search, tab_ideas = st.tabs(
    ["Full S&P Scan", "Flags", "Search Any Ticker", "Ideas"])

with tab_scan:
    if st.button("Scan Full S&P 500", type="primary"):
        with st.spinner("Scanning ~500 stocks from cached data..."):
            st.session_state["scan"] = full_scan(w_mom, w_qual, w_val, w_trend, top_n)
    if "scan" in st.session_state:
        df = st.session_state["scan"]
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

        if pdf is None:
            st.success("Baseline saved — flags appear on next scan.")
            st.markdown("**Today's top 20 (baseline)**")
            show_table(scored.head(20))
        else:
            prev_regime = str(pdf["regime"].iloc[0]) if "regime" in pdf.columns and len(pdf) else None
            if prev_regime and prev_regime != fs["regime"]:
                st.warning(f"Regime changed since last scan ({prev_regime} → {fs['regime']}). "
                           f"Composite deltas below span two different weight sets.")
            flags = compute_flags(scored, pdf, pdate, top_n=20)
            render_flags("New Entrants", flags["new"], reg, "flag_new")
            render_flags("Trend Confirmations", flags["trend"], reg, "flag_trend")
            render_flags("Big Movers", flags["movers"], reg, "flag_mover")
            render_dropped(flags["dropped"])

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
                rc[4].write(f"{pnl:+.1%}" if pnl is not None else "—")
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
            st.dataframe(
                sc[["date", "ticker", "entry_price", "close_date", "close_price",
                    "pnl", "outcome_note"]].style.format(
                    {"entry_price": "${:.2f}", "close_price": "${:.2f}",
                     "pnl": "{:+.1%}"}, na_rep="—"),
                use_container_width=True)