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

PRICE_DIR = "data/universe_prices"
TRADE_LOG = "trade_log.csv"
YEAR = datetime.now().year

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
def regime_banner():
    spy = spy_prices()
    ma200 = spy.rolling(200).mean()
    above = spy.iloc[-1] > ma200.iloc[-1]
    pct = (spy.iloc[-1] / ma200.iloc[-1]) - 1
    if above:
        st.success(f"MARKET REGIME: TRENDING — SPY {pct:+.1%} vs 200d MA. "
                   f"Backtest evidence favors momentum-heavy weights.")
    else:
        st.warning(f"MARKET REGIME: CHOPPY/DEFENSIVE — SPY {pct:+.1%} vs 200d MA. "
                   f"Backtest evidence favors quality/trend-heavy weights.")
    return above


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
def full_scan(w_mom, w_qual, w_val, w_trend, top_n):
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
    return df.head(top_n)


# ---------- Trade log ----------
def log_trade(ticker, price, composite, thesis):
    row = pd.DataFrame([{
        "date": datetime.now().strftime("%Y-%m-%d"),
        "ticker": ticker, "price": price,
        "composite": round(composite, 2), "thesis": thesis,
    }])
    if os.path.exists(TRADE_LOG):
        row.to_csv(TRADE_LOG, mode="a", header=False, index=False)
    else:
        row.to_csv(TRADE_LOG, index=False)


# ---------- Deep dive ----------
def deep_dive(df, key_prefix):
    st.subheader("Deep Dive")
    pick = st.selectbox("Stock", df["Ticker"].tolist(), key=f"{key_prefix}_pick")
    row = df[df["Ticker"] == pick].iloc[0]

    thesis = make_thesis(row)
    st.info(thesis)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Price", f"${row['price']:.2f}")
    c1.metric("Composite", f"{row['composite']:.2f}")
    c2.metric("Momentum 12-1", f"{row['momentum']:+.1%}")
    c2.metric("vs 200d MA", f"{row['trend']:+.1%}" if pd.notna(row['trend']) else "—")
    c3.metric("F-Score", f"{row['fscore']:.0f}/9" if pd.notna(row['fscore']) else "—")
    c3.metric("Earnings Yield", f"{row['earn_yield']:.1%}" if pd.notna(row['earn_yield']) else "—")

    if c4.button("Log this idea", key=f"{key_prefix}_log"):
        log_trade(pick, row["price"], row["composite"], thesis)
        c4.success("Logged")

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
trending = regime_banner()

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
if trending:
    st.sidebar.caption("Regime suggests: momentum-heavy")
    d_mom, d_qual, d_val, d_tr = 0.5, 0.5, 0.0, 0.0
else:
    st.sidebar.caption("Regime suggests: quality/trend-heavy")
    d_mom, d_qual, d_val, d_tr = 0.3, 0.4, 0.0, 0.3
w_mom = st.sidebar.slider("Momentum", 0.0, 1.0, d_mom, 0.05)
w_qual = st.sidebar.slider("Quality (F-score)", 0.0, 1.0, d_qual, 0.05)
w_val = st.sidebar.slider("Value (earnings yield)", 0.0, 1.0, d_val, 0.05)
w_trend = st.sidebar.slider("Trend (vs 200d)", 0.0, 1.0, d_tr, 0.05)
top_n = st.sidebar.number_input("Show top N", 5, 50, 20)

tab_scan, tab_search = st.tabs(["Full S&P Scan", "Search Any Ticker"])

with tab_scan:
    if st.button("Scan Full S&P 500", type="primary"):
        with st.spinner("Scanning ~500 stocks from cached data..."):
            st.session_state["scan"] = full_scan(w_mom, w_qual, w_val, w_trend, top_n)
    if "scan" in st.session_state:
        df = st.session_state["scan"]
        st.subheader(f"Top {len(df)} — ranked by your composite")
        show_table(df)
        deep_dive(df, "scan")
    else:
        st.info("Click Scan Full S&P 500 to rank the whole index by your weights.")

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
        deep_dive(sdf, "search")