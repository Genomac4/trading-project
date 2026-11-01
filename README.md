# Systematic Equity Screener & Backtest Engine

A factor screener and survivorship-free backtest engine for the S&P 500, built to test whether a momentum + quality strategy has a real edge — and built so that it could return the answer "no."

It returned no. That finding, and the machinery that makes it trustworthy, are the point of this repository.

---

## The result

The strategy ranks S&P 500 members on 12-1 price momentum and a point-in-time Piotroski F-score, holds the top 30 equal-weighted (22 after a max-2-per-sector constraint), and rebalances monthly. Tested across three regimes:

| Period | Strategy Sharpe | SPY Sharpe | Ann. return @ 5bp | SPY |
|---|---|---|---|---|
| 2007–2012 | 0.27 | 0.16 | 5.1% | 3.0% |
| 2013–2018 | 0.69 | 0.86 | 7.8% | 9.3% |
| 2019–2026 | 0.75 | 0.91 | 13.9% | 15.7% |

**Monthly excess return over SPY is statistically indistinguishable from zero in every period and both track configurations.** Largest t-statistic anywhere: 1.12 (p = 0.27). Pooled: p = 0.35 and p = 0.92. A White's Reality Check bootstrap over the factor-mix decision returns **p = 1.000** — the chosen configuration's margin over the runner-up is fully explained by best-of-N selection.

The conclusion is not that momentum and quality don't work; they are among the best-documented premia in the literature. It is that *this implementation* shows no edge that can be distinguished from noise on the available evidence, and no achievable path to measuring one (see [Statistical power](#statistical-power)).

---

## Why the test is trustworthy

Most retail backtests fail for one of two reasons before they are ever evaluated. Both are addressed here, and the fixes are what make a null result meaningful rather than merely disappointing.

**Survivorship-free universe.** Testing on today's index members only tests companies that survived. The universe is reconstructed as it existed on each historical date, via a `members_on_date` index with explicit start/end membership windows, covering every company that has been an S&P 500 constituent since 2007 — 800+ tickers including delisted names. Delistings are handled with a cited convention (Shumway 1997, −30%).

**Point-in-time fundamentals.** Using a company's fiscal-year numbers to make a January decision uses data that had not yet been filed. Every fundamental is keyed to its real filing-availability date, so a backtest dated 2020-01-31 can only see what a trader could actually have read on 2020-01-31.

**A look-ahead test harness.** `test_backtest_lookahead.py` asserts 13 invariants over the fundamentals path. It caught a real bug: a year-keyed cache (`cached_fscore(ticker, year)`) that resolved to a fixed June-30 as-of date regardless of the simulated month, leaking up to eleven months of future information. Fixed in `52d502e`; the test fails against the pre-fix code.

**Multiple-testing correction.** `walk_forward_wrc.py` implements a block-bootstrap White's Reality Check, so the best of N tested configurations is evaluated against the distribution of the best-of-N under the null rather than against a naive single-hypothesis threshold.

---

## Selected measured findings

| Finding | Value |
|---|---|
| One-way turnover, annualized | 274% (two-way 547%) |
| Average holding period | 4.4 months |
| Break-even cost per side vs SPY | 44bp in 2007–2012; **none** in the other two periods — they trail SPY at *zero* cost |
| Cost of the 2-per-sector constraint | −7.1 pp/yr in 2019–2026 |
| Structural effect of that constraint | 22 names, not the intended 30, in 195 of 195 rebalances |
| Quarterly vs monthly rebalancing | Better in 5 of 6 period × track cells, on gross return as well as net |

The quarterly result has a mechanism behind it rather than just a bigger number: 12-1 momentum is an eleven-month signal, and re-ranking it monthly samples mostly noise.

### Statistical power

Backing the tracking error out of the t-statistics gives ~8% annualized. The implied time to reach t = 2 on a live record:

| True edge over SPY | Information ratio | Years to significance |
|---|---|---|
| 2%/yr | 0.25 | 64 |
| 4%/yr | 0.50 | 16 |
| 6%/yr | 0.75 | 7 |

A forward track record cannot settle this question on any decision-relevant horizon. **Achievable statistical power is a property of the problem, not of the effort spent on it** — which is the most portable lesson in this repository.

---

## Architecture

```
dashboard.py            Streamlit operator console — scan, rank, flag, paper-portfolio accounting
screener.py             regime weight selection (200-day MA switch)
pit_factors.py          point-in-time factor computation
point_in_time.py        filing-date-aware fundamentals engine
data_layer.py           EODHD bulk refresh, sectors, valuation
explore_constituents.py index membership (point-in-time)
build_universe.py       price cache construction and refresh
daily_refresh.py        scheduled daily job
schwab_client.py        broker view — READ-ONLY (quotes, chains, balances, positions)

backtest_composite.py   factor-mix comparison across three regimes
backtest_sector_cap.py  sector-constraint comparison; closest to the live configuration
walk_forward_wrc.py     White's Reality Check significance test
test_backtest_lookahead.py  13 look-ahead guards
data_quality.py         per-ticker staleness and coverage audit
```

**No code in this repository places an order.** The Schwab integration is read-only by construction.

### Running it

```bash
python -m pip install -r requirements.txt   # Python 3.14
streamlit run dashboard.py
```

Requires a `config.py` with an EODHD API key (not committed). Price and fundamentals caches are regenerable via `build_universe.py` and `build_fundamentals.py`.

---

## Known limitations

Self-reported, because a backtest's credibility is set by what its author discloses without being asked.

- **Test coverage is thin.** One test file, covering look-ahead in fundamentals lookups only. The selection logic (`select_top`, `select_with_cap`), the cost model, and the equity-curve construction are all untested despite deciding what the portfolio holds.
- **The strategy is not separated from the console.** The ~200 lines that decide holdings live inside a ~1,700-line Streamlit module alongside UI and P&L formatting. This is the root cause of a recurring bug class in which a UI claim drifts from the underlying logic.
- **Live scan and backtest diverge on missing factors.** The live path assigns a median z-score to a stock with no F-score (`fillna(0)`); the backtest excludes it (`dropna()`). Currently latent — fundamentals coverage is 100% across all saved snapshots — but it would activate on a new index addition without filed fundamentals.
- **Period labels overstate coverage by 13 months each.** Momentum warm-up is consumed from inside the sliced window rather than from prior history, so "2019–2026" actually begins 2020-02.
- **Execution is idealized.** Both ranking and fill use the same month-end close. No slippage, market impact, or partial fills. Transaction cost is a single blended 5bp/side assumption; a realistic retail all-in figure is 10–20bp.
- **Taxes are not modeled.** At 4.4-month average holding periods, essentially all gains are short-term — a drag larger than every trading friction combined, and the largest single omission in the cost analysis.
- **An open data-quality bug.** A delisted constituent (AGN, acquired 2020) is still scored in current scans from stale cached prices, and the dashboard's freshness check takes a maximum across tickers so it is structurally unable to detect per-ticker staleness. `data_quality.py` performs the correct check and is not yet wired into the daily job.
- **No out-of-sample record.** Every number here comes from the same three periods used to select the configuration.

---

## Status

Not funded, and not recommended for funding in this configuration. Retained as a research instrument: the point-in-time data layer, the survivorship-free universe, and the look-ahead harness transfer to any event-driven study, where observation counts are orders of magnitude higher and achievable power is correspondingly better.

Full write-ups: the cost-realism and structural audit, and the decision memo that followed it, are the primary documents on the findings above.
