# Stock Investment System (US equities, Alpaca)

An automated pipeline built on top of qlib: train an alpha model on US
equities, validate it with qlib's backtester, then run it day-to-day to
generate a target portfolio and place orders through Alpaca -- **paper
trading by default**, live trading only behind explicit, deliberate
confirmation.

```
qlib data (SP500, US)  -->  LightGBM/Alpha158 model  -->  qlib backtest (validate)
                                      |
                                      v
                     signals.py  (today's prediction ranking)
                                      |
                                      v
                    portfolio.py  (topk-dropout target portfolio,
                                    diffed against current broker holdings)
                                      |
                                      v
              run_live_trading.py  -->  broker_alpaca.py  -->  Alpaca (paper/live)
```

## 1. Set up data

qlib ships a US-equities collector. Get a ready-made dataset:

```bash
python scripts/get_data.py qlib_data --target_dir ~/.qlib/qlib_data/us_data --region us --interval 1d
```

For a current SP500 instrument list (used as the trading universe), either
rely on whatever ships with the data above, or regenerate it:

```bash
python scripts/data_collector/us_index/collector.py --qlib_dir ~/.qlib/qlib_data/us_data \
    --index_name SP500 --method parse_instruments
```

See `scripts/data_collector/yahoo/README.md` and
`scripts/data_collector/us_index/README.md` for the full data pipeline,
including how to pull fresh daily data going forward (needed so live signals
use up-to-date prices, not a stale snapshot).

## 2. Train and validate the model

This step is pure qlib -- it trains an alpha model and backtests it, so you
can judge whether the strategy is worth running at all *before* any broker
is involved:

```bash
qrun workflow_config_us_lightgbm.yaml
```

This logs the trained model + dataset to a qlib/mlflow experiment named
`us_lightgbm_topk`. Check the `PortAnaRecord` output (Sharpe, drawdown,
turnover, IC) via `qlib.workflow.R` or `mlflow ui` before trusting this model
with any money, paper or otherwise. Re-run periodically to retrain on fresh
data -- a model trained once will decay.

**This backtest is not a guarantee of live performance.** It doesn't model
slippage realistically, assumes fills at the close price, and (like any
backtest) can overfit to its historical window. Treat a good backtest as a
reason to *paper trade*, not as a reason to skip straight to live money.

## 3. Configure broker credentials

```bash
cp .env.example .env
# edit .env: fill in ALPACA_API_KEY / ALPACA_API_SECRET from
# https://app.alpaca.markets/paper/dashboard/overview (paper keys)
```

Load `.env` into your shell/scheduler however you prefer (`export $(cat .env
| xargs)`, direnv, systemd `EnvironmentFile=`, etc). `ALPACA_TRADING_MODE`
defaults to `paper` if unset -- you have to opt into `live` explicitly, and
even then `run_live_trading.py` requires a second, separate confirmation
before it will submit a real order (see below).

Before anything else, confirm the credentials actually work:

```bash
pip install -r requirements.txt
python check_connection.py
```

This is read-only -- it prints account equity, cash, buying power, current
positions, and whether the market is open, and never submits an order.

## 4. Dry-run the live pipeline

```bash
pip install -r requirements.txt
python run_live_trading.py
```

With no `--execute` flag this only prints what it *would* do: today's target
portfolio, and the buy/sell orders needed to get there from your current
Alpaca positions. Nothing is sent to Alpaca. Run this daily for a while and
sanity-check the output before trusting it with `--execute`.

### Optional: research notes on each proposed order

If `ANTHROPIC_API_KEY` is set (see `.env.example`), each proposed order gets
a short research note from Claude with web search -- a check for recent,
significant news the price-based model has no way to know about (earnings
surprises, regulatory action, lawsuits, executive departures, M&A) before you
decide whether to `--execute`. Each note ends with a flag: `CLEAR`, `WATCH`,
or `CAUTION`.

**This is informational only.** It never filters, blocks, or resizes an
order -- the model already decided what to trade; the note just gives you
one more thing to read before *you* decide. Skip it with `--skip-research`,
or just don't set `ANTHROPIC_API_KEY`.

## 5. Paper trade for real

```bash
python run_live_trading.py --execute
```

With `ALPACA_TRADING_MODE=paper` (the default) this submits orders to
Alpaca's paper account -- simulated fills against real market data, zero
real money at risk. This is the recommended way to validate the system for
weeks/months before considering live trading.

Run it once per trading day, ideally a few minutes after market open (so
the previous day's data is fully settled). A simple way to schedule it:

```cron
# crontab -e
30 9 * * 1-5 cd /path/to/examples/stock_investment_system && \
  /usr/bin/env bash -c 'set -a; source .env; set +a; python3 run_live_trading.py --execute' \
  >> logs/cron.log 2>&1
```

(`run_live_trading.py` also writes its own timestamped log to `logs/live_trading.log`.)

## 6. Going live (optional, real money)

Only do this after you've watched the paper-trading run behave sensibly for
a meaningful stretch of time, and you're comfortable with what the strategy
does in practice, not just its backtest.

1. Get **live** Alpaca API keys (different from paper keys).
2. In `.env`, set:
   ```
   ALPACA_TRADING_MODE=live
   I_CONFIRM_LIVE_TRADING=yes
   ```
3. Run with an *additional* explicit flag:
   ```bash
   python run_live_trading.py --execute --i-know-this-trades-real-money
   ```

Both the env var and the CLI flag are required together -- missing either
one blocks execution. This is intentional friction: a stray environment
variable or a copy-pasted cron line should never be enough, on its own, to
start trading real money.

## Portfolio construction

Position sizes are **not** equal-weighted across the topk set. Within the
names selected for this rebalance, capital is tilted toward higher-ranked
and lower-volatility names -- a heuristic risk adjustment (similar in spirit
to inverse-volatility/risk-parity weighting), not a full mean-variance
optimum. It deliberately does *not* model correlation between names via a
covariance matrix: a sample covariance estimated from limited history is a
well-known source of unstable, overfit weights without careful shrinkage,
which is more failure surface than this needs. `--max-position-pct` is what
bounds single-name concentration instead, regardless of what the weighting
formula would otherwise assign. See `portfolio.py`'s `_signal_vol_weights`
for the exact math, and `signals.py` for how volatility is computed (trailing
30-day daily-return std, pulled from the same qlib data used for prices).

If qlib's own `EnhancedIndexingStrategy` (optimization-based, `examples/portfolio/`
in this repo) is ever wired into live execution here, that would be the next
step up in sophistication -- full mean-variance with a proper risk model
rather than this system's simpler heuristic.

## Risk controls built in

- **Dry run by default.** `--execute` is required to send any order.
- **Position cap.** `--max-position-pct` (default 10%) caps how much of the
  account a single name can be sized to, regardless of the signal/volatility
  weighting math.
- **Cash buffer.** `--cash-buffer-pct` (default 2%) leaves a slice of the
  account uninvested.
- **Bounded turnover.** `--topk`/`--n-drop` (topk-dropout, same idea as
  qlib's `TopkDropoutStrategy`) limits how many positions are sold per
  rebalance, so a noisy day in the model's ranking doesn't churn the whole
  portfolio.
- **Untracked positions are left alone.** If you hold something outside the
  model's universe, this system won't touch it.
- **Every run is logged** to `logs/live_trading.log` with what was computed
  and what was (or wasn't) submitted.

## What this does *not* do

- No options, margin, shorting, or leverage -- long-only equities, cash
  account assumptions.
- No real-time intraday execution -- this is a once-a-day rebalance, not a
  trading bot reacting to intraday moves.
- No correlation/covariance modeling between names -- volatility is used
  per-name (see "Portfolio construction" above), but not how names move
  together, so concentration in a correlated sector/theme isn't detected.
  See `examples/portfolio/` in this repo for qlib's optimization-based
  `EnhancedIndexingStrategy` if you want full mean-variance, though wiring
  it into live execution here would be further work.
- Nothing here is investment advice. Past backtest performance does not
  predict future results; you are responsible for understanding and
  accepting the risk of any capital you put behind this.

## Files

| File | Purpose |
|---|---|
| `workflow_config_us_lightgbm.yaml` | qlib training/backtest config (LightGBM + Alpha158, SP500 universe) |
| `signals.py` | Loads the trained model and generates today's prediction ranking |
| `portfolio.py` | Topk-dropout target portfolio construction + order diffing (pure logic, unit tested) |
| `research.py` | Optional per-order research note via Claude + web search -- informational only, never blocks a trade |
| `broker_alpaca.py` | Minimal Alpaca REST client (account, positions, orders, market clock) |
| `check_connection.py` | Read-only credential/connectivity smoke test -- run this before anything else |
| `run_live_trading.py` | CLI entrypoint tying the above together, with the safety gates described above |
| `tests/test_portfolio.py` | Unit tests for the order-sizing/turnover logic |

## Running the tests

```bash
pip install -r requirements.txt
python -m pytest tests/ -v
```

These test `portfolio.py` in isolation with synthetic data -- no qlib data,
network, or broker credentials required.
