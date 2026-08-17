# Crypto Investment System (Coinbase)

A second, independent pipeline alongside `examples/stock_investment_system/`:
train an alpha model on crypto OHLCV data, validate it with qlib's
backtester, then run it day-to-day to generate a target portfolio and place
orders through Coinbase.

**Read this before running anything with `--execute`: Coinbase has no
paper-trading sandbox.** Unlike the stock system, there is no safe
"simulated" execution tier here -- the first `--execute` run is real money.
Dry-run (the default) is your only free testing option; after that, the
honest way to de-risk is starting with a small amount of real capital, not a
simulated one.

```
Coinbase candles  -->  data_collector.py  -->  qlib dump_bin.py  -->  qlib data dir
                                                                            |
                                                                            v
                                       LightGBM/Alpha158 model  -->  qlib backtest (validate)
                                                                            |
                                                                            v
                                                          signals.py  (today's prediction ranking)
                                                                            |
                                                                            v
                                        portfolio.py  (topk-dropout target portfolio,
                                                        diffed against current Coinbase holdings)
                                                                            |
                                                                            v
                              run_live_trading.py  -->  broker_coinbase.py  -->  Coinbase (real money only)
```

## Why this doesn't use qlib's built-in crypto collector

qlib ships a crypto data collector at `scripts/data_collector/crypto/`, but
it's backed by CoinGecko's free tier, which only provides price and volume --
**no open/high/low**. qlib's own README for it says plainly: *"Crypto dataset
only support Data retrieval function but not support backtest function due
to the lack of OHLC data."* That rules out the Alpha158 handler and the
backtester, which both need real OHLC bars.

`data_collector.py` in this directory instead pulls full daily OHLCV candles
directly from Coinbase's own Advanced Trade API (via the official
`coinbase-advanced-py` SDK), for every USD-quoted spot product Coinbase
lists. That's real candle data, so the same Alpha158/LightGBM approach used
for stocks works here too.

## 1. Set up data

```bash
pip install -r requirements.txt

# Public market data -- no API keys needed for this step.
python data_collector.py download --source_dir ~/.qlib/coinbase_data/source/1d --start 2018-01-01
python data_collector.py normalize --source_dir ~/.qlib/coinbase_data/source/1d \
    --normalize_dir ~/.qlib/coinbase_data/source/1d_normalized

# Hand off to qlib's own (already-tested) binary dumper:
cd ../../scripts
python dump_bin.py dump_all \
    --data_path ~/.qlib/coinbase_data/source/1d_normalized \
    --qlib_dir ~/.qlib/qlib_data/crypto_data \
    --freq day --date_field_name date \
    --include_fields open,close,high,low,volume,factor
cd -
```

`dump_bin.py` auto-generates `instruments/all.txt` from every symbol you
downloaded -- that's the `market: all` universe the training config uses, and
it's exactly "every USD-quoted spot product Coinbase listed at collection
time." `data_collector.py normalize` drops any coin with fewer than 100 days
of history (`--min_rows`), since Alpha158's feature windows need a minimum
run of data to compute at all.

**Re-run `download`/`normalize`/`dump_bin.py` periodically** to pick up newly
listed coins and keep prices current -- there's no incremental-update path
here, it's a full re-pull each time.

### On universe breadth

This is deliberately configured for **every** USD spot pair Coinbase lists,
not a curated large-cap list. That means the training data and the live
universe both include a lot of thin, volatile, easily-manipulated small-cap
tokens alongside majors like BTC and ETH. `TopkDropoutStrategy`'s ranking
doesn't know the difference between "genuinely mispriced" and "thinly traded
and noisy." If you see the model consistently rotating into illiquid names,
consider filtering the universe (e.g. by `volume_24h` from the product list)
before dumping -- `data_collector.py` doesn't do this filtering for you.

## 2. Train and validate the model

```bash
qrun workflow_config_crypto_lightgbm.yaml
```

Logs to a qlib/mlflow experiment named `crypto_lightgbm_topk`. Check the
`PortAnaRecord` output (Sharpe, drawdown, turnover, IC) before trusting this
model with any money. The config uses `ann_scaler: 365` (not 252) for Sharpe
annualization, since crypto trades every calendar day, and sets
`limit_threshold`/`trade_unit` to `null` since crypto has no daily price
limits or lot-size rounding. Default `open_cost`/`close_cost` (0.6% each) are
a rough Coinbase taker-fee estimate -- adjust to your actual fee tier before
trusting the backtest's return numbers.

**This backtest is not a guarantee of live performance** -- same caveat as
the stock system, arguably sharper here given how much noisier and more
manipulable small-cap crypto markets are than equities.

## 3. Configure Coinbase credentials

```bash
cp .env.example .env
# edit .env: fill in COINBASE_API_KEY / COINBASE_API_SECRET from
# https://portal.cdp.coinbase.com/
```

Use a **trade-only key** (View + Trade permissions, not Transfer/Withdraw) --
that caps the damage if the key is ever exposed to unauthorized trades rather
than fund movement.

Then verify the credentials actually work before touching the data/training
pipeline:

```bash
python check_connection.py
```

This only reads your account (equity, cash, current positions) -- it never
submits an order. If this fails, nothing downstream will work either, so fix
it here first.

## 4. Dry-run the live pipeline

```bash
python run_live_trading.py
```

With no `--execute` flag this only prints what it *would* do -- today's
target portfolio and the buy/sell orders needed to reach it from your current
Coinbase holdings. Nothing is sent to Coinbase. **Run this daily for a while
and actually read the output** before trusting it with `--execute` -- this is
the only no-risk feedback loop you get, since there's no paper account behind
it.

### Optional: research notes on each proposed order

If `ANTHROPIC_API_KEY` is set (see `.env.example`), each proposed order gets
a short research note from Claude with web search -- a check for recent,
significant news the price-based model has no way to know about (exploits,
depegs, exchange delistings, regulatory action, a project going dark) before
you decide whether to `--execute`. This matters more here than for stocks:
the universe is every USD-quoted Coinbase pair, including thin, easily
manipulated small-cap tokens where a single bad headline is the whole story.
Each note ends with a flag: `CLEAR`, `WATCH`, or `CAUTION`.

**This is informational only.** It never filters, blocks, or resizes an
order -- the model already decided what to trade; the note just gives you
one more thing to read before *you* decide. Skip it with `--skip-research`,
or just don't set `ANTHROPIC_API_KEY`.

## 5. Going live (real money, no smaller step available)

```bash
export I_CONFIRM_LIVE_TRADING=yes   # in .env or your shell/scheduler
python run_live_trading.py --execute --i-know-this-trades-real-money
```

Both the env var and the CLI flag are required together -- missing either one
blocks execution, the same pattern the stock system uses to gate *live*
mode. Here it gates *all* execution, since Coinbase has no other kind.
**Start with capital you're fully prepared to lose entirely** and scale up
only after watching it run. Consider setting `--max-position-pct` low (e.g.
2-5%) initially, given the universe includes small, volatile names.

Run once per day; crypto trades 24/7 so there's no market-open check to wait
for. A simple way to schedule it:

```cron
# crontab -e
0 0 * * * cd /path/to/examples/crypto_investment_system && \
  /usr/bin/env bash -c 'set -a; source .env; set +a; python3 run_live_trading.py --execute --i-know-this-trades-real-money' \
  >> logs/cron.log 2>&1
```

## Portfolio construction

Position sizes are **not** equal-weighted across the topk set -- capital is
tilted toward higher-ranked and lower-volatility coins (trailing 30-day
daily-return std), a heuristic risk adjustment rather than a full
mean-variance optimum. This matters more here than for stocks: the universe
spans majors like BTC/ETH down to thin small-caps whose volatility can be an
order of magnitude apart, so equal-weighting would have put the same dollar
bet on a stablecoin-adjacent major and a wildly noisy micro-cap. No
correlation between coins is modeled -- `--max-position-pct` is what bounds
single-name concentration instead. See
`examples/stock_investment_system/README.md` § Portfolio construction for
the full rationale (the logic in `portfolio.py` is identical) and
`signals.py` for how volatility is computed.

## Risk controls built in

Same shape as the stock system (`--max-position-pct`, `--cash-buffer-pct`,
topk/n_drop-bounded turnover, untracked positions left alone, every run
logged to `logs/live_trading.log`) -- see
`examples/stock_investment_system/README.md` § Risk controls for the full
list. The one addition here: because there's no paper mode, both safety
flags are required for *every* execution, not just a "live" tier.

## What this does *not* do

- No margin, leverage, futures/perpetuals, or shorting -- spot only, long-only.
- No liquidity/manipulation filtering of the universe -- see "On universe
  breadth" above.
- No real-time intraday execution -- a once-a-day rebalance.
- No correlation/covariance modeling between coins -- volatility is used
  per-name, but not how coins move together.
- Nothing here is investment advice. Crypto markets are more volatile and
  less regulated than equities; a good backtest here is weaker evidence of
  future performance than the same backtest would be for stocks.

## Files

| File | Purpose |
|---|---|
| `data_collector.py` | Downloads OHLCV candles from Coinbase and normalizes them for qlib's `dump_bin.py` |
| `workflow_config_crypto_lightgbm.yaml` | qlib training/backtest config (LightGBM + Alpha158, all Coinbase USD pairs) |
| `signals.py` | Loads the trained model and generates today's prediction ranking |
| `portfolio.py` | Topk-dropout target portfolio construction + order diffing (identical to the stock system's -- asset-agnostic, unit tested) |
| `research.py` | Optional per-order research note via Claude + web search -- informational only, never blocks a trade |
| `broker_coinbase.py` | Coinbase broker adapter (accounts, positions, price lookup, market orders), built on the official `coinbase-advanced-py` SDK |
| `check_connection.py` | Read-only credential/connectivity smoke test -- run this before anything else |
| `run_live_trading.py` | CLI entrypoint, with the stricter safety gates described above |
| `tests/test_portfolio.py` | Unit tests for the order-sizing/turnover logic |

## Running the tests

```bash
pip install -r requirements.txt
python -m pytest tests/ -v
```

Tests cover `portfolio.py` only, with synthetic data -- no Coinbase
credentials, network, or qlib data required.
