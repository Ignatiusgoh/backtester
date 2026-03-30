"""
RV BTC 30m — SL Lookback Sweep
================================
Tests the effect of using a rolling N-candle low/high as the initial
stop-loss instead of the signal candle's own low/high.

SL modes tested:
  lookback=1  — current candle's low/high (live baseline)
  lookback=5  — lowest low / highest high of last 5 candles
  lookback=10 — lowest low / highest high of last 10 candles
  lookback=15 — lowest low / highest high of last 15 candles

All other params are fixed at live bot values:
  rv_period=10, rv_threshold=4.0, ema_period=9,
  min_sl_pct=0.6%, trailing_pct=0.4%
"""

import os
import numpy as np
import pandas as pd
from datetime import datetime, timezone

# ── Params (fixed at live BTC values) ─────────────────────────────────────────
RISK_AMOUNT       = 10.0
MARKET_FEE        = 0.0005   # 0.05% per side
FEE_PCT           = 0.1      # used in position sizing
RV_THRESHOLD      = 4.0
RV_PERIOD         = 10
EMA_PERIOD        = 9
TRAILING_PCT      = 0.40
MIN_SL_PCT        = 0.6
SESSION_START_UTC = 14
SESSION_END_UTC   = 22

SL_LOOKBACKS = [1, 5, 10, 15]

START = datetime(2023, 3, 11, tzinfo=timezone.utc)
END   = datetime(2026, 3, 11, tzinfo=timezone.utc)

# ── Data ──────────────────────────────────────────────────────────────────────
HERE      = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(HERE, "data/BTCUSDT_30m_20230311_20260311.csv")


def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_FILE)
    df["open_time"] = pd.to_datetime(df["open_time"], utc=True)
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.sort_values("open_time").reset_index(drop=True)


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["rv"] = df["volume"] / df["volume"].shift(1).rolling(RV_PERIOD).mean()
    alpha = 2.0 / (EMA_PERIOD + 1.0)
    closes = df["close"].to_numpy()
    ema = np.empty(len(closes))
    ema[0] = closes[0]
    for i in range(1, len(closes)):
        ema[i] = alpha * closes[i] + (1 - alpha) * ema[i - 1]
    df["ema"] = ema
    return df


# ── Backtest engine ────────────────────────────────────────────────────────────

def run_backtest(df: pd.DataFrame, sl_lookback: int) -> list:
    opens      = df["open"].to_numpy()
    highs      = df["high"].to_numpy()
    lows       = df["low"].to_numpy()
    closes     = df["close"].to_numpy()
    rvs        = df["rv"].to_numpy()
    emas       = df["ema"].to_numpy()
    open_times = df["open_time"].to_numpy()
    hours      = pd.DatetimeIndex(df["open_time"]).hour.to_numpy()
    n          = len(closes)

    trades = []
    active = None  # [side, entry, sl, qty, entry_time, trailing_price, next_sl, trailing_value, steps]

    for i in range(1, n):

        # ── Manage open trade ──────────────────────────────────────────────────
        if active is not None:
            hi, lo, cl = highs[i], lows[i], closes[i]
            side = active[0]

            sl_hit = (side == "long"  and lo <= active[2]) or \
                     (side == "short" and hi >= active[2])

            if sl_hit:
                ep    = active[2]
                entry = active[1]; qty = active[3]; et = active[4]
                gross = (ep - entry) * qty if side == "long" else (entry - ep) * qty
                fees  = (entry * qty + ep * qty) * MARKET_FEE
                trades.append({
                    "entry_time":     et,
                    "exit_time":      open_times[i],
                    "side":           side,
                    "entry_price":    round(entry, 2),
                    "exit_price":     round(ep, 2),
                    "initial_sl":     round(active[2], 2),
                    "qty":            round(qty, 6),
                    "net_pnl":        round(gross - fees, 4),
                    "exit_reason":    "sl",
                    "trailing_steps": active[8],
                })
                active = None
                continue

            # Trailing step
            tv = active[7]
            if side == "long" and cl > active[5]:
                active[2] = active[6]; active[5] += tv; active[6] += tv; active[8] += 1
            elif side == "short" and cl < active[5]:
                active[2] = active[6]; active[5] -= tv; active[6] -= tv; active[8] += 1
            continue

        # ── Look for new signal ────────────────────────────────────────────────
        rv  = rvs[i]
        ema = emas[i]
        if np.isnan(rv) or np.isnan(ema):
            continue
        if not (SESSION_START_UTC <= hours[i] < SESSION_END_UTC):
            continue
        if rv <= RV_THRESHOLD:
            continue

        op, cl = opens[i], closes[i]
        long_sig  = cl > ema and op < ema
        short_sig = cl < ema and op > ema
        if not long_sig and not short_sig:
            continue
        if i + 1 >= n:
            continue

        entry = opens[i + 1]

        # ── SL: rolling low/high over last sl_lookback candles (incl. current) ─
        lb_start = max(0, i - sl_lookback + 1)

        if long_sig:
            sl = float(np.min(lows[lb_start: i + 1]))
            if sl >= entry:
                continue
            R           = entry - sl
            sl_dist_pct = R / entry * 100
            if sl_dist_pct < MIN_SL_PCT:
                continue
            pos_usdt       = RISK_AMOUNT / ((sl_dist_pct + FEE_PCT) / 100)
            qty            = pos_usdt / entry
            trailing_value = R * TRAILING_PCT
            trailing_price = entry + trailing_value
            next_sl        = sl + trailing_value
            side           = "long"
        else:
            sl = float(np.max(highs[lb_start: i + 1]))
            if sl <= entry:
                continue
            R           = sl - entry
            sl_dist_pct = R / entry * 100
            if sl_dist_pct < MIN_SL_PCT:
                continue
            pos_usdt       = RISK_AMOUNT / ((sl_dist_pct + FEE_PCT) / 100)
            qty            = pos_usdt / entry
            trailing_value = R * TRAILING_PCT
            trailing_price = entry - trailing_value
            next_sl        = sl - trailing_value
            side           = "short"

        active = [side, entry, sl, qty, open_times[i + 1],
                  trailing_price, next_sl, trailing_value, 0]

    # Close open trade at end of data
    if active is not None:
        ep    = closes[-1]
        entry = active[1]; qty = active[3]; et = active[4]; side = active[0]
        gross = (ep - entry) * qty if side == "long" else (entry - ep) * qty
        fees  = (entry * qty + ep * qty) * MARKET_FEE
        trades.append({
            "entry_time":     et,
            "exit_time":      open_times[-1],
            "side":           side,
            "entry_price":    round(entry, 2),
            "exit_price":     round(ep, 2),
            "initial_sl":     round(active[2], 2),
            "qty":            round(qty, 6),
            "net_pnl":        round(gross - fees, 4),
            "exit_reason":    "eod",
            "trailing_steps": active[8],
        })

    return trades


# ── Summary metrics ────────────────────────────────────────────────────────────

def summarise(trades: list, sl_lookback: int) -> dict:
    if not trades:
        return {"sl_lookback": sl_lookback, "trades": 0}
    df = pd.DataFrame(trades)
    df["entry_time"] = pd.to_datetime(df["entry_time"], utc=True)
    df["exit_time"]  = pd.to_datetime(df["exit_time"],  utc=True)
    df["win"]        = df["net_pnl"] > 0
    df["cum_pnl"]    = df["net_pnl"].cumsum()
    df["month"]      = df["entry_time"].dt.to_period("M")

    wins   = df[df["net_pnl"] > 0]
    losses = df[df["net_pnl"] <= 0]
    total  = len(df)
    wr     = len(wins) / total
    gp     = wins["net_pnl"].sum()
    gl     = abs(losses["net_pnl"].sum())
    pf     = gp / gl if gl > 0 else float("inf")
    aw     = wins["net_pnl"].mean()   if len(wins)   > 0 else 0.0
    al     = losses["net_pnl"].mean() if len(losses) > 0 else 0.0
    ev     = wr * aw + (1 - wr) * al
    max_dd = float((df["cum_pnl"] - df["cum_pnl"].cummax()).min())
    tpm    = total / df["month"].nunique()

    df["sl_dist_pct"] = abs(df["entry_price"] - df["initial_sl"]) / df["entry_price"] * 100
    avg_sl_dist = df["sl_dist_pct"].mean()

    return {
        "sl_lookback":    sl_lookback,
        "trades":         total,
        "trades_pm":      round(tpm, 1),
        "win_rate":       round(wr, 4),
        "net_pnl":        round(df["net_pnl"].sum(), 2),
        "profit_factor":  round(pf, 3),
        "avg_win":        round(aw, 2),
        "avg_loss":       round(al, 2),
        "ev_per_trade":   round(ev, 3),
        "max_drawdown":   round(max_dd, 2),
        "avg_sl_dist_pct": round(avg_sl_dist, 3),
        "avg_trail_steps": round(df["trailing_steps"].mean(), 2),
        "exits_sl":       int((df["exit_reason"] == "sl").sum()),
        "exits_eod":      int((df["exit_reason"] == "eod").sum()),
    }


def yearly_breakdown(trades: list, sl_lookback: int):
    if not trades:
        return
    df = pd.DataFrame(trades)
    df["entry_time"] = pd.to_datetime(df["entry_time"], utc=True)
    df["year"]       = df["entry_time"].dt.year
    df["cum_pnl"]    = df["net_pnl"].cumsum()
    print(f"\n  Year-by-year  (sl_lookback={sl_lookback}):")
    print(f"    {'Year':>5}  {'Trades':>7}  {'Win%':>6}  {'Net PnL':>9}  {'PF':>6}  {'MaxDD':>9}")
    for yr, g in df.groupby("year"):
        wr_y  = (g["net_pnl"] > 0).mean()
        net_y = g["net_pnl"].sum()
        gp_y  = g[g["net_pnl"] > 0]["net_pnl"].sum()
        gl_y  = abs(g[g["net_pnl"] <= 0]["net_pnl"].sum())
        pf_y  = gp_y / gl_y if gl_y > 0 else float("inf")
        cum_y = g["net_pnl"].cumsum()
        dd_y  = (cum_y - cum_y.cummax()).min()
        print(f"    {yr:>5}  {len(g):>7}  {wr_y:>6.1%}  ${net_y:>8.2f}  {pf_y:>6.3f}  ${dd_y:>8.2f}")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Loading data...")
    df_raw = load_data()
    print(f"  {len(df_raw):,} 30m candles  ({df_raw['open_time'].iloc[0].date()} → {df_raw['open_time'].iloc[-1].date()})")

    print("Computing indicators...")
    df = compute_indicators(df_raw)

    results = []
    all_trades = {}

    for lb in SL_LOOKBACKS:
        label = "current candle (live)" if lb == 1 else f"last {lb} candles"
        print(f"\nRunning sl_lookback={lb}  ({label})...")
        trades = run_backtest(df, sl_lookback=lb)
        all_trades[lb] = trades
        s = summarise(trades, lb)
        results.append(s)
        print(f"  {s['trades']} trades | WR={s['win_rate']:.1%} | PF={s['profit_factor']:.3f} | "
              f"Net=${s['net_pnl']:.2f} | MaxDD=${s['max_drawdown']:.2f} | "
              f"Avg SL dist={s['avg_sl_dist_pct']:.2f}%")

    # ── Summary table ─────────────────────────────────────────────────────────
    W = "=" * 80
    print(f"\n\n{W}")
    print("  RV BTC 30m — SL LOOKBACK COMPARISON")
    print(f"  Period: {START.date()} → {END.date()}")
    print(W)

    cols = ["sl_lookback", "trades", "trades_pm", "win_rate", "net_pnl",
            "profit_factor", "avg_win", "avg_loss", "ev_per_trade",
            "max_drawdown", "avg_sl_dist_pct", "avg_trail_steps"]
    labels = {
        "sl_lookback":    "SL Lookback",
        "trades":         "Trades",
        "trades_pm":      "Trades/mo",
        "win_rate":       "Win Rate",
        "net_pnl":        "Net PnL",
        "profit_factor":  "Prof. Factor",
        "avg_win":        "Avg Win",
        "avg_loss":       "Avg Loss",
        "ev_per_trade":   "EV/Trade",
        "max_drawdown":   "Max DD",
        "avg_sl_dist_pct":"Avg SL%",
        "avg_trail_steps":"Avg Trail",
    }

    print(f"\n  {'Metric':<18}", end="")
    for lb in SL_LOOKBACKS:
        tag = "baseline" if lb == 1 else f"last {lb}c"
        print(f"  {tag:>14}", end="")
    print()
    print("  " + "-" * (18 + 16 * len(SL_LOOKBACKS)))

    for col in cols:
        print(f"  {labels[col]:<18}", end="")
        for s in results:
            val = s.get(col, "n/a")
            if col == "win_rate":
                fmt = f"{val:.1%}"
            elif col in ("net_pnl", "avg_win", "avg_loss", "max_drawdown"):
                fmt = f"${val:.2f}"
            elif col in ("profit_factor", "ev_per_trade"):
                fmt = f"{val:.3f}"
            elif col == "avg_sl_dist_pct":
                fmt = f"{val:.2f}%"
            elif col in ("trades_pm", "avg_trail_steps"):
                fmt = f"{val:.1f}"
            else:
                fmt = str(val)
            print(f"  {fmt:>14}", end="")
        print()

    # ── Year-by-year for each lookback ────────────────────────────────────────
    print(f"\n{W}")
    print("  YEAR-BY-YEAR BREAKDOWN")
    print(W)
    for lb in SL_LOOKBACKS:
        yearly_breakdown(all_trades[lb], lb)

    # ── Save summary CSV ─────────────────────────────────────────────────────
    out_csv = os.path.join(HERE, "btc_sl_lookback_results.csv")
    pd.DataFrame(results).to_csv(out_csv, index=False)
    print(f"\n\nResults saved → {out_csv}")
