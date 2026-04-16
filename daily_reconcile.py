"""
Daily Reconciliation Script
============================
Runs every morning to check whether yesterday's live trades match
what the backtesting signal logic would have generated.

Checks all 4 live strategies:
  - RV SOL      (SOLUSDT, 30m, EMA crossover + RV spike)
  - RV SOL 15m  (SOLUSDT, 15m parallel layer — :15/:45 closes only)
  - RV BTC      (BTCUSDT, 30m, EMA crossover + RV spike)
  - SMR SOL     (SOLUSDT, 15m+1H, composite score + reversal candle)

Discrepancy types:
  MATCH   — backtest signal and live trade both fired (same direction, within 90 min)
  MISSED  — backtest signal fired, no live trade found
  GHOST   — live trade found, no backtest signal

Usage:
  python3 daily_reconcile.py                      # check yesterday
  python3 daily_reconcile.py --date 2026-03-27    # check specific date
  python3 daily_reconcile.py --days 3             # check last 3 days

Environment variables (from .env file or shell):
  SUPABASE_URL        — Supabase project URL
  SUPABASE_API_KEY    — Supabase anon/service key
"""

import os, sys, math, argparse, json
from typing import Optional
from datetime import datetime, timezone, timedelta
from collections import deque

import numpy as np
import requests
from dotenv import load_dotenv

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────
SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_KEY = os.getenv("SUPABASE_API_KEY", "")
BINANCE_BASE = "https://fapi.binance.com"

# RV strategy params (must match .env.sol / .env.btc on the bot)
RV_STRATEGIES = [
    dict(name="RV SOL", symbol="SOLUSDT", interval="30m",
         rv_threshold=3.4, ema_period=9, rv_period=10, rv_cap=4.0,
         base_risk=10, max_risk=50,
         min_sl_pct=0.4, sl_lookback=1, trailing_pct=0.60, db_table="trades"),
    # 15m parallel layer: same params, but only acts on :00/:30 open-time candles
    # (those closing at :15/:45). Candles opening at :15/:45 are skipped — they
    # close at 30m boundaries and are handled by the base 30m bot above.
    dict(name="RV SOL 15m", symbol="SOLUSDT", interval="15m",
         rv_threshold=3.4, ema_period=9, rv_period=10, rv_cap=4.0,
         base_risk=10, max_risk=50,
         min_sl_pct=0.4, sl_lookback=1, trailing_pct=0.60, db_table="trades",
         skip_minutes={15, 45}),
    dict(name="RV BTC", symbol="BTCUSDT", interval="30m",
         rv_threshold=4.0, ema_period=9, rv_period=10, rv_cap=6.0,
         base_risk=10, max_risk=50,
         min_sl_pct=0.6, sl_lookback=1, trailing_pct=0.40, db_table="btc_trades"),
]

# SMR strategy params (must match signal_engine/utils/config.py)
SMR_STRATEGIES = [
    dict(name="SMR SOL", symbol="SOLUSDT",
         h1_sma_period=20, atr_period=14,
         vol_lookback=50, rank_window=200,
         score_threshold=0.50, min_bb_dist=1.0,
         setup_bars=30, rank_bb_w=0.60, rank_vol_w=0.40,
         sl_atr_mult=1.5, tp_frac=0.20, min_rr=2.0,  # must match executor config
         db_table="smr_trades"),
]

MATCH_WINDOW_MIN  = 90      # minutes within which a signal and trade are considered a match
WARMUP_BARS       = 250     # extra bars to fetch before window for indicator warm-up
SGT               = timezone(timedelta(hours=8))
RESULTS_JSON_PATH = "/root/reconcile_results.json"   # written on server, read by status server
MAX_HISTORY_DAYS  = 7       # how many days of results to keep in the JSON file


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  CANDLE FETCHING                                                             ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def fetch_candles(symbol: str, interval: str, start_ms: int, end_ms: int) -> list[dict]:
    """Fetch OHLCV candles from Binance Futures REST API."""
    rows, cur = [], start_ms
    while cur < end_ms:
        r = requests.get(f"{BINANCE_BASE}/fapi/v1/klines",
                         params={"symbol": symbol, "interval": interval,
                                 "startTime": cur, "endTime": end_ms - 1, "limit": 1500},
                         timeout=15)
        r.raise_for_status()
        batch = r.json()
        if not batch:
            break
        rows.extend(batch)
        last = int(batch[-1][0])
        if len(batch) < 1500 or last >= end_ms:
            break
        cur = last + 1

    candles = []
    for c in rows:
        ts = datetime.fromtimestamp(int(c[0]) / 1000, tz=timezone.utc)
        candles.append({
            "ts":     ts,
            "open":   float(c[1]),
            "high":   float(c[2]),
            "low":    float(c[3]),
            "close":  float(c[4]),
            "volume": float(c[5]),
        })
    return candles


def interval_to_minutes(interval: str) -> int:
    units = {"m": 1, "h": 60, "d": 1440}
    return int(interval[:-1]) * units[interval[-1]]


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  SUPABASE QUERIES                                                            ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def _supabase_get(table: str, params: dict) -> list[dict]:
    if not SUPABASE_URL or not SUPABASE_KEY:
        return []
    headers = {
        "apikey":        SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type":  "application/json",
    }
    r = requests.get(f"{SUPABASE_URL}/rest/v1/{table}",
                     headers=headers, params=params, timeout=15)
    r.raise_for_status()
    return r.json()


def fetch_live_trades(table: str, symbol: str, start: datetime, end: datetime) -> list[dict]:
    """Fetch live trades from Supabase for a given symbol and time window."""
    start_iso = start.strftime("%Y-%m-%dT%H:%M:%S+00:00")
    end_iso   = end.strftime("%Y-%m-%dT%H:%M:%S+00:00")
    rows = _supabase_get(table, {
        "symbol":     f"eq.{symbol}",
        "entry_time": f"gte.{start_iso}",
        "entry_time": f"lte.{end_iso}",
        "order":      "entry_time.asc",
        "limit":      "500",
    })
    trades = []
    for r in rows:
        try:
            et = r.get("entry_time", "")
            if not et:
                continue
            # Parse ISO timestamp (handles +00:00 or Z suffix)
            et = et.replace("+00:00", "").replace("Z", "")
            ts = datetime.fromisoformat(et).replace(tzinfo=timezone.utc)
            if start <= ts <= end:
                trades.append({
                    "ts":        ts,
                    "direction": r.get("direction", "").upper(),
                    "entry_price": float(r.get("entry_price", 0) or 0),
                    "pnl":       float(r.get("realized_pnl") or r.get("pnl") or r.get("net_pnl") or 0),
                    "is_closed": r.get("is_closed", False),
                })
        except Exception:
            continue
    return trades


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  RV SIGNAL REPLAY                                                            ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def _in_trading_window(ts: datetime) -> bool:
    """Returns True if timestamp is within 22:00–06:00 SGT."""
    sgt = ts.astimezone(SGT)
    h = sgt.hour
    return h >= 22 or h < 6


def _calc_ema(closes: list, period: int) -> Optional[float]:
    if len(closes) < period:
        return None
    alpha = 2.0 / (period + 1)
    ema = closes[0]
    for c in closes[1:]:
        ema = alpha * c + (1 - alpha) * ema
    return ema


def replay_rv_signals(candles: list, window_start: datetime, window_end: datetime,
                      rv_threshold: float, ema_period: int, rv_period: int,
                      min_sl_pct: float, sl_lookback: int = 1,
                      skip_minutes: set = None) -> list[dict]:
    """Replay RV entry signal logic. Returns signals that fired within the window.

    skip_minutes: set of candle open-time minutes to skip for signal evaluation
    (buffer is still updated so indicators stay warm). Used by the 15m parallel
    layer to skip :15/:45 open-time candles that close at 30m boundaries.
    """
    signals = []
    buf = deque(maxlen=max(rv_period + 1, 80, sl_lookback + 1))  # rolling buffer

    for c in candles:
        buf.append(c)

        # Only check on completed candles within our window
        if c["ts"] < window_start or c["ts"] >= window_end:
            continue

        # Skip candles whose open time coincides with 30m boundaries (handled by base bot)
        if skip_minutes and c["ts"].minute in skip_minutes:
            continue

        if len(buf) < rv_period + 1:
            continue

        volumes = [x["volume"] for x in buf]
        current_vol = volumes[-1]
        avg_vol = np.mean(volumes[-(rv_period + 1):-1])  # prior rv_period candles
        if avg_vol == 0:
            continue
        rv = current_vol / avg_vol

        closes = [x["close"] for x in buf]
        ema = _calc_ema(closes, ema_period)
        if ema is None:
            continue

        last_close = c["close"]
        last_open  = c["open"]

        # Rolling SL: lowest low / highest high of last sl_lookback candles (incl. current)
        recent = list(buf)[-sl_lookback:]
        sl_low  = min(x["low"]  for x in recent)
        sl_high = max(x["high"] for x in recent)

        # Trading window gate
        if not _in_trading_window(c["ts"]):
            continue

        long_cond  = rv > rv_threshold and last_close > ema and last_open < ema
        short_cond = rv > rv_threshold and last_close < ema and last_open > ema

        if long_cond:
            sl_pct = (last_close - sl_low) / last_close * 100
            if sl_pct < min_sl_pct:
                continue
            signals.append({"ts": c["ts"], "direction": "LONG",
                             "rv": round(rv, 2), "ema": round(ema, 4),
                             "close": last_close, "sl_price": sl_low,
                             "sl_pct": round(sl_pct, 3)})

        elif short_cond:
            sl_pct = (sl_high - last_close) / last_close * 100
            if sl_pct < min_sl_pct:
                continue
            signals.append({"ts": c["ts"], "direction": "SHORT",
                             "rv": round(rv, 2), "ema": round(ema, 4),
                             "close": last_close, "sl_price": sl_high,
                             "sl_pct": round(sl_pct, 3)})

    return signals


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  SMR SIGNAL REPLAY                                                           ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def _percentile_rank(history: list[float], value: float) -> float:
    """Fraction of history values <= value (pandas-style rank)."""
    if not history:
        return 0.0
    return sum(1 for x in history if x <= value) / len(history)


def replay_smr_signals(candles_15m: list[dict], candles_1h: list[dict],
                       window_start: datetime, window_end: datetime,
                       h1_sma_period: int, vol_lookback: int, rank_window: int,
                       score_threshold: float, min_bb_dist: float,
                       setup_bars: int, rank_bb_w: float, rank_vol_w: float,
                       atr_period: int = 14, sl_atr_mult: float = 3.0,
                       tp_frac: float = 0.30, min_rr: float = 2.0) -> list[dict]:
    """Replay SMR entry signal logic. Returns entries that fired within the window."""

    # Build 1H SMA/std lookup: {hour_floor → (sma, std)} using shifted (prior bar)
    h1_closes_buf = []
    h1_lookup = {}  # ts → (sma, std) from PRIOR 1H bar
    for bar in candles_1h:
        if len(h1_closes_buf) >= h1_sma_period:
            arr = np.array(h1_closes_buf[-h1_sma_period:])
            sma = float(np.mean(arr))
            std = float(np.std(arr, ddof=0))
            h1_lookup[bar["ts"]] = (sma, std)
        h1_closes_buf.append(bar["close"])

    def _get_h1_indicators(ts: datetime):
        """Get h1_sma and h1_std valid at this 15m bar (no lookahead)."""
        # Round down to hour, then use the PREVIOUS hour's completed bar
        hour_floor = ts.replace(minute=0, second=0, microsecond=0)
        prev_hour  = hour_floor - timedelta(hours=1)
        return h1_lookup.get(prev_hour) or h1_lookup.get(hour_floor)

    # Replay 15m bars
    vol_buf     = deque(maxlen=vol_lookback + 1)
    bb_dist_buf = deque(maxlen=rank_window)
    rel_vol_buf = deque(maxlen=rank_window)

    # ATR (EWM, alpha=2/(period+1), matches live bot and SMR_Strategy_Overview)
    _alpha_atr     = 2.0 / (atr_period + 1)
    atr_val        = None
    prev_close_atr: Optional[float] = None

    active_setup = None   # {direction, bars_left, score, setup_ts}
    entries = []
    prev_prev_candle = None  # candle[i-2] for reversal trigger
    prev_candle      = None  # candle[i-1]

    for c in candles_15m:
        vol_buf.append(c["volume"])

        # ── True Range / ATR ───────────────────────────────────────────────
        if prev_close_atr is not None:
            tr = max(c["high"] - c["low"],
                     abs(c["high"] - prev_close_atr),
                     abs(c["low"]  - prev_close_atr))
        else:
            tr = c["high"] - c["low"]
        if atr_val is None:
            atr_val = tr   # initialise with first TR (matches live bot)
        else:
            atr_val = _alpha_atr * tr + (1 - _alpha_atr) * atr_val
        prev_close_atr = c["close"]

        h1 = _get_h1_indicators(c["ts"])
        if h1:
            h1_sma, h1_std = h1
            if h1_std > 0:
                bb_dist = (c["close"] - h1_sma) / h1_std
                abs_bb_dist = abs(bb_dist)
            else:
                bb_dist = 0.0
                abs_bb_dist = 0.0
        else:
            bb_dist = None

        # Relative volume: current / mean of prior vol_lookback
        if len(vol_buf) > vol_lookback:
            prior_vols = list(vol_buf)[-(vol_lookback + 1):-1]
            avg_vol = np.mean(prior_vols) if prior_vols else 0
            rel_vol = c["volume"] / avg_vol if avg_vol > 0 else 0.0
        else:
            rel_vol = 0.0

        if bb_dist is not None:
            bb_dist_buf.append(abs_bb_dist)
            rel_vol_buf.append(rel_vol)

        # -- on_bar_open equivalent: reversal check BEFORE decrement
        # Live bot calls on_bar_open (reversal check) before on_bar_close
        # (decrement), so the setup is still alive on the bar it fires.
        if active_setup is not None and prev_candle is not None and prev_prev_candle is not None:
            d = active_setup["direction"]
            long_reversal  = (prev_candle["close"] > prev_prev_candle["high"])
            short_reversal = (prev_candle["close"] < prev_prev_candle["low"])
            triggered = (d == "LONG" and long_reversal) or (d == "SHORT" and short_reversal)

            if triggered and window_start <= c["ts"] < window_end:
                rr_ok = True
                if atr_val is not None:
                    h1_now = _get_h1_indicators(c["ts"])
                    if h1_now is not None:
                        h1_sma_val, _ = h1_now
                        entry_est = c["open"]
                        dir_int   = 1 if d == "LONG" else -1
                        sl_dist   = max(sl_atr_mult * atr_val, atr_val * 0.3)
                        gap       = (h1_sma_val - entry_est) * dir_int
                        if gap <= 0:
                            rr_ok = False   # price past H1 SMA — mirrors executor overshoot guard
                        else:
                            tp_dist = tp_frac * gap
                            rr = tp_dist / sl_dist if sl_dist > 0 else 0.0
                            if rr < min_rr:
                                rr_ok = False

                if rr_ok:
                    entries.append({
                        "ts":        c["ts"],
                        "direction": d,
                        "score":     active_setup["score"],
                        "bb_dist":   active_setup["bb_dist"],
                        "setup_ts":  active_setup["setup_ts"],
                        "close":     c["close"],
                    })
                    active_setup = None

        # -- on_bar_close equivalent: tick countdown, then check new signal
        if active_setup is not None:
            active_setup["bars_left"] -= 1
            if active_setup["bars_left"] <= 0:
                active_setup = None

        # Check for signal setup (score gate)
        if (bb_dist is not None
                and len(bb_dist_buf) >= max(2, rank_window // 4)
                and len(rel_vol_buf) >= max(2, rank_window // 4)):

            rank_bb  = _percentile_rank(list(bb_dist_buf)[:-1], abs_bb_dist)
            rank_vol = _percentile_rank(list(rel_vol_buf)[:-1], rel_vol)
            score    = rank_bb_w * rank_bb + rank_vol_w * rank_vol

            if score >= score_threshold and abs_bb_dist >= min_bb_dist:
                direction = "LONG" if bb_dist < 0 else "SHORT"
                if active_setup is None:
                    active_setup = {
                        "direction":  direction,
                        "bars_left":  setup_bars,
                        "score":      round(score, 3),
                        "bb_dist":    round(bb_dist, 3),
                        "setup_ts":   c["ts"],
                    }

        prev_prev_candle = prev_candle
        prev_candle      = c

    return entries


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  RV TRADE SIMULATION (1m management)                                         ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

MARKET_FEE    = 0.0005   # 0.05% per side
SIZE_FEE_PCT  = 0.1      # % used in position sizing (matches backtester)
RISK_AMOUNT   = 10.0

def simulate_rv_trade_1m(
    direction: str,
    entry_price: float,
    sl_price: float,
    trailing_pct: float,
    candles_1m: list[dict],
    entry_ts: datetime,
    rv: float = 0.0,
    rv_threshold: float = 3.4,
    rv_cap: float = 4.0,
    base_risk: float = 10.0,
    max_risk: float = 50.0,
) -> Optional[dict]:
    """
    Simulate an RV trade using 1m candles for SL checks and trailing advancement.
    Mirrors the management loop in backtester.py v3.

    Returns a dict with exit_time, exit_price, exit_reason, trailing_steps, net_pnl,
    or None if no 1m data is available after entry.
    """
    R = abs(entry_price - sl_price)
    if R <= 0:
        return None

    tv = R * trailing_pct
    sl_pct = R / entry_price * 100
    # Dynamic risk sizing — mirrors live bot formula
    rv_strength = min(1.0, max(0.0, rv - rv_threshold) / (rv_cap - rv_threshold)) if rv_cap > rv_threshold else 0.0
    risk_amt    = base_risk + rv_strength * (max_risk - base_risk)
    pos_usdt    = risk_amt / ((sl_pct + SIZE_FEE_PCT) / 100)
    qty         = pos_usdt / entry_price

    if direction == "LONG":
        sl             = sl_price
        trailing_price = entry_price + tv
        next_sl        = sl_price + tv
    else:
        sl             = sl_price
        trailing_price = entry_price - tv
        next_sl        = sl_price - tv

    trailing_steps = 0

    for c in candles_1m:
        # Skip bars up to and including the entry bar
        if c["ts"] <= entry_ts:
            continue

        hi = c["high"]
        lo = c["low"]

        # Step 1: SL check
        if direction == "LONG" and lo <= sl:
            ep    = sl
            gross = (ep - entry_price) * qty
            fees  = entry_price * qty * MARKET_FEE + ep * qty * MARKET_FEE
            return {
                "exit_time":      c["ts"],
                "exit_price":     round(ep, 6),
                "exit_reason":    "sl",
                "trailing_steps": trailing_steps,
                "net_pnl":        round(gross - fees, 4),
            }
        elif direction == "SHORT" and hi >= sl:
            ep    = sl
            gross = (entry_price - ep) * qty
            fees  = entry_price * qty * MARKET_FEE + ep * qty * MARKET_FEE
            return {
                "exit_time":      c["ts"],
                "exit_price":     round(ep, 6),
                "exit_reason":    "sl",
                "trailing_steps": trailing_steps,
                "net_pnl":        round(gross - fees, 4),
            }

        # Step 2: Multi-step trailing advancement
        if direction == "LONG":
            while hi > trailing_price:
                sl             = next_sl
                trailing_price += tv
                next_sl        += tv
                trailing_steps += 1
        else:
            while lo < trailing_price:
                sl             = next_sl
                trailing_price -= tv
                next_sl        -= tv
                trailing_steps += 1

    # End of available 1m data — trade still open
    last = next((c for c in reversed(candles_1m) if c["ts"] > entry_ts), None)
    if last is None:
        return None
    ep    = last["close"]
    gross = (ep - entry_price) * qty if direction == "LONG" else (entry_price - ep) * qty
    fees  = entry_price * qty * MARKET_FEE + ep * qty * MARKET_FEE
    return {
        "exit_time":      last["ts"],
        "exit_price":     round(ep, 6),
        "exit_reason":    "open",
        "trailing_steps": trailing_steps,
        "net_pnl":        round(gross - fees, 4),
    }


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  RECONCILIATION LOGIC                                                        ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def reconcile(backtest_signals: list[dict], live_trades: list[dict]) -> dict:
    """
    Match backtest signals to live trades within MATCH_WINDOW_MIN minutes.
    Returns dict with matched, missed, ghost lists.
    """
    match_delta = timedelta(minutes=MATCH_WINDOW_MIN)
    matched, missed, ghost = [], [], []

    # Track which live trades were matched
    live_matched = set()

    for sig in backtest_signals:
        found = False
        for i, trade in enumerate(live_trades):
            if i in live_matched:
                continue
            time_ok = abs((trade["ts"] - sig["ts"]).total_seconds()) <= match_delta.total_seconds()
            dir_ok  = trade["direction"] == sig["direction"]
            if time_ok and dir_ok:
                matched.append({"signal": sig, "trade": trade})
                live_matched.add(i)
                found = True
                break
            elif time_ok and not dir_ok:
                # Direction mismatch — still flag it
                matched.append({"signal": sig, "trade": trade, "direction_mismatch": True})
                live_matched.add(i)
                found = True
                break
        if not found:
            missed.append(sig)

    for i, trade in enumerate(live_trades):
        if i not in live_matched:
            ghost.append(trade)

    return {"matched": matched, "missed": missed, "ghost": ghost}


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  REPORT GENERATION                                                           ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def _fmt_ts(ts: datetime) -> str:
    return ts.strftime("%H:%M UTC")

def _fmt_date(ts: datetime) -> str:
    return ts.strftime("%Y-%m-%d")


def generate_report(date: datetime, results: list[dict]) -> str:
    lines = []
    sep  = "─" * 55
    dsep = "═" * 55

    lines.append(dsep)
    lines.append(f"  DAILY RECONCILIATION — {_fmt_date(date)}")
    lines.append(dsep)
    lines.append(f"  Window : {_fmt_date(date)} 00:00 UTC → {_fmt_date(date + timedelta(days=1))} 00:00 UTC")
    lines.append("")

    total_match = total_missed = total_ghost = 0

    for r in results:
        lines.append(sep)
        lines.append(f"  {r['name']}  ({r['symbol']} · {r.get('interval', '15m')})")
        lines.append(sep)

        sigs   = r["signals"]
        trades = r["live_trades"]
        rec    = r["reconciliation"]

        # Backtest signals
        lines.append(f"  Backtest signals : {len(sigs)}")
        for s in sigs:
            extra = ""
            if "rv" in s:
                extra = f"  rv={s['rv']}  ema={s['ema']}  close={s['close']}"
            elif "score" in s:
                extra = f"  score={s['score']}  bb_dist={s['bb_dist']}  (setup {_fmt_ts(s['setup_ts'])})"
            lines.append(f"    [{_fmt_ts(s['ts'])}]  {s['direction']}{extra}")

        if not sigs:
            lines.append("    (none)")

        # Live trades
        lines.append(f"  Live trades      : {len(trades)}")
        for t in trades:
            status = " (open)" if not t["is_closed"] else ""
            pnl_str = f"  pnl=${t['pnl']:+.2f}" if t["is_closed"] else ""
            lines.append(f"    [{_fmt_ts(t['ts'])}]  {t['direction']}  entry={t['entry_price']}{pnl_str}{status}")

        if not trades:
            lines.append("    (none)")

        # Result
        lines.append("")
        if not sigs and not trades:
            lines.append("  ✓  NO SIGNAL — NO TRADE  (expected)")
        elif rec["missed"] or rec["ghost"]:
            lines.append("  ⚠️  DISCREPANCIES FOUND:")
            for m in rec["missed"]:
                lines.append(f"    MISSED  [{_fmt_ts(m['ts'])}]  {m['direction']}  — signal fired, no live trade")
            for g in rec["ghost"]:
                lines.append(f"    GHOST   [{_fmt_ts(g['ts'])}]  {g['direction']}  — live trade, no backtest signal")
        else:
            for m in rec["matched"]:
                dt_sec = int((m["trade"]["ts"] - m["signal"]["ts"]).total_seconds())
                mismatch = "  ⚠️ DIRECTION MISMATCH" if m.get("direction_mismatch") else ""
                lines.append(f"  ✓  MATCH  signal {_fmt_ts(m['signal']['ts'])} → trade {_fmt_ts(m['trade']['ts'])}  (Δt={dt_sec:+d}s){mismatch}")
                # Backtest exit simulation
                sim = m["signal"].get("simulation")
                if sim:
                    reason_label = {"sl": "SL hit", "open": "still open"}.get(sim["exit_reason"], sim["exit_reason"])
                    lines.append(f"       BT exit : {_fmt_ts(sim['exit_time'])}  @{sim['exit_price']}  {reason_label}  steps={sim['trailing_steps']}  pnl={sim['net_pnl']:+.2f}")
                # Live exit
                t = m["trade"]
                if t["is_closed"]:
                    lines.append(f"       LV exit : pnl={t['pnl']:+.2f}")
                else:
                    lines.append(f"       LV exit : (still open)")

        total_match  += len(rec["matched"])
        total_missed += len(rec["missed"])
        total_ghost  += len(rec["ghost"])
        lines.append("")

    lines.append(dsep)
    status_icon = "✓" if total_missed == 0 and total_ghost == 0 else "⚠️"
    lines.append(f"  {status_icon}  SUMMARY: {total_match} match · {total_missed} missed · {total_ghost} ghost")
    lines.append(dsep)

    return "\n".join(lines)


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  MAIN                                                                        ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def run_for_date(date: datetime) -> str:
    """Run full reconciliation for a single UTC date. Returns the report string."""
    window_start = date.replace(hour=0, minute=0, second=0, microsecond=0, tzinfo=timezone.utc)
    window_end   = window_start + timedelta(days=1)

    results = []

    # ── RV strategies ──────────────────────────────────────────────────────────
    for cfg in RV_STRATEGIES:
        interval_min = interval_to_minutes(cfg["interval"])
        warmup_ms    = WARMUP_BARS * interval_min * 60 * 1000
        start_ms     = int(window_start.timestamp() * 1000) - warmup_ms
        end_ms       = int(window_end.timestamp() * 1000)
        # 1m data: fetch from window_start through window_end + 4 days to cover trade exits
        end_ms_1m    = int((window_end + timedelta(days=4)).timestamp() * 1000)

        print(f"  Fetching {cfg['name']} 30m candles...", flush=True)
        try:
            candles = fetch_candles(cfg["symbol"], cfg["interval"], start_ms, end_ms)
        except Exception as e:
            print(f"    ⚠️  Fetch failed: {e}")
            candles = []

        print(f"  Fetching {cfg['name']} 1m candles for trade simulation...", flush=True)
        try:
            candles_1m = fetch_candles(cfg["symbol"], "1m",
                                       int(window_start.timestamp() * 1000), end_ms_1m)
        except Exception as e:
            print(f"    ⚠️  1m fetch failed (simulation unavailable): {e}")
            candles_1m = []

        signals = replay_rv_signals(
            candles, window_start, window_end,
            rv_threshold=cfg["rv_threshold"],
            ema_period=cfg["ema_period"],
            rv_period=cfg["rv_period"],
            min_sl_pct=cfg["min_sl_pct"],
            sl_lookback=cfg.get("sl_lookback", 1),
            skip_minutes=cfg.get("skip_minutes"),
        ) if candles else []

        # Attach 1m simulation to each signal
        trailing_pct = cfg.get("trailing_pct", 0.6)
        for sig in signals:
            # Entry is at the open of the next candle after the signal
            # (interval_min covers both 30m and 15m strategies)
            next_entry_ts = sig["ts"] + timedelta(minutes=interval_min)
            entry_price = next(
                (c["open"] for c in candles if c["ts"] >= next_entry_ts), None
            )
            if entry_price is not None and candles_1m:
                sim = simulate_rv_trade_1m(
                    direction    = sig["direction"],
                    entry_price  = entry_price,
                    sl_price     = sig["sl_price"],
                    trailing_pct = trailing_pct,
                    candles_1m   = candles_1m,
                    entry_ts     = next_entry_ts,
                    rv           = sig.get("rv", 0.0),
                    rv_threshold = cfg.get("rv_threshold", 3.4),
                    rv_cap       = cfg.get("rv_cap", 4.0),
                    base_risk    = cfg.get("base_risk", 10.0),
                    max_risk     = cfg.get("max_risk", 50.0),
                )
                sig["entry_price_bt"] = round(entry_price, 6)
                sig["simulation"]     = sim
            else:
                sig["entry_price_bt"] = entry_price
                sig["simulation"]     = None

        print(f"  Fetching {cfg['name']} live trades...", flush=True)
        try:
            live = fetch_live_trades(cfg["db_table"], cfg["symbol"], window_start, window_end)
        except Exception as e:
            print(f"    ⚠️  Supabase fetch failed: {e}")
            live = []

        results.append({
            "name":           cfg["name"],
            "symbol":         cfg["symbol"],
            "interval":       cfg["interval"],
            "signals":        signals,
            "live_trades":    live,
            "reconciliation": reconcile(signals, live),
        })

    # ── SMR strategies ─────────────────────────────────────────────────────────
    for cfg in SMR_STRATEGIES:
        warmup_ms_15m = WARMUP_BARS * 15 * 60 * 1000
        warmup_ms_1h  = (cfg["h1_sma_period"] + cfg["rank_window"] // 4 + 30) * 60 * 60 * 1000
        start_ms_15m  = int(window_start.timestamp() * 1000) - warmup_ms_15m
        start_ms_1h   = int(window_start.timestamp() * 1000) - warmup_ms_1h
        end_ms        = int(window_end.timestamp() * 1000)

        print(f"  Fetching {cfg['name']} candles (15m + 1H)...", flush=True)
        try:
            candles_15m = fetch_candles(cfg["symbol"], "15m", start_ms_15m, end_ms)
            candles_1h  = fetch_candles(cfg["symbol"], "1h",  start_ms_1h,  end_ms)
        except Exception as e:
            print(f"    ⚠️  Fetch failed: {e}")
            candles_15m = candles_1h = []

        signals = replay_smr_signals(
            candles_15m, candles_1h, window_start, window_end,
            h1_sma_period=cfg["h1_sma_period"],
            vol_lookback=cfg["vol_lookback"],
            rank_window=cfg["rank_window"],
            score_threshold=cfg["score_threshold"],
            min_bb_dist=cfg["min_bb_dist"],
            setup_bars=cfg["setup_bars"],
            rank_bb_w=cfg["rank_bb_w"],
            rank_vol_w=cfg["rank_vol_w"],
            atr_period=cfg["atr_period"],
            sl_atr_mult=cfg.get("sl_atr_mult", 3.0),
            tp_frac=cfg.get("tp_frac", 0.30),
            min_rr=cfg.get("min_rr", 2.0),
        ) if candles_15m else []

        print(f"  Fetching {cfg['name']} live trades...", flush=True)
        try:
            live = fetch_live_trades(cfg["db_table"], cfg["symbol"], window_start, window_end)
        except Exception as e:
            print(f"    ⚠️  Supabase fetch failed: {e}")
            live = []

        results.append({
            "name":           cfg["name"],
            "symbol":         cfg["symbol"],
            "interval":       "15m+1H",
            "signals":        signals,
            "live_trades":    live,
            "reconciliation": reconcile(signals, live),
        })

    return generate_report(date, results), results


def _serialize(obj):
    """JSON serializer that handles datetime objects."""
    if isinstance(obj, datetime):
        return obj.strftime("%Y-%m-%dT%H:%M:%SZ")
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def save_results_json(date: datetime, results: list) -> None:
    """Append today's reconciliation results to the rolling JSON file."""
    run_date = date.strftime("%Y-%m-%d")

    # Build structured record
    total_matched = total_missed = total_ghost = 0
    strategies_out = []
    for r in results:
        rec  = r["reconciliation"]
        m, ms, g = len(rec["matched"]), len(rec["missed"]), len(rec["ghost"])
        total_matched += m; total_missed += ms; total_ghost += g

        if ms > 0 or g > 0:
            status = "DISCREPANCY"
        elif m > 0:
            status = "MATCH"
        else:
            status = "NO_ACTIVITY"

        # Direction mismatch check
        if any(x.get("direction_mismatch") for x in rec["matched"]):
            status = "DISCREPANCY"

        # Enrich matched items with simulation data for the dashboard
        enriched_matches = []
        for match in rec["matched"]:
            em = dict(match)
            em["signal"] = dict(match["signal"])  # copy so we don't mutate
            sim = em["signal"].get("simulation")
            if sim:
                em["bt_exit_time"]      = sim["exit_time"]
                em["bt_exit_price"]     = sim["exit_price"]
                em["bt_exit_reason"]    = sim["exit_reason"]
                em["bt_trailing_steps"] = sim["trailing_steps"]
                em["bt_net_pnl"]        = sim["net_pnl"]
            enriched_matches.append(em)

        strategies_out.append({
            "name":             r["name"],
            "symbol":           r["symbol"],
            "interval":         r.get("interval", "15m"),
            "status":           status,
            "backtest_signals": len(r["signals"]),
            "live_trades":      len(r["live_trades"]),
            "matched":          m,
            "missed":           ms,
            "ghost":            g,
            "signals":          r["signals"],
            "live_trades_detail": r["live_trades"],
            "matches":          enriched_matches,
            "missed_signals":   rec["missed"],
            "ghost_trades":     rec["ghost"],
        })

    overall = ("DISCREPANCY" if total_missed > 0 or total_ghost > 0
               else "MATCH" if total_matched > 0 else "NO_ACTIVITY")
    # Also flag direction mismatches
    if any(any(x.get("direction_mismatch") for x in r["reconciliation"]["matched"]) for r in results):
        overall = "DISCREPANCY"

    new_record = {
        "run_date":       run_date,
        "created_at":     datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "overall_status": overall,
        "summary":        {"matched": total_matched, "missed": total_missed, "ghost": total_ghost},
        "strategies":     strategies_out,
    }

    # Load existing file
    try:
        with open(RESULTS_JSON_PATH, "r") as f:
            history = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        history = []

    # Replace existing entry for this date, or prepend
    history = [h for h in history if h.get("run_date") != run_date]
    history.insert(0, new_record)
    history = history[:MAX_HISTORY_DAYS]

    with open(RESULTS_JSON_PATH, "w") as f:
        json.dump(history, f, default=_serialize, indent=2)

    print(f"\n  Results saved → {RESULTS_JSON_PATH}")


def main():
    parser = argparse.ArgumentParser(description="Daily trade reconciliation")
    parser.add_argument("--date", type=str, default=None,
                        help="Date to check in YYYY-MM-DD format (default: yesterday)")
    parser.add_argument("--days", type=int, default=1,
                        help="Number of days to check going back from date (default: 1)")
    parser.add_argument("--out", type=str, default=None,
                        help="Optional output file to append report to")
    args = parser.parse_args()

    if args.date:
        base_date = datetime.strptime(args.date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    else:
        base_date = datetime.now(timezone.utc).replace(
            hour=0, minute=0, second=0, microsecond=0) - timedelta(days=1)

    dates = [base_date - timedelta(days=i) for i in range(args.days - 1, -1, -1)]

    for date in dates:
        print(f"\nReconciling {date.strftime('%Y-%m-%d')}...", flush=True)
        report, results = run_for_date(date)
        print("\n" + report)

        if args.out:
            with open(args.out, "a") as f:
                f.write(report + "\n\n")
            print(f"Appended to {args.out}")

        # Save structured JSON (read by status server → dashboard)
        try:
            save_results_json(date, results)
        except Exception as e:
            print(f"  ⚠️  Could not save JSON results: {e}")


if __name__ == "__main__":
    main()
