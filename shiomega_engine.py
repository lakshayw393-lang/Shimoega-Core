"""
Shiomega Daily — signal engine, scanner, and portfolio backtest.

Shared by shiomega_daily_screener.py, shiomega_daily_backtest.py and the
Streamlit app so all three use the exact same gate logic.

Gates (long; shorts mirrored):
  L1  Tenkan gold-crosses Kijun while close is BELOW the Kumo,
      within the last CROSS_MAX_AGE (40) daily bars      — the heads-up
  L2  Tenkan sloping up over the last 3 bars
  L3  FRESH clean Chikou breakout: close clears the 11-bar high window
      around the Chikou position (bars -31…-21), not cleared yesterday
                                                          — the trigger
  L4  Regime: close above SMA(50)

Exits:
  Hard stop  = min(Kijun@entry, 10-bar swing low) - 0.1*ATR14   (static)
  Kijun trail: close through Kijun against the position -> exit next open
  Conservative intrabar: stop resolves first; gaps fill at the open.

Fills: signal on close of bar i -> entry at open of bar i+1 (no look-ahead).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from shiomega_core import (
    ichimoku, gold_cross, death_cross, chikou_break_up, chikou_break_down,
    CROSS_MAX_AGE, DISP, CHIKOU_WIN,
)

# ---- portfolio conventions ------------------------------------------------ #
CAPITAL = 100_000.0
PER_TRADE = 5_000.0
MAX_POSITIONS = 20
COMM_PER_SHARE = 0.005
COMM_MIN = 1.00
SMA_REGIME = 50
ATR_N = 14
STOP_ATR_BUFFER = 0.10
SWING_N = 10


# --------------------------------------------------------------------------- #
def atr(df: pd.DataFrame, n: int = ATR_N) -> pd.Series:
    hi, lo, cl = df["High"], df["Low"], df["Close"]
    pc = cl.shift(1)
    tr = pd.concat([hi - lo, (hi - pc).abs(), (lo - pc).abs()], axis=1).max(axis=1)
    return tr.rolling(n).mean()


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    """Indicator frame with all gate columns for BOTH sides."""
    d = ichimoku(df)
    d["sma50"] = d["Close"].rolling(SMA_REGIME).mean()
    d["atr14"] = atr(d)
    d["swing_low"] = d["Low"].rolling(SWING_N).min()
    d["swing_high"] = d["High"].rolling(SWING_N).max()

    # L1/S1 — location-qualified cross, recency window
    gcross = (gold_cross(d) & (d["Close"] < d["cloud_bot"])).fillna(False)
    dcross = (death_cross(d) & (d["Close"] > d["cloud_top"])).fillna(False)
    d["gold_cross_ok"] = gcross
    d["death_cross_ok"] = dcross
    win = CROSS_MAX_AGE + 1
    d["long_cross_recent"] = gcross.rolling(win, min_periods=1).max().astype(bool)
    d["short_cross_recent"] = dcross.rolling(win, min_periods=1).max().astype(bool)

    # bars since the qualifying cross (NaN if never)
    idx = np.arange(len(d), dtype=float)
    last_g = pd.Series(np.where(gcross.to_numpy(), idx, np.nan), index=d.index).ffill()
    last_d = pd.Series(np.where(dcross.to_numpy(), idx, np.nan), index=d.index).ffill()
    d["bars_since_gold"] = idx - last_g
    d["bars_since_death"] = idx - last_d

    # L2/S2 — Tenkan slope over 3 bars
    d["tenkan_up"] = d["tenkan"] > d["tenkan"].shift(3)
    d["tenkan_dn"] = d["tenkan"] < d["tenkan"].shift(3)

    # L3/S3 — fresh, clean Chikou breakout (bool shifts use fill_value to
    # dodge the pandas 3.0 object-dtype trap)
    bku = chikou_break_up(d).fillna(False)
    bkd = chikou_break_down(d).fillna(False)
    d["chikou_up"] = bku
    d["chikou_dn"] = bkd
    d["chikou_up_fresh"] = bku & ~bku.shift(1, fill_value=False)
    d["chikou_dn_fresh"] = bkd & ~bkd.shift(1, fill_value=False)

    # L4/S4 — regime
    d["regime_long"] = d["Close"] > d["sma50"]
    d["regime_short"] = d["Close"] < d["sma50"]

    d["long_signal"] = (d["long_cross_recent"] & d["tenkan_up"]
                        & d["chikou_up_fresh"] & d["regime_long"])
    d["short_signal"] = (d["short_cross_recent"] & d["tenkan_dn"]
                         & d["chikou_dn_fresh"] & d["regime_short"])

    # static protective stops evaluated on the signal bar
    d["long_stop"] = np.minimum(d["kijun"], d["swing_low"]) - STOP_ATR_BUFFER * d["atr14"]
    d["short_stop"] = np.maximum(d["kijun"], d["swing_high"]) + STOP_ATR_BUFFER * d["atr14"]

    # watchlist: cross + slope + regime aligned, Chikou not yet through
    d["long_watch"] = (d["long_cross_recent"] & d["tenkan_up"]
                       & d["regime_long"] & ~bku)
    d["short_watch"] = (d["short_cross_recent"] & d["tenkan_dn"]
                        & d["regime_short"] & ~bkd)
    return d


# --------------------------------------------------------------------------- #
# Scanner — evaluate the last CLOSED daily bar of one ticker
# --------------------------------------------------------------------------- #
def scan_last_bar(ticker: str, df: pd.DataFrame) -> dict | None:
    if len(df) < 90:
        return None
    d = prepare(df)
    i = -1
    row = d.iloc[i]

    def pack(status: str, side: str) -> dict:
        long = side == "LONG"
        stop = float(row["long_stop"] if long else row["short_stop"])
        close = float(row["Close"])
        entry = close  # next-open estimate; actual fill is tomorrow's open
        risk_pct = abs(entry - stop) / entry * 100 if entry else np.nan
        bars = row["bars_since_gold"] if long else row["bars_since_death"]
        return {
            "ticker": ticker, "status": status, "side": side,
            "date": d.index[i].date().isoformat(),
            "close": round(close, 2),
            "entry": round(entry, 2) if status == "TRIGGERED" else np.nan,
            "stop": round(stop, 2),
            "risk_pct": round(risk_pct, 1) if status == "TRIGGERED" else np.nan,
            "bars_since_cross": int(bars) if np.isfinite(bars) else np.nan,
        }

    if bool(row["long_signal"]):
        return pack("TRIGGERED", "LONG")
    if bool(row["short_signal"]):
        return pack("TRIGGERED", "SHORT")
    if bool(row["long_watch"]):
        return pack("WATCHLIST", "LONG")
    if bool(row["short_watch"]):
        return pack("WATCHLIST", "SHORT")
    return None


# --------------------------------------------------------------------------- #
# Event-driven portfolio backtest
# --------------------------------------------------------------------------- #
def _commission(shares: int) -> float:
    return max(COMM_MIN, shares * COMM_PER_SHARE)


def backtest_portfolio(data: dict[str, pd.DataFrame],
                       capital: float = CAPITAL,
                       per_trade: float = PER_TRADE,
                       max_positions: int = MAX_POSITIONS,
                       progress=None) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """data: {ticker: OHLCV daily DataFrame}. Returns (trades, equity, summary)."""
    prepped: dict[str, pd.DataFrame] = {}
    for k, (tkr, df) in enumerate(data.items()):
        if len(df) >= 90:
            prepped[tkr] = prepare(df)
        if progress:
            progress((k + 1) / max(1, len(data)), f"indicators · {tkr}")

    all_dates = sorted(set().union(*[set(d.index) for d in prepped.values()])) \
        if prepped else []
    cash = capital
    positions: dict[str, dict] = {}
    pending_entries: list[dict] = []   # signals from yesterday -> fill today open
    trades: list[dict] = []
    equity_rows: list[dict] = []

    for date in all_dates:
        # ---- 1. fill pending entries at today's open --------------------- #
        still_pending = []
        for sig in pending_entries:
            tkr = sig["ticker"]
            d = prepped[tkr]
            if date not in d.index:
                still_pending.append(sig)          # ticker halted today; retry
                continue
            if tkr in positions or len(positions) >= max_positions:
                continue                            # one per underlier / cap
            o = float(d.loc[date, "Open"])
            shares = int(per_trade // o)
            if shares < 1:
                continue
            cost = shares * o
            comm = _commission(shares)
            if cash < cost + comm and sig["side"] == "LONG":
                continue
            cash -= comm
            cash += -cost if sig["side"] == "LONG" else cost
            positions[tkr] = {
                "side": sig["side"], "entry": o, "shares": shares,
                "stop": sig["stop"], "signal_date": sig["signal_date"],
                "entry_date": date, "bars": 0, "exit_flag": False,
            }
        pending_entries = still_pending

        # ---- 2. manage open positions on this bar ------------------------ #
        for tkr in list(positions):
            d = prepped[tkr]
            if date not in d.index:
                continue
            p = positions[tkr]
            row = d.loc[date]
            o, hi, lo, cl = float(row["Open"]), float(row["High"]), float(row["Low"]), float(row["Close"])
            p["bars"] += 1
            exit_px, reason = None, None

            if p["exit_flag"]:                      # Kijun trail fired on prior close
                exit_px, reason = o, "KIJUN_TRAIL"
            elif p["side"] == "LONG":
                if o <= p["stop"]:
                    exit_px, reason = o, "HARD_STOP"     # gap through
                elif lo <= p["stop"]:
                    exit_px, reason = p["stop"], "HARD_STOP"
            else:
                if o >= p["stop"]:
                    exit_px, reason = o, "HARD_STOP"
                elif hi >= p["stop"]:
                    exit_px, reason = p["stop"], "HARD_STOP"

            if exit_px is None:
                kij = float(row["kijun"])
                if np.isfinite(kij) and ((p["side"] == "LONG" and cl < kij)
                                         or (p["side"] == "SHORT" and cl > kij)):
                    p["exit_flag"] = True           # exit at NEXT open
                continue

            shares = p["shares"]
            comm = _commission(shares)
            cash -= comm
            cash += shares * exit_px if p["side"] == "LONG" else -shares * exit_px
            sgn = 1.0 if p["side"] == "LONG" else -1.0
            gross = sgn * (exit_px - p["entry"]) * shares
            net = gross - 2 * _commission(shares)   # entry + exit legs
            risk = abs(p["entry"] - p["stop"]) * shares
            trades.append({
                "ticker": tkr, "side": p["side"],
                "signal_date": p["signal_date"].date().isoformat(),
                "entry_date": p["entry_date"].date().isoformat(),
                "exit_date": date.date().isoformat(),
                "entry_price": round(p["entry"], 4),
                "exit_price": round(exit_px, 4),
                "shares": shares, "stop": round(p["stop"], 4),
                "net_pnl": round(net, 2),
                "r_multiple": round(net / risk, 3) if risk > 0 else np.nan,
                "exit_reason": reason, "bars_held": p["bars"],
            })
            del positions[tkr]

        # ---- 3. evaluate new signals on today's close -> fill tomorrow --- #
        for tkr, d in prepped.items():
            if date not in d.index or tkr in positions:
                continue
            row = d.loc[date]
            if bool(row["long_signal"]):
                pending_entries.append({"ticker": tkr, "side": "LONG",
                                        "stop": float(row["long_stop"]),
                                        "signal_date": date})
            elif bool(row["short_signal"]):
                pending_entries.append({"ticker": tkr, "side": "SHORT",
                                        "stop": float(row["short_stop"]),
                                        "signal_date": date})

        # ---- 4. mark-to-market ------------------------------------------- #
        mtm = cash
        for tkr, p in positions.items():
            d = prepped[tkr]
            px = float(d.loc[date, "Close"]) if date in d.index \
                else float(d["Close"].asof(date))
            mtm += p["shares"] * px if p["side"] == "LONG" else -p["shares"] * px
        equity_rows.append({"date": date, "equity": mtm,
                            "open_positions": len(positions)})

    trades_df = pd.DataFrame(trades)
    equity_df = pd.DataFrame(equity_rows).set_index("date") if equity_rows \
        else pd.DataFrame(columns=["equity", "open_positions"])
    summary = summarize(trades_df, equity_df, capital)
    return trades_df, equity_df, summary


def summarize(trades: pd.DataFrame, equity: pd.DataFrame, capital: float) -> dict:
    s = {"trades": len(trades)}
    if len(equity):
        eq = equity["equity"]
        ret = eq.pct_change().dropna()
        yrs = max(1e-9, (eq.index[-1] - eq.index[0]).days / 365.25)
        s["final_equity"] = float(eq.iloc[-1])
        s["total_return"] = float(eq.iloc[-1] / capital - 1)
        s["cagr"] = float((eq.iloc[-1] / capital) ** (1 / yrs) - 1) if yrs > 0.05 else None
        s["sharpe"] = float(ret.mean() / ret.std() * np.sqrt(252)) if ret.std() > 0 else None
        peak = eq.cummax()
        s["max_dd"] = float(((peak - eq) / peak).max())
    if len(trades):
        wins = trades[trades["net_pnl"] > 0]
        losses = trades[trades["net_pnl"] <= 0]
        gw, gl = wins["net_pnl"].sum(), -losses["net_pnl"].sum()
        s["net_pnl"] = float(trades["net_pnl"].sum())
        s["win_rate"] = len(wins) / len(trades)
        s["profit_factor"] = float(gw / gl) if gl > 0 else float("inf")
        s["expectancy"] = float(trades["net_pnl"].mean())
        s["expectancy_r"] = float(trades["r_multiple"].dropna().mean()) \
            if trades["r_multiple"].notna().any() else None
        s["avg_bars"] = float(trades["bars_held"].mean())
    return s
