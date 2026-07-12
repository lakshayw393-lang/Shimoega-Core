"""
Shiomega Daily — core engine (no Streamlit imports; unit-testable).

潮目が変わる — "the tide has turned."
Sequence: prior trend -> TK cross on the wrong side of the Kumo (heads-up)
-> WAIT -> clean Chikou breakout (trigger) -> stop at Kijun/swing
-> ride while the Kijun is respected.

Conventions match the Python backtest/screener suite:
$100K capital, $5K/position, next-bar-open fills, Ichimoku 9/26/52 (disp. 26).

pandas 3.0 note: .shift() on a boolean Series yields object dtype, which
breaks `~` negation — every boolean shift below is followed by .astype(bool).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TENKAN, KIJUN, SENKOU_B, DISP = 9, 26, 52, 26
CHIKOU_WIN = 5          # 11-bar window centred on the Chikou position
CROSS_MAX_AGE = 40      # trigger must arrive within 40 bars of the cross
CAPITAL = 100_000.0


# --------------------------------------------------------------------------- #
# Ichimoku
# --------------------------------------------------------------------------- #
def ichimoku(df: pd.DataFrame) -> pd.DataFrame:
    """Add tenkan/kijun/span_a/span_b (RAW, undisplaced) + chikou columns."""
    out = df.copy()
    hi, lo = out["High"], out["Low"]
    out["tenkan"] = (hi.rolling(TENKAN).max() + lo.rolling(TENKAN).min()) / 2
    out["kijun"] = (hi.rolling(KIJUN).max() + lo.rolling(KIJUN).min()) / 2
    out["span_a"] = (out["tenkan"] + out["kijun"]) / 2
    out["span_b"] = (hi.rolling(SENKOU_B).max() + lo.rolling(SENKOU_B).min()) / 2
    # cloud in effect at bar i = raw span values from i-26
    out["cloud_a"] = out["span_a"].shift(DISP)
    out["cloud_b"] = out["span_b"].shift(DISP)
    out["cloud_top"] = out[["cloud_a", "cloud_b"]].max(axis=1)
    out["cloud_bot"] = out[["cloud_a", "cloud_b"]].min(axis=1)
    return out


def gold_cross(df: pd.DataFrame) -> pd.Series:
    above = df["tenkan"] > df["kijun"]
    prev = above.shift(1).astype(bool)          # pandas 3.0 fix
    return above & ~prev & df["tenkan"].shift(1).notna()


def death_cross(df: pd.DataFrame) -> pd.Series:
    below = df["tenkan"] < df["kijun"]
    prev = below.shift(1).astype(bool)          # pandas 3.0 fix
    return below & ~prev & df["tenkan"].shift(1).notna()


def chikou_break_up(df: pd.DataFrame) -> pd.Series:
    """Close clears the 11-bar HIGH window centred on the Chikou position
    (bars i-31 … i-21)."""
    ceiling = df["High"].rolling(2 * CHIKOU_WIN + 1).max().shift(DISP - CHIKOU_WIN)
    return df["Close"] > ceiling


def chikou_break_down(df: pd.DataFrame) -> pd.Series:
    floor = df["Low"].rolling(2 * CHIKOU_WIN + 1).min().shift(DISP - CHIKOU_WIN)
    return df["Close"] < floor


# --------------------------------------------------------------------------- #
# Setup search (for chart annotation): first valid cross -> break pair
# --------------------------------------------------------------------------- #
def find_setup(df: pd.DataFrame, side: str) -> dict:
    d = ichimoku(df)
    if side == "long":
        crosses = np.flatnonzero((gold_cross(d) & (d["Close"] < d["cloud_bot"])).to_numpy())
        breaks = np.flatnonzero(chikou_break_up(d).fillna(False).to_numpy())
    else:
        crosses = np.flatnonzero((death_cross(d) & (d["Close"] > d["cloud_top"])).to_numpy())
        breaks = np.flatnonzero(chikou_break_down(d).fillna(False).to_numpy())

    for c in crosses:
        later = breaks[(breaks >= c) & (breaks - c <= CROSS_MAX_AGE)]
        if later.size:
            b = int(later[0])
            if side == "long":
                stop = float(min(d["kijun"].iloc[b], d["Low"].iloc[max(0, b - 10): b + 1].min()))
            else:
                stop = float(max(d["kijun"].iloc[b], d["High"].iloc[max(0, b - 10): b + 1].max()))
            return {"cross": int(c), "trigger": b, "stop": stop, "df": d}
    return {"cross": None, "trigger": None, "stop": None, "df": d}


# --------------------------------------------------------------------------- #
# Synthetic daily V-reversal (same shape as the verified JS generator)
# --------------------------------------------------------------------------- #
def gen_series(side: str = "long", seed: int = 7, n: int = 200,
               reverse_tail: bool = False) -> pd.DataFrame:
    """Synthetic daily V-reversal that produces a valid Shiomega setup.

    reverse_tail=True bends the recovery back down after the setup so a
    long position will eventually hit its Kijun trail / stop — used in tests
    to exercise the exit paths (the default clean trend never closes)."""
    rng = np.random.default_rng(seed)
    close = np.empty(n)
    opn = np.empty(n)
    px = 100.0
    for i in range(n):
        if i < 100:
            drift = -0.55
        elif i < 112:
            drift = 0.95
        elif reverse_tail and i >= 150:
            drift = -0.85          # roll the recovery back over
        else:
            drift = 0.26
        px += drift + (rng.random() - 0.5) * 0.9
        opn[i] = px - drift * 0.5 + (rng.random() - 0.5) * 0.3
        close[i] = px
    hi = np.maximum(opn, close) + rng.random(n) * 0.55
    lo = np.minimum(opn, close) - rng.random(n) * 0.55
    if side == "short":
        mean = close.mean()
        close, opn = 2 * mean - close, 2 * mean - opn
        hi, lo = 2 * mean - lo, 2 * mean - hi
    # bdate_range rolls a weekend end-date back and can come up one short:
    # over-generate and slice to exactly n.
    idx = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=n + 5)[-n:]
    return pd.DataFrame({"Open": opn, "High": hi, "Low": lo, "Close": close}, index=idx)


# --------------------------------------------------------------------------- #
# Tolerant CSV normalisation
# --------------------------------------------------------------------------- #
def _norm_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [
        str(c).strip().lower().replace("%", "pct").replace("$", "usd")
        .replace(" ", "_").replace("-", "_").replace("/", "_").strip("_")
        for c in df.columns
    ]
    return df


def _pick(df: pd.DataFrame, names: list[str]) -> pd.Series | None:
    for n in names:
        if n in df.columns:
            return df[n]
    return None


def _num(s: pd.Series | None) -> pd.Series | None:
    if s is None:
        return None
    return pd.to_numeric(
        s.astype(str).str.replace(",", "", regex=False).str.replace("$", "", regex=False),
        errors="coerce",
    )


def normalise_trades(raw: pd.DataFrame) -> pd.DataFrame:
    """Map a trade-log CSV with drifting header names onto a fixed schema.
    Derives pnl from prices×shares (side-aware) and R from the stop distance
    when those columns are absent."""
    df = _norm_cols(raw)
    side = _pick(df, ["side", "direction"])
    side = (side.astype(str).str.upper().str.startswith("S")).map({True: "SHORT", False: "LONG"}) \
        if side is not None else pd.Series("LONG", index=df.index)

    out = pd.DataFrame({
        "ticker": _pick(df, ["ticker", "symbol"]),
        "side": side,
        "entry_date": _pick(df, ["entry_date", "entry_time", "entry_dt", "open_date"]),
        "exit_date": _pick(df, ["exit_date", "exit_time", "exit_dt", "close_date"]),
        "entry": _num(_pick(df, ["entry_price", "entry", "fill_price", "open_price"])),
        "exit": _num(_pick(df, ["exit_price", "avg_exit_price", "exit", "close_price"])),
        "shares": _num(_pick(df, ["shares", "qty", "quantity", "size"])),
        "stop": _num(_pick(df, ["stop", "stop_loss", "stop_price", "initial_stop"])),
        "pnl": _num(_pick(df, ["net_pnl", "pnl_usd", "pnl", "profit", "pl"])),
        "r": _num(_pick(df, ["r_multiple", "pnl_r", "r_mult", "r"])),
        "reason": _pick(df, ["exit_reason", "reason"]),
        "bars": _num(_pick(df, ["bars_held", "days_held", "bars"])),
    })
    out["ticker"] = out["ticker"].fillna("?") if out["ticker"] is not None else "?"
    out["reason"] = out["reason"].astype(str).str.upper().replace({"NAN": "", "NONE": ""})

    sgn = np.where(out["side"].eq("SHORT"), -1.0, 1.0)
    derived_pnl = (out["exit"] - out["entry"]) * out["shares"] * sgn
    out["pnl"] = out["pnl"].fillna(pd.Series(derived_pnl, index=out.index))

    risk = (out["entry"] - out["stop"]).abs() * out["shares"]
    derived_r = out["pnl"] / risk.where(risk > 0)
    out["r"] = out["r"].fillna(derived_r)

    for c in ("entry_date", "exit_date"):
        out[c] = pd.to_datetime(out[c], errors="coerce")
    return out.dropna(subset=["pnl"]).reset_index(drop=True)


def normalise_signals(raw: pd.DataFrame) -> pd.DataFrame:
    df = _norm_cols(raw)
    status = _pick(df, ["status"])
    status = status.astype(str).str.upper().str.startswith("W").map(
        {True: "WATCHLIST", False: "TRIGGERED"}) if status is not None else "TRIGGERED"
    side = _pick(df, ["side", "direction"])
    side = side.astype(str).str.upper().str.startswith("S").map(
        {True: "SHORT", False: "LONG"}) if side is not None else "LONG"
    return pd.DataFrame({
        "ticker": _pick(df, ["ticker", "symbol"]),
        "status": status,
        "side": side,
        "date": _pick(df, ["date", "signal_date", "bar_time", "bar_date"]),
        "close": _num(_pick(df, ["close", "last", "price", "last_close"])),
        "entry": _num(_pick(df, ["entry", "entry_price", "next_open_est"])),
        "stop": _num(_pick(df, ["stop", "stop_loss", "stop_price"])),
        "risk_pct": _num(_pick(df, ["risk_pct", "riskpct", "risk"])),
        "bars_since_cross": _num(_pick(df, ["bars_since_cross", "cross_age",
                                            "bars_since", "days_since_cross"])),
    })


# --------------------------------------------------------------------------- #
# Backtest metrics off the $100K base
# --------------------------------------------------------------------------- #
def compute_metrics(trades: pd.DataFrame, capital: float = CAPITAL) -> dict:
    t = trades.sort_values("exit_date", na_position="last").reset_index(drop=True)
    n = len(t)
    wins, losses = t[t["pnl"] > 0], t[t["pnl"] <= 0]
    gross_w = float(wins["pnl"].sum())
    gross_l = float(-losses["pnl"].sum())
    net = gross_w - gross_l

    curve = capital + t["pnl"].cumsum()
    curve = pd.concat([pd.Series([capital]), curve], ignore_index=True)
    peak = curve.cummax()
    max_dd = float(((peak - curve) / peak).max()) if n else 0.0

    cagr = None
    d0 = t["entry_date"].min() if n else pd.NaT
    d1 = t["exit_date"].max() if n else pd.NaT
    if pd.notna(d0) and pd.notna(d1) and d1 > d0:
        yrs = (d1 - d0).days / 365.25
        if yrs > 0.05:
            cagr = float((curve.iloc[-1] / capital) ** (1 / yrs) - 1)

    r_vals = t["r"].dropna()
    return {
        "n": n,
        "net": net,
        "win_rate": len(wins) / n if n else 0.0,
        "profit_factor": (gross_w / gross_l) if gross_l > 0 else (np.inf if gross_w > 0 else 0.0),
        "expectancy": net / n if n else 0.0,
        "expectancy_r": float(r_vals.mean()) if len(r_vals) else None,
        "avg_win": float(wins["pnl"].mean()) if len(wins) else 0.0,
        "avg_loss": float(losses["pnl"].mean()) if len(losses) else 0.0,
        "max_dd": max_dd,
        "cagr": cagr,
        "curve": curve,
        "r_vals": r_vals,
        "reasons": t["reason"].replace("", "OTHER").value_counts().to_dict(),
        "long_n": int((t["side"] == "LONG").sum()),
        "long_pnl": float(t.loc[t["side"] == "LONG", "pnl"].sum()),
        "short_n": int((t["side"] == "SHORT").sum()),
        "short_pnl": float(t.loc[t["side"] == "SHORT", "pnl"].sum()),
        "avg_bars": float(t["bars"].dropna().mean()) if t["bars"].notna().any() else None,
        "trades": t,
    }


# --------------------------------------------------------------------------- #
# Demo data (illustrative shapes only — NOT results)
# --------------------------------------------------------------------------- #
def demo_trades(seed: int = 42, n: int = 120) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    tickers = ["AAPL", "MSFT", "NVDA", "AMD", "SHOP.TO", "CNQ.TO", "TSLA", "META",
               "SU.TO", "GOOGL", "AVGO", "ENB.TO", "COST", "PYPL", "UBER", "MRNA"]
    rows, day = [], pd.Timestamp("2024-06-03")
    for _ in range(n):
        day += pd.Timedelta(days=int(2 + rng.random() * 5))
        side = "LONG" if rng.random() < 0.68 else "SHORT"
        px = 20 + rng.random() * 180
        shares = max(1, int(5000 // px))
        risk_ps = px * (0.04 + rng.random() * 0.05)
        stop = px - risk_ps if side == "LONG" else px + risk_ps
        win = rng.random() < 0.45
        r = (0.6 + rng.random() * 2.9) if win else -(0.55 + rng.random() * 0.55)
        pnl = r * risk_ps * shares
        move = (1 if side == "LONG" else -1) * r * risk_ps
        bars = max(2, round((18 + rng.random() * 40) if win else (5 + rng.random() * 14)))
        rows.append({
            "ticker": tickers[int(rng.random() * len(tickers))], "side": side,
            "entry_date": day.date().isoformat(),
            "exit_date": (day + pd.Timedelta(days=int(bars * 1.45))).date().isoformat(),
            "entry_price": round(px, 2), "exit_price": round(px + move, 2),
            "shares": shares, "stop": round(stop, 2), "net_pnl": round(pnl, 2),
            "r_multiple": round(r, 2),
            "exit_reason": "KIJUN_TRAIL" if win else "HARD_STOP", "bars_held": bars,
        })
    return pd.DataFrame(rows)


def demo_signals() -> pd.DataFrame:
    return pd.DataFrame([
        {"ticker": "NVAX", "status": "TRIGGERED", "side": "LONG", "date": "2026-07-10",
         "close": 14.62, "entry": 14.70, "stop": 13.55, "risk_pct": 7.8, "bars_since_cross": 12},
        {"ticker": "CVE.TO", "status": "TRIGGERED", "side": "LONG", "date": "2026-07-10",
         "close": 29.84, "entry": 29.95, "stop": 28.40, "risk_pct": 5.2, "bars_since_cross": 7},
        {"ticker": "MRNA", "status": "TRIGGERED", "side": "SHORT", "date": "2026-07-10",
         "close": 88.10, "entry": 87.90, "stop": 93.25, "risk_pct": 6.1, "bars_since_cross": 15},
        {"ticker": "SHOP.TO", "status": "WATCHLIST", "side": "LONG", "date": "2026-07-10",
         "close": 102.35, "entry": None, "stop": 96.80, "risk_pct": None, "bars_since_cross": 4},
        {"ticker": "AMD", "status": "WATCHLIST", "side": "LONG", "date": "2026-07-10",
         "close": 171.20, "entry": None, "stop": 164.10, "risk_pct": None, "bars_since_cross": 9},
        {"ticker": "PYPL", "status": "WATCHLIST", "side": "SHORT", "date": "2026-07-10",
         "close": 61.44, "entry": None, "stop": 64.90, "risk_pct": None, "bars_since_cross": 3},
    ])
