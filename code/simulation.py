"""Trade simulation for the quote-imbalance signal.

All concepts and terminology come directly from the lecture notes:

  Lecture 4 - quote imbalance I_t and its predictive value for next mid move.
  Lecture 5 - round-trip P&L is the only benchmark-independent P&L; we use it.
  Lecture 6 - market/limit order duality. A market order pays half-spread;
              a limit order gains half-spread if filled, but is exposed to
              adverse selection.
  Lecture 7 - aggressive (market) orders pay the spread up-front and control
              execution time; passive (limit) orders gain the spread but
              face adverse selection and volatility risk.

The signal: when |I_t| >= L*_sym, take a 1-share position in the predicted
direction, hold for delta_t = 200 ms, then close. We measure the round-trip
P&L per share and in basis points of midpx_t at signal time.

We compare two execution styles, both taught in class:

  AGGRESSIVE (market order on entry, market order on exit)
    Long:  buy at askpx_t,  sell at exit_bidpx
    Short: sell at bidpx_t, buy  at exit_askpx
    Pays a full bid-ask spread round-trip.

  PASSIVE  (limit order on entry, market order on exit)
    Long:  post buy limit at bidpx_t. Filled iff the bid is consumed within
           delta_t (proxied by nextbidstamp - tstamp < delta_t and
           nextbid < bidpx_t). Exit by selling at exit_bidpx.
    Short: post sell limit at askpx_t. Filled iff the ask is consumed within
           delta_t. Exit by buying at exit_askpx.
    Gains half-spread on entry IF filled. The fill mechanism is exactly the
    adverse-selection channel from Lecture 6: the order fills when the
    favored side is being lifted, i.e. when the market is moving against the
    forecast. We expect fills concentrated on losing trades.

Per-share P&L: pnl = direction * (exit_px - entry_px),
where direction = +1 for long, -1 for short.
P&L in bps: 1e4 * pnl / midpx_t.

Sizing is 1 share per signal. EOD trades whose 200ms exit lands at or after
16:00:00 are dropped.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from forecast import compute_fcastdir, filter_mid_change_events


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _attach_exit_quotes(
    trades: pd.DataFrame,
    book: pd.DataFrame,
    horizon_ms: int,
) -> pd.DataFrame:
    """Find the prevailing (bidpx, askpx) at tstamp + horizon_ms via merge_asof."""
    trades = trades.copy()
    trades["exit_tstamp"] = trades["tstamp"] + pd.Timedelta(milliseconds=horizon_ms)
    trades = trades.sort_values("exit_tstamp").reset_index(drop=True)
    book_q = (
        book[["sym", "tstamp", "bidpx", "askpx"]]
        .sort_values("tstamp")
        .rename(columns={"bidpx": "exit_bidpx", "askpx": "exit_askpx"})
    )
    return pd.merge_asof(
        trades,
        book_q,
        by="sym",
        left_on="exit_tstamp",
        right_on="tstamp",
        direction="backward",
        allow_exact_matches=True,
        suffixes=("", "_quote"),
    )


def _apply_l_star(df: pd.DataFrame, l_star: pd.Series) -> pd.DataFrame:
    """Apply per-symbol cutoff and tag fcastdir."""
    parts = []
    for sym, g in df.groupby("sym", sort=False):
        if sym not in l_star.index:
            continue
        parts.append(compute_fcastdir(g, float(l_star[sym])))
    return pd.concat(parts, ignore_index=True)


# ---------------------------------------------------------------------------
# Aggressive: market order on entry, market order on exit
# ---------------------------------------------------------------------------

def simulate_aggressive(
    df: pd.DataFrame,
    l_star: pd.Series,
    horizon_ms: int = 200,
    eod_cutoff: str = "16:00:00",
) -> pd.DataFrame:
    """Cross the spread on both legs.

    Input df should be filter_mid_change_events(prepare_features(...)).
    """
    df_f = _apply_l_star(df, l_star)
    trades = df_f[df_f["fcastdir"] != 0].copy()
    if trades.empty:
        return trades

    trades = _attach_exit_quotes(trades, df, horizon_ms)
    eod = trades["tstamp"].dt.normalize() + pd.Timedelta(eod_cutoff)
    trades = trades[trades["exit_tstamp"] < eod]
    trades = trades.dropna(subset=["exit_bidpx", "exit_askpx"]).copy()

    is_long = trades["fcastdir"] == 1
    trades["entry_px"] = np.where(is_long, trades["askpx"], trades["bidpx"])
    trades["exit_px"]  = np.where(is_long, trades["exit_bidpx"], trades["exit_askpx"])

    direction = trades["fcastdir"].astype(float)
    trades["pnl_per_share"] = direction * (trades["exit_px"] - trades["entry_px"])
    trades["pnl_bps"]       = 1e4 * trades["pnl_per_share"] / trades["midpx"]
    trades["spread"]        = trades["askpx"] - trades["bidpx"]

    cols = ["sym", "tstamp", "exit_tstamp", "fcastdir", "qimbal",
            "midpx", "spread", "entry_px", "exit_px",
            "pnl_per_share", "pnl_bps"]
    return trades[cols].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Passive: limit order on entry (gains half-spread if filled), market exit
# ---------------------------------------------------------------------------

def simulate_passive(
    df: pd.DataFrame,
    l_star: pd.Series,
    horizon_ms: int = 200,
    eod_cutoff: str = "16:00:00",
) -> tuple[pd.DataFrame, dict]:
    """Post a passive limit on the favored side; cross the spread to flatten.

    Fill proxy (assumes front-of-queue, i.e. an upper bound on fill rate):
      * Long:  buy limit at bidpx_t fills iff nextbid < bidpx_t and
               nextbidstamp - tstamp < horizon_ms.
      * Short: sell limit at askpx_t fills iff nextask > askpx_t and
               nextaskstamp - tstamp < horizon_ms.

    Unfilled signals are cancelled (no P&L). The diagnostic dict carries
    n_signals, n_filled, fill_rate per symbol so we can comment on
    adverse selection at the symbol level.

    Returns (trades_df, diagnostics_dict).
    """
    df_f = _apply_l_star(df, l_star)
    sig = df_f[df_f["fcastdir"] != 0].copy()
    if sig.empty:
        return sig, {}

    # time-to-next bid/ask change in ms
    sig["ttnextbid"] = (sig["nextbidstamp"] - sig["tstamp"]).dt.total_seconds() * 1000
    sig["ttnextask"] = (sig["nextaskstamp"] - sig["tstamp"]).dt.total_seconds() * 1000

    is_long = sig["fcastdir"] == 1
    long_fill  = is_long  & (sig["nextbid"] < sig["bidpx"]) & (sig["ttnextbid"] < horizon_ms)
    short_fill = (~is_long) & (sig["nextask"] > sig["askpx"]) & (sig["ttnextask"] < horizon_ms)
    sig["filled"] = (long_fill | short_fill).fillna(False)

    diag = (
        sig.groupby("sym")["filled"]
        .agg(n_signals="count", n_filled="sum")
        .assign(fill_rate=lambda x: x["n_filled"] / x["n_signals"])
    )

    trades = sig[sig["filled"]].copy()
    if trades.empty:
        return trades, {"per_symbol": diag}

    # entry on the favored side (gain half-spread)
    trades["entry_px"] = np.where(trades["fcastdir"] == 1,
                                  trades["bidpx"], trades["askpx"])

    trades = _attach_exit_quotes(trades, df, horizon_ms)
    eod = trades["tstamp"].dt.normalize() + pd.Timedelta(eod_cutoff)
    trades = trades[trades["exit_tstamp"] < eod]
    trades = trades.dropna(subset=["exit_bidpx", "exit_askpx"]).copy()
    if trades.empty:
        return trades, {"per_symbol": diag}

    is_long_t = trades["fcastdir"] == 1
    trades["exit_px"] = np.where(is_long_t, trades["exit_bidpx"], trades["exit_askpx"])

    direction = trades["fcastdir"].astype(float)
    trades["pnl_per_share"] = direction * (trades["exit_px"] - trades["entry_px"])
    trades["pnl_bps"]       = 1e4 * trades["pnl_per_share"] / trades["midpx"]
    trades["spread"]        = trades["askpx"] - trades["bidpx"]

    cols = ["sym", "tstamp", "exit_tstamp", "fcastdir", "qimbal",
            "midpx", "spread", "entry_px", "exit_px",
            "pnl_per_share", "pnl_bps"]
    return trades[cols].reset_index(drop=True), {"per_symbol": diag}


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def summarize(trades: pd.DataFrame) -> pd.DataFrame:
    """Per-symbol summary: trade count, hit rate, mean P&L, total P&L."""
    if trades.empty:
        return pd.DataFrame()

    def _agg(g: pd.DataFrame) -> pd.Series:
        n = len(g)
        wins = (g["pnl_per_share"] > 0).sum()
        return pd.Series({
            "n_trades": n,
            "hit_rate": wins / n if n else np.nan,
            "mean_pnl_per_share": g["pnl_per_share"].mean(),
            "mean_pnl_bps": g["pnl_bps"].mean(),
            "total_pnl": g["pnl_per_share"].sum(),
            "mean_spread": g["spread"].mean(),
        })

    return trades.groupby("sym").apply(_agg, include_groups=False)
