"""Trade simulation for the quote-imbalance forecasting signal.

Design assumptions (all flagged inline for review):
  * Entry: when |qimbal| >= L_star[sym], enter in the predicted direction.
            Cross the spread - buy at askpx (long) or sell at bidpx (short).
            Zero latency: fill at prevailing prices at signal time.
  * Exit:  Entry time + horizon_ms. Exit by crossing the spread back -
            sell at the prevailing bid (long) or buy at the prevailing ask (short).
            Prevailing prices at exit are the most recent observed prices
            at-or-before exit_time, found via merge_asof.
  * Filter: Apply the same filter_mid_change_events as the forecasting model,
            so simulation runs only on clean signals (no concurrent price move).
  * Sizing: 1 share per signal. Overlapping positions allowed - this is a
            per-signal P&L measurement, not a capital-constrained strategy.
  * EOD:   Drop trades whose exit_tstamp >= 16:00:00 (closing auction excluded).

The output P&L is GROSS of any fees/rebates and uses the prevailing displayed
quotes (no impact from our own order is modelled).
"""
import numpy as np
import pandas as pd

from forecast import compute_fcastdir, filter_mid_change_events


def _attach_exit_prices(
    trades: pd.DataFrame,
    book: pd.DataFrame,
    horizon_ms: int,
) -> pd.DataFrame:
    """Append exit-time prevailing bidpx/askpx to each trade row.

    Uses merge_asof (backward) to find the most recent quote at-or-before
    entry_tstamp + horizon_ms within the same symbol.
    """
    trades = trades.copy()
    trades["exit_tstamp"] = trades["tstamp"] + pd.Timedelta(milliseconds=horizon_ms)
    trades = trades.sort_values("exit_tstamp").reset_index(drop=True)
    book_sorted = book[["sym", "tstamp", "bidpx", "askpx"]].sort_values("tstamp")
    merged = pd.merge_asof(
        trades,
        book_sorted.rename(columns={"bidpx": "exit_bidpx",
                                     "askpx": "exit_askpx",
                                     "tstamp": "exit_quote_tstamp"}),
        by="sym",
        left_on="exit_tstamp",
        right_on="exit_quote_tstamp",
        direction="backward",
        allow_exact_matches=True,
    )
    return merged


def simulate_trades(
    df: pd.DataFrame,
    l_star: pd.Series,
    horizon_ms: int = 200,
    eod_cutoff: str = "16:00:00",
) -> pd.DataFrame:
    """Run the simulation on a feature-prepared, filtered DataFrame.

    Args:
        df: Output of filter_mid_change_events(prepare_features(...)).
            Must contain columns: sym, tstamp, bidpx, askpx, qimbal.
        l_star: Series indexed by symbol, mapping sym -> optimal cutoff L*.
        horizon_ms: Exit horizon in milliseconds (default 200, matches signal).
        eod_cutoff: Time-of-day after which trades are dropped.

    Returns:
        Per-trade DataFrame with columns:
          sym, tstamp, exit_tstamp, fcastdir, qimbal,
          entry_px, exit_px, pnl_per_share, pnl_bps, spread_at_entry, midpx
    """
    # 1) Apply per-symbol cutoffs to get fcastdir
    parts = []
    for sym, g in df.groupby("sym", sort=False):
        if sym not in l_star.index:
            continue
        gf = compute_fcastdir(g, float(l_star[sym]))
        parts.append(gf)
    df_with_fcast = pd.concat(parts, ignore_index=True)

    # 2) Keep only firing signals
    trades = df_with_fcast[df_with_fcast["fcastdir"] != 0].copy()
    if trades.empty:
        return trades

    # 3) Find prevailing bid/ask at exit time
    trades = _attach_exit_prices(trades, df, horizon_ms)

    # 4) Drop trades whose exit falls after EOD cutoff
    eod_per_day = trades["tstamp"].dt.normalize() + pd.Timedelta(eod_cutoff)
    trades = trades[trades["exit_tstamp"] < eod_per_day].copy()

    # 5) Drop trades where we couldn't find an exit quote
    trades = trades.dropna(subset=["exit_bidpx", "exit_askpx"]).copy()

    # 6) Compute entry and exit prices (cross-the-spread both ways)
    is_long = trades["fcastdir"] == 1
    trades["entry_px"] = np.where(is_long, trades["askpx"], trades["bidpx"])
    trades["exit_px"] = np.where(is_long, trades["exit_bidpx"], trades["exit_askpx"])

    # 7) P&L per share (signed by direction)
    sign = trades["fcastdir"].astype(float)
    trades["pnl_per_share"] = sign * (trades["exit_px"] - trades["entry_px"])
    trades["pnl_bps"] = 1e4 * trades["pnl_per_share"] / trades["midpx"]
    trades["spread_at_entry"] = trades["askpx"] - trades["bidpx"]

    cols = ["sym", "tstamp", "exit_tstamp", "fcastdir", "qimbal",
            "midpx", "spread_at_entry",
            "entry_px", "exit_px", "pnl_per_share", "pnl_bps"]
    return trades[cols].reset_index(drop=True)


def summarize(trades: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-trade results to per-symbol summary statistics."""
    if trades.empty:
        return pd.DataFrame()

    def _agg(g: pd.DataFrame) -> pd.Series:
        n = len(g)
        wins = (g["pnl_per_share"] > 0).sum()
        losses = (g["pnl_per_share"] < 0).sum()
        flat = (g["pnl_per_share"] == 0).sum()
        mean_pnl = g["pnl_per_share"].mean()
        std_pnl = g["pnl_per_share"].std(ddof=1) if n > 1 else np.nan
        sharpe_per_trade = mean_pnl / std_pnl if std_pnl and std_pnl > 0 else np.nan
        return pd.Series({
            "n_trades": n,
            "n_long": (g["fcastdir"] == 1).sum(),
            "n_short": (g["fcastdir"] == -1).sum(),
            "hit_rate": wins / n if n else np.nan,
            "win_loss_flat": f"{wins}/{losses}/{flat}",
            "mean_pnl_per_share": mean_pnl,
            "mean_pnl_bps": g["pnl_bps"].mean(),
            "total_pnl": g["pnl_per_share"].sum(),
            "sharpe_per_trade": sharpe_per_trade,
            "mean_spread": g["spread_at_entry"].mean(),
        })

    out = trades.groupby("sym").apply(_agg, include_groups=False)
    return out


def sweep_cutoffs(
    df: pd.DataFrame,
    cutoffs: list[float],
    horizon_ms: int = 200,
    eod_cutoff: str = "16:00:00",
) -> pd.DataFrame:
    """Run simulate_trades for each cutoff in `cutoffs`, applied uniformly to all symbols.

    Returns a long-format DataFrame with one row per (sym, L) summarising trade count,
    total P&L, mean P&L per share, mean P&L in bps, and hit rate.
    """
    syms = df["sym"].unique()
    rows = []
    for L in cutoffs:
        l_star = pd.Series({s: L for s in syms})
        trades = simulate_trades(df, l_star, horizon_ms=horizon_ms, eod_cutoff=eod_cutoff)
        if trades.empty:
            continue
        for sym, g in trades.groupby("sym"):
            rows.append({
                "sym": sym,
                "L": L,
                "n_trades": len(g),
                "total_pnl": g["pnl_per_share"].sum(),
                "mean_pnl_per_share": g["pnl_per_share"].mean(),
                "mean_pnl_bps": g["pnl_bps"].mean(),
                "hit_rate": (g["pnl_per_share"] > 0).mean(),
            })
    return pd.DataFrame(rows)


def best_cutoff_per_symbol(
    sweep_df: pd.DataFrame,
    metric: str = "total_pnl",
) -> pd.Series:
    """Pick the cutoff that maximises `metric` per symbol from a sweep result."""
    idx = sweep_df.groupby("sym")[metric].idxmax()
    return sweep_df.loc[idx].set_index("sym")["L"].rename("L_star_pnl")


def simulate_limit_trades(
    df: pd.DataFrame,
    l_star: pd.Series,
    horizon_ms: int = 200,
    eod_cutoff: str = "16:00:00",
) -> pd.DataFrame:
    """Limit-order variant: post passively on favored side, exit at horizon by crossing the spread.

    Design assumptions (review carefully):
      * Entry: when |qimbal| >= L_star[sym], post a limit on the favored side.
                fcastdir=+1 -> BUY limit at the prevailing bidpx.
                fcastdir=-1 -> SELL limit at the prevailing askpx.
      * Fill proxy (MAXIMUM-GENEROUS, perfect queue position assumed):
                BUY limit "fills" iff  nextbid < bidpx AND nextbidstamp-tstamp < horizon_ms.
                SELL limit "fills" iff nextask > askpx AND nextaskstamp-tstamp < horizon_ms.
                This uses only the IMMEDIATE next price change on the relevant side,
                so it is conservative against late fills (a bid that dips and recovers
                within horizon would be missed).
      * Exit:  signal_time + horizon_ms, cross the spread back at prevailing quotes
                (sell at bid for long, buy at ask for short). Market exit.
      * Sizing: 1 share per filled signal.
      * EOD:   Drop trades whose exit_tstamp >= eod_cutoff.

    Returns only FILLED trades (unfilled signals are cancelled with no P&L impact).
    The 'filled' column is preserved for diagnostics.
    """
    # 1) Apply per-symbol cutoffs
    parts = []
    for sym, g in df.groupby("sym", sort=False):
        if sym not in l_star.index:
            continue
        gf = compute_fcastdir(g, float(l_star[sym]))
        parts.append(gf)
    df_with_fcast = pd.concat(parts, ignore_index=True)

    # 2) Keep firing signals
    sig = df_with_fcast[df_with_fcast["fcastdir"] != 0].copy()
    if sig.empty:
        return sig

    # 3) Time-to-next bid/ask change (ms)
    sig["ttnextbid"] = (sig["nextbidstamp"] - sig["tstamp"]).dt.total_seconds() * 1000
    sig["ttnextask"] = (sig["nextaskstamp"] - sig["tstamp"]).dt.total_seconds() * 1000

    # 4) Fill proxy
    is_long = sig["fcastdir"] == 1
    buy_fill = is_long & (sig["nextbid"] < sig["bidpx"]) & (sig["ttnextbid"] < horizon_ms)
    sell_fill = (~is_long) & (sig["nextask"] > sig["askpx"]) & (sig["ttnextask"] < horizon_ms)
    sig["filled"] = (buy_fill | sell_fill).fillna(False)

    n_signals = len(sig)
    n_filled = int(sig["filled"].sum())
    fill_rate = n_filled / n_signals if n_signals else float("nan")

    trades = sig[sig["filled"]].copy()
    if trades.empty:
        return trades

    # 5) Entry price = the favored side (we got filled passively)
    trades["entry_px"] = np.where(trades["fcastdir"] == 1, trades["bidpx"], trades["askpx"])

    # 6) Find prevailing exit quotes via merge_asof
    trades = _attach_exit_prices(trades, df, horizon_ms)

    # 7) EOD filter and missing-quote drop
    eod_per_day = trades["tstamp"].dt.normalize() + pd.Timedelta(eod_cutoff)
    trades = trades[trades["exit_tstamp"] < eod_per_day].copy()
    trades = trades.dropna(subset=["exit_bidpx", "exit_askpx"]).copy()
    if trades.empty:
        return trades

    # 8) Exit price (market exit, cross the spread)
    is_long_t = trades["fcastdir"] == 1
    trades["exit_px"] = np.where(is_long_t, trades["exit_bidpx"], trades["exit_askpx"])

    # 9) P&L
    sign = trades["fcastdir"].astype(float)
    trades["pnl_per_share"] = sign * (trades["exit_px"] - trades["entry_px"])
    trades["pnl_bps"] = 1e4 * trades["pnl_per_share"] / trades["midpx"]
    trades["spread_at_entry"] = trades["askpx"] - trades["bidpx"]

    # Stash signal counts for reporting
    trades.attrs["n_signals"] = n_signals
    trades.attrs["n_filled"] = n_filled
    trades.attrs["fill_rate"] = fill_rate

    cols = ["sym", "tstamp", "exit_tstamp", "fcastdir", "qimbal",
            "midpx", "spread_at_entry",
            "entry_px", "exit_px", "pnl_per_share", "pnl_bps"]
    return trades[cols].reset_index(drop=True)


def simulate_inverted_limit_trades(
    df: pd.DataFrame,
    l_star: pd.Series,
    horizon_ms: int = 200,
    eod_cutoff: str = "16:00:00",
) -> pd.DataFrame:
    """Inverted limit-order variant: post AGAINST the prediction (market-maker mode).

    Reasoning: in simulate_limit_trades the fill mechanism is adversely selected -
    we get filled exactly when the signal is wrong. Here we invert: we post on the
    side that aggressors will hit when the signal IS right. We capture half-spread
    on entry but take a position OPPOSITE to the prediction.

    Entry rule:
      * fcastdir = +1: post SELL limit at askpx -> we are SHORT if filled.
      * fcastdir = -1: post BUY  limit at bidpx -> we are LONG  if filled.

    Fill proxy (same maximum-generous queue assumption as simulate_limit_trades):
      * SELL at ask filled iff nextask > askpx AND nextaskstamp - tstamp < horizon_ms.
      * BUY  at bid filled iff nextbid < bidpx AND nextbidstamp - tstamp < horizon_ms.

    Exit: signal_time + horizon_ms, cross the spread to flatten the position.
          Long exits at exit_bidpx, short exits at exit_askpx.

    Position sign is OPPOSITE to fcastdir; P&L is signed by position, not by prediction.
    """
    # 1) Apply per-symbol cutoffs
    parts = []
    for sym, g in df.groupby("sym", sort=False):
        if sym not in l_star.index:
            continue
        gf = compute_fcastdir(g, float(l_star[sym]))
        parts.append(gf)
    df_with_fcast = pd.concat(parts, ignore_index=True)

    sig = df_with_fcast[df_with_fcast["fcastdir"] != 0].copy()
    if sig.empty:
        return sig

    sig["ttnextbid"] = (sig["nextbidstamp"] - sig["tstamp"]).dt.total_seconds() * 1000
    sig["ttnextask"] = (sig["nextaskstamp"] - sig["tstamp"]).dt.total_seconds() * 1000

    # 2) Fill proxy on the OPPOSITE side from the prediction
    is_predict_up = sig["fcastdir"] == 1
    sell_fill = is_predict_up & (sig["nextask"] > sig["askpx"]) & (sig["ttnextask"] < horizon_ms)
    buy_fill = (~is_predict_up) & (sig["nextbid"] < sig["bidpx"]) & (sig["ttnextbid"] < horizon_ms)
    sig["filled"] = (sell_fill | buy_fill).fillna(False)

    n_signals = len(sig)
    n_filled = int(sig["filled"].sum())
    fill_rate = n_filled / n_signals if n_signals else float("nan")

    trades = sig[sig["filled"]].copy()
    if trades.empty:
        return trades

    # 3) Entry price = the side we posted on (opposite to fcastdir)
    trades["entry_px"] = np.where(trades["fcastdir"] == 1, trades["askpx"], trades["bidpx"])

    # 4) Position sign = OPPOSITE of fcastdir
    position = -trades["fcastdir"]

    # 5) Find prevailing exit quotes
    trades = _attach_exit_prices(trades, df, horizon_ms)
    eod_per_day = trades["tstamp"].dt.normalize() + pd.Timedelta(eod_cutoff)
    trades = trades[trades["exit_tstamp"] < eod_per_day].copy()
    trades = trades.dropna(subset=["exit_bidpx", "exit_askpx"]).copy()
    if trades.empty:
        return trades

    # 6) Exit: long sells at bid, short buys back at ask
    position = -trades["fcastdir"]  # re-derive after filtering
    is_long = position == 1
    trades["exit_px"] = np.where(is_long, trades["exit_bidpx"], trades["exit_askpx"])

    # 7) P&L is signed by position
    sign = position.astype(float)
    trades["pnl_per_share"] = sign * (trades["exit_px"] - trades["entry_px"])
    trades["pnl_bps"] = 1e4 * trades["pnl_per_share"] / trades["midpx"]
    trades["spread_at_entry"] = trades["askpx"] - trades["bidpx"]

    trades.attrs["n_signals"] = n_signals
    trades.attrs["n_filled"] = n_filled
    trades.attrs["fill_rate"] = fill_rate

    cols = ["sym", "tstamp", "exit_tstamp", "fcastdir", "qimbal",
            "midpx", "spread_at_entry",
            "entry_px", "exit_px", "pnl_per_share", "pnl_bps"]
    return trades[cols].reset_index(drop=True)


def cumulative_pnl(trades: pd.DataFrame) -> pd.DataFrame:
    """Return per-symbol cumulative P&L over time (for plotting)."""
    if trades.empty:
        return trades
    t = trades.sort_values("tstamp").copy()
    t["cum_pnl"] = t.groupby("sym")["pnl_per_share"].cumsum()
    return t[["sym", "tstamp", "pnl_per_share", "cum_pnl"]]
