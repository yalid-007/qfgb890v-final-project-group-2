import numpy as np
import pandas as pd

_FAR_FUTURE = pd.Timestamp("2099-01-01")


def compute_midpx(df: pd.DataFrame) -> pd.DataFrame:
    """Add midpx = (bidpx + askpx) / 2."""
    df = df.copy()
    df["midpx"] = (df["bidpx"] + df["askpx"]) / 2
    return df


def compute_qimbal(df: pd.DataFrame) -> pd.DataFrame:
    """Add qimbal = (bidsz - asksz) / (bidsz + asksz)."""
    df = df.copy()
    df["qimbal"] = (df["bidsz"] - df["asksz"]) / (df["bidsz"] + df["asksz"])
    return df


def compute_next_mid(df: pd.DataFrame) -> pd.DataFrame:
    """Add nextmid and ttnextmid (ms) from the earlier of the next bid/ask price change.

    Validity is determined by whether nextbid/nextask prices are non-null.
    End-of-day sentinel rows (price is NaN) yield nextmid=NaN and ttnextmid=NaN.
    """
    df = df.copy()

    bid_stamp = df["nextbidstamp"]
    ask_stamp = df["nextaskstamp"]
    bid_valid = df["nextbid"].notna()
    ask_valid = df["nextask"].notna()

    # Next mid-change time: min of the two stamps, treating invalid side as +infinity
    bid_for_min = bid_stamp.where(bid_valid, _FAR_FUTURE)
    ask_for_min = ask_stamp.where(ask_valid, _FAR_FUTURE)
    next_stamp_filled = bid_for_min.where(bid_for_min <= ask_for_min, ask_for_min)
    # Restore NaT where no valid price change exists on either side
    next_stamp = next_stamp_filled.where(bid_valid | ask_valid, pd.NaT)

    # Which sides change at next_stamp
    bid_changes = bid_valid & (bid_stamp == next_stamp)
    ask_changes = ask_valid & (ask_stamp == next_stamp)

    new_bid = df["bidpx"].where(~bid_changes, df["nextbid"])
    new_ask = df["askpx"].where(~ask_changes, df["nextask"])

    df["nextmid"] = ((new_bid + new_ask) / 2).where(bid_valid | ask_valid)
    df["ttnextmid"] = (next_stamp - df["tstamp"]).dt.total_seconds() * 1000

    return df


def compute_nextdir(df: pd.DataFrame, delta_t_ms: int = 200) -> pd.DataFrame:
    """Add nextdir: 1 (up), -1 (down), or 0 (no directional change within horizon).

    Rows where ttnextmid >= delta_t_ms or nextmid is NaN receive nextdir = 0.
    """
    df = df.copy()
    mid = df["midpx"]
    next_mid = df["nextmid"]
    tt = df["ttnextmid"]

    within_horizon = tt < delta_t_ms  # False when tt is NaN

    df["nextdir"] = np.where(
        within_horizon & (next_mid > mid), 1,
        np.where(within_horizon & (next_mid < mid), -1, 0),
    ).astype(int)

    return df


def prepare_features(df: pd.DataFrame, delta_t_ms: int = 200) -> pd.DataFrame:
    """Compute midpx, qimbal, nextmid, ttnextmid, and nextdir for a full day DataFrame."""
    df = compute_midpx(df)
    df = compute_qimbal(df)
    df = compute_next_mid(df)
    df = compute_nextdir(df, delta_t_ms)
    return df
