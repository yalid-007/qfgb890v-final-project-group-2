import os
import pandas as pd


def load_nbbosz(date: str, data_dir: str = "data") -> pd.DataFrame:
    """Load NBBO size-change events for a single trading date.

    Args:
        date: Trading date in 'YYYY-MM-DD' format.
        data_dir: Path to the folder containing .pqt files.

    Returns:
        DataFrame sorted by (sym, tstamp) with original schema.
        Columns: date, sym, tstamp, bidpx, askpx, bidsz, asksz,
                 nextbidstamp, nextbid, nextaskstamp, nextask.
    """
    date_compact = date.replace("-", "")
    path = os.path.join(data_dir, f"nbbosz_{date_compact}.pqt")
    df = pd.read_parquet(path)
    df = df.sort_values(["sym", "tstamp"]).reset_index(drop=True)
    return df
