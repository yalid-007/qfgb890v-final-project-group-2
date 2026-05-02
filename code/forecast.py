import numpy as np
import pandas as pd

LABELS = [-1, 0, 1]


def filter_mid_change_events(df: pd.DataFrame) -> pd.DataFrame:
    """Remove rows where midpx changed from the previous event within each symbol.

    These are events where a price change coincided with the size change,
    so the imbalance reflects the new price rather than predicting it.
    The first row of each symbol is always dropped (no prior reference).
    """
    prev_mid = df.groupby("sym", sort=False)["midpx"].shift(1)
    return df[df["midpx"] == prev_mid].reset_index(drop=True)


def compute_fcastdir(df: pd.DataFrame, L: float) -> pd.DataFrame:
    """Add fcastdir using symmetric threshold L.

    fcastdir = 1 if qimbal >= L, -1 if qimbal <= -L, 0 otherwise.
    """
    df = df.copy()
    q = df["qimbal"]
    df["fcastdir"] = np.where(q >= L, 1, np.where(q <= -L, -1, 0)).astype(int)
    return df


def confusion_matrix_df(y_true: pd.Series, y_pred: pd.Series) -> pd.DataFrame:
    """Return a 3×3 confusion matrix with labels {-1, 0, 1}.

    Rows = actual class, Columns = predicted class.
    """
    cm = pd.crosstab(y_true, y_pred)
    cm = cm.reindex(index=LABELS, columns=LABELS, fill_value=0)
    cm.index.name = "actual"
    cm.columns.name = "predicted"
    return cm


def f1_per_class(cm: pd.DataFrame) -> pd.Series:
    """Compute F1 score for each class from a confusion matrix.

    Returns a Series indexed by class label. Zero-safe: returns 0.0 when
    both precision and recall are zero.
    """
    scores = {}
    for c in LABELS:
        tp = cm.loc[c, c]
        precision = tp / cm[c].sum() if cm[c].sum() > 0 else 0.0
        recall = tp / cm.loc[c].sum() if cm.loc[c].sum() > 0 else 0.0
        denom = precision + recall
        scores[c] = 2 * precision * recall / denom if denom > 0 else 0.0
    return pd.Series(scores, name="f1")


def macro_f1_from_cm(cm: pd.DataFrame) -> float:
    """Compute macro-average F1 from a confusion matrix."""
    return float(f1_per_class(cm).mean())


def macro_f1(y_true: pd.Series, y_pred: pd.Series) -> float:
    """Compute macro-average F1 directly from two label series."""
    return macro_f1_from_cm(confusion_matrix_df(y_true, y_pred))


def optimize_cutoff(
    df: pd.DataFrame,
    cutoffs: list[float],
) -> tuple[pd.DataFrame, pd.Series]:
    """Scan cutoffs per symbol and return the F1 table and optimal L* per symbol.

    Args:
        df: Filtered DataFrame with columns 'sym', 'qimbal', 'nextdir'.
            Call filter_mid_change_events before passing here.
        cutoffs: Cutoff values to scan (symmetric: L_up = -L_lo = L).

    Returns:
        f1_table: DataFrame (rows=symbol, columns=cutoff) of macro F1 scores.
        l_star: Series (index=symbol) with the cutoff that maximizes macro F1.
    """
    results = {}
    for sym, g in df.groupby("sym"):
        row = {}
        for L in cutoffs:
            g_f = compute_fcastdir(g, L)
            row[L] = macro_f1(g_f["nextdir"], g_f["fcastdir"])
        results[sym] = row

    f1_table = pd.DataFrame(results).T
    f1_table.index.name = "sym"
    f1_table.columns.name = "cutoff"

    l_star = f1_table.idxmax(axis=1).rename("L_star")
    return f1_table, l_star


def eval_cutoffs(
    df: pd.DataFrame,
    l_star: pd.Series,
) -> pd.Series:
    """Apply per-symbol optimal cutoffs and return macro F1 per symbol.

    Args:
        df: Filtered DataFrame with columns 'sym', 'qimbal', 'nextdir'.
        l_star: Series with index=symbol, values=cutoff (from optimize_cutoff).

    Returns:
        Series (index=symbol) of macro F1 scores.
    """
    results = {}
    for sym, g in df.groupby("sym"):
        g_f = compute_fcastdir(g, l_star[sym])
        results[sym] = macro_f1(g_f["nextdir"], g_f["fcastdir"])
    return pd.Series(results, name="macro_f1")
