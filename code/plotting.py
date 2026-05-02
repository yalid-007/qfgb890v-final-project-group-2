import os
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd
import seaborn as sns


def plot_qimbal_dist(
    df: pd.DataFrame,
    symbols: list[str],
    figsize: tuple = (11, 8),
    save_path: str | None = None,
) -> plt.Figure:
    """Plot qimbal histogram with KDE for each symbol in a 2x2 grid.

    Args:
        df: DataFrame with columns 'sym' and 'qimbal'.
        symbols: Exactly four symbol strings to plot.
        figsize: Figure size (width, height) in inches.
        save_path: If provided, save the figure to this path.

    Returns:
        The matplotlib Figure object.
    """
    if len(symbols) != 4:
        raise ValueError(f"Expected 4 symbols, got {len(symbols)}")

    fig, axes = plt.subplots(2, 2, figsize=figsize)
    axes = axes.flatten()

    for ax, sym in zip(axes, symbols):
        data = df.loc[df["sym"] == sym, "qimbal"].dropna()
        sns.histplot(
            data,
            kde=True,
            bins=60,
            stat="density",
            ax=ax,
            color="steelblue",
            alpha=0.6,
            line_kws={"linewidth": 1.8},
        )
        ax.axvline(0, color="black", linestyle="--", linewidth=1, label="zero")
        ax.axvline(data.mean(), color="crimson", linestyle="-", linewidth=1.2,
                   label=f"mean={data.mean():.3f}")
        ax.set_title(sym, fontsize=12, fontweight="bold")
        ax.set_xlabel("Quote Imbalance")
        ax.set_ylabel("Density")
        ax.xaxis.set_major_locator(mticker.MultipleLocator(0.5))
        ax.legend(fontsize=8)

    fig.suptitle("Distribution of Quote Imbalance by Symbol", fontsize=13, y=1.01)
    fig.tight_layout()

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True) if os.path.dirname(save_path) else None
        fig.savefig(save_path, bbox_inches="tight", dpi=150)

    return fig
