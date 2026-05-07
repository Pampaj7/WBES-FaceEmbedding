import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
from typing import List, Tuple


def compute_correlation(refs: List[float], ests: List[float]) -> Tuple[float, float]:
    """
    Compute Pearson correlation on all elements and on top-5 entries.

    Parameters
    ----------
    refs : list[float]
        Reference (ground-truth) errors.
    ests : list[float]
        Estimated errors.

    Returns
    -------
    tuple[float, float]
        (Pearson correlation full, Pearson correlation on top-5)
    """
    if len(refs) < 5:
        raise ValueError("Need at least 5 samples to compute top-5 correlation")

    corr_all = scipy.stats.pearsonr(refs, ests)[0]
    corr_top5 = scipy.stats.pearsonr(refs[:5], ests[:5])[0]
    return corr_all, corr_top5


def plot_correlation(
        refs: np.ndarray,
        ests: np.ndarray,
        method_name: str,
        save_path: str = None,
        divide_error_by: float = 1.0,
        dpi: int = 100,
        mode: str = "scatter",  # "scatter" or "hexbin"
        alpha: float = 0.05,
        s: int = 5,
        xlim: tuple = None,
        ylim: tuple = None,
):
    """
    Plot correlation between reference and estimated errors.

    Parameters
    ----------
    refs : array-like
        Reference (ground-truth) errors.
    ests : array-like
        Estimated errors.
    method_name : str
        Label for the plot (used in title).
    save_path : str, optional
        If provided, the plot will be saved to this path.
    divide_error_by : float
        Factor to divide errors by (e.g. 1e-3 to convert to mm).
    dpi : int
        Resolution of the figure.
    mode : str
        "scatter" or "hexbin" visualization.
    alpha : float
        Transparency of the scatter points.
    s : int
        Marker size for scatter.
    xlim, ylim : tuple, optional
        Limits for axes.
    """
    refs = np.array(refs) / divide_error_by
    ests = np.array(ests) / divide_error_by

    corr, corr_top5 = compute_correlation(refs, ests)

    plt.figure(figsize=(5, 7), dpi=dpi)

    if mode == "hexbin":
        plt.hexbin(refs, ests, gridsize=100, cmap="viridis", bins="log")
        plt.colorbar(label="log(count)")
    else:
        plt.scatter(refs, ests, alpha=alpha, s=s)

    # Axis labels and limits
    plt.xlabel("Reference error")
    plt.ylabel("Estimated error")
    plt.title(f"Method: {method_name}")

    if xlim:
        plt.xlim(*xlim)
    if ylim:
        plt.ylim(*ylim)

    # Correlation text box
    mnx, mny = np.min(refs), np.min(ests)
    dx = np.max(refs) - mnx
    dy = np.max(ests) - mny

    plt.text(mnx + dx * 0.01, mny + dy * 0.95, f"$\\rho_{{Top5}}$ = {corr_top5:.2f}",
             bbox=dict(boxstyle="round", fc="white", ec="white", pad=0.1), fontsize=10)

    plt.text(mnx + dx * 0.01, mny + dy * 0.85, f"$\\rho$ = {corr:.2f}",
             bbox=dict(boxstyle="round", fc="white", ec="white", pad=0.1), fontsize=10)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
