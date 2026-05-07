import pandas as pd
import numpy as np
from typing import Tuple, Union


def generate_results_table(
        results: Union[np.ndarray, list],
        method_names: Union[None, list] = None,
        precision: int = 3,
        print_table: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generate summary and per-subject error tables from evaluation results.

    Parameters
    ----------
    results : np.ndarray or list
        - If np.ndarray: shape (subjects, methods, vertices) → output of pipeline
        - If list or 1D np.ndarray: list of errors (e.g., from one mesh)
    method_names : list of str, optional
        Names of the methods (required if results is 3D)
    precision : int
        Number of decimal digits to round to (in mm)
    print_table : bool
        Whether to print the tables.

    Returns
    -------
    summary_df : pd.DataFrame
        Table with mean, std, median, min, max per method (in mm)
    subject_df : pd.DataFrame
        Table with per-subject mean error (in mm)
    """
    if isinstance(results, np.ndarray) and results.ndim == 3:
        if method_names is None:
            raise ValueError("If 'results' is a 3D array, 'method_names' must be provided.")

        # Mean over vertices → shape: (subjects, methods)
        mean_errors = results.mean(axis=2) * 1000
        subject_df = pd.DataFrame(mean_errors, columns=method_names)
        subject_df.index.name = "subject_id"

        summary_df = subject_df.agg(['mean', 'std', 'median', 'min', 'max']).round(precision).T
        summary_df.index.name = "method"

        if print_table:
            print("\n📋 Per-subject Mean Errors Table:")
            print(subject_df.round(precision))
            print("\n📊 Summary Statistics Table:")
            print(summary_df)

        return summary_df, subject_df

    else:
        # Assume it's a single list of errors
        errors_mm = np.array(results) * 1000
        summary_df = pd.DataFrame([{
            "mean": round(errors_mm.mean(), precision),
            "std": round(errors_mm.std(), precision),
            "median": round(np.median(errors_mm), precision),
            "min": round(errors_mm.min(), precision),
            "max": round(errors_mm.max(), precision),
        }], index=["method"])

        if print_table:
            print("\n📊 Summary Statistics Table:")
            print(summary_df)

        return summary_df, pd.DataFrame()
