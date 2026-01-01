"""
Data loading utilities.
"""
from __future__ import annotations

import pandas as pd
from ..config import DATASET_PATH


def load_data(filepath: str = None) -> pd.DataFrame:
    """
    Load the COVID-19 dataset from Excel file.

    Parameters
    ----------
    filepath : str, optional
        Path to the dataset. If None, uses default from config.

    Returns
    -------
    pd.DataFrame
        Loaded dataset
    """
    if filepath is None:
        filepath = DATASET_PATH

    data = pd.read_excel(filepath)
    df = data.copy()

    # Set display options for better visualization
    pd.set_option('display.max_row', 111)
    pd.set_option('display.max_column', 111)

    return df
