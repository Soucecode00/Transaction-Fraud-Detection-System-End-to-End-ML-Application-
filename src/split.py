# src/data_split.py

import pandas as pd
from sklearn.model_selection import train_test_split


def split_dev_holdout(
    df: pd.DataFrame,
    label_col: str,
    holdout_size: float = 0.1,
    random_state: int = 42,
    stratify: bool = True
):
    """
    Perform initial split into development data and holdout data.

    This split should happen ONCE and the holdout set must never
    be used during training or hyperparameter tuning.

    Args:
        df (pd.DataFrame): Full feature-engineered dataset
        label_col (str): Target column name
        holdout_size (float): Fraction for holdout dataset
        random_state (int): Reproducibility
        stratify (bool): Preserve class distribution

    Returns:
        df_dev (pd.DataFrame): Development dataset
        df_holdout (pd.DataFrame): Unseen holdout dataset
    """

    y = df[label_col] if stratify else None

    df_dev, df_holdout = train_test_split(
        df,
        test_size=holdout_size,
        stratify=y,
        random_state=random_state
    )

    return df_dev.reset_index(drop=True), df_holdout.reset_index(drop=True)