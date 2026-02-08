import numpy as np
import pandas as pd


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build behavioral fraud features for transactional fraud detection.

    This function performs deterministic feature engineering on raw
    transaction-level data and returns a dataframe containing ONLY
    the selected engineered features used for modeling.

    Design principles:
    - No file I/O (expects a dataframe as input)
    - No plotting or EDA logic
    - No model logic
    - Safe to reuse across training, validation, and inference

    Parameters
    ----------
    df : pd.DataFrame
        Raw transaction dataframe (e.g., PaySim-style schema)

    Returns
    -------
    pd.DataFrame
        Dataframe containing engineered fraud features
        (and `isFraud` if present in input for training use)
    """

    # ------------------------------------------------------------------
    # Defensive copy to avoid mutating the original dataframe
    # ------------------------------------------------------------------
    df = df.copy()

    # ------------------------------------------------------------------
    # Focus on high-risk transaction types
    # In PaySim, almost all fraud occurs in TRANSFER and CASH_OUT
    # ------------------------------------------------------------------
    df['type'] = df['type'].str.upper()
    df = df[df['type'].isin(['TRANSFER', 'CASH_OUT'])].copy()

    # Small constant to avoid division-by-zero errors
    EPS = 1e-6

    # ------------------------------------------------------------------
    # Transaction type flags
    # ------------------------------------------------------------------
    df['is_transfer'] = (df['type'] == 'TRANSFER').astype(int)
    df['is_cash_out'] = (df['type'] == 'CASH_OUT').astype(int)

    # ------------------------------------------------------------------
    # Origin account (victim) behavior features
    # ------------------------------------------------------------------
    # Absolute balance change at origin
    df['orig_balance_delta'] = df['oldbalanceOrg'] - df['newbalanceOrig']

    # Fraction of origin balance drained
    df['orig_drop_ratio'] = df['orig_balance_delta'] / (df['oldbalanceOrg'] + EPS)

    # Post-transaction balance ratio
    df['orig_post_ratio'] = df['newbalanceOrig'] / (df['oldbalanceOrg'] + EPS)

    # Transaction amount relative to origin balance
    df['amount_to_oldbalanceOrg'] = df['amount'] / (df['oldbalanceOrg'] + EPS)

    # Indicator: origin account empty before transaction
    df['orig_zero_before'] = (df['oldbalanceOrg'] == 0).astype(int)

    # ------------------------------------------------------------------
    # Destination account (mule) behavior features
    # ------------------------------------------------------------------
    # Absolute balance gain at destination
    df['dest_balance_delta'] = df['newbalanceDest'] - df['oldbalanceDest']

    # Fractional gain relative to destination balance
    df['dest_gain_ratio'] = df['dest_balance_delta'] / (df['oldbalanceDest'] + EPS)

    # Destination enrichment ratio after transaction
    df['dest_enrichment_ratio'] = df['newbalanceDest'] / (df['oldbalanceDest'] + EPS)

    # Transaction amount relative to destination balance
    df['amount_to_oldbalanceDest'] = df['amount'] / (df['oldbalanceDest'] + EPS)

    # Indicator: destination account empty before transaction
    df['dest_zero_before'] = (df['oldbalanceDest'] == 0).astype(int)

    # ------------------------------------------------------------------
    # Accounting consistency features
    # These capture balance inconsistencies common in PaySim fraud
    # ------------------------------------------------------------------
    df['error_balance_org'] = (df['newbalanceOrig'] + df['amount']) - df['oldbalanceOrg']
    df['error_balance_dest'] = (df['oldbalanceDest'] + df['amount']) - df['newbalanceDest']

    df['abs_error_org'] = df['error_balance_org'].abs()
    df['abs_error_dest'] = df['error_balance_dest'].abs()

    # ------------------------------------------------------------------
    # Time-based features
    # ------------------------------------------------------------------
    df['hour'] = df['step'] % 24
    df['day'] = df['step'] // 24

    # ------------------------------------------------------------------
    # Scale-stabilized amount feature
    # ------------------------------------------------------------------
    df['log_amount'] = np.log1p(df['amount'])

    # ------------------------------------------------------------------
    # FINAL FEATURE SET
    # Selected based on fraud vs non-fraud median separation
    # ------------------------------------------------------------------
    final_features = [
        'abs_error_dest',
        'amount_to_oldbalanceOrg',
        'orig_drop_ratio',
        'orig_balance_delta',
        'log_amount'
    ]

    # If label exists (training/evaluation), keep it
    if 'isFraud' in df.columns:
        return df[final_features + ['isFraud']]

    # Otherwise (inference), return only features
    return df[final_features]
