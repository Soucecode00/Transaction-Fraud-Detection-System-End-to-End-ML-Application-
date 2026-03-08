import pandas as pd
from xgboost import XGBClassifier


def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    params: dict,
    random_state: int = 42
):
    """
    Train an XGBoost fraud detection model.

    Args:
        X_train (pd.DataFrame): Training features
        y_train (pd.Series): Training labels
        params (dict): Model hyperparameters

    Returns:
        trained XGBClassifier model
    """

    # Handle class imbalance
    scale_pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)

    model = XGBClassifier(
        **params,
        scale_pos_weight=scale_pos_weight,
        eval_metric="auc",
        random_state=random_state,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    return model