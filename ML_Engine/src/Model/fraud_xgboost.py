import numpy as np
import pandas as pd
import joblib

from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
    precision_recall_fscore_support
)

def train_model(
    df_features: pd.DataFrame,
    label_col: str = "isFraud",
    test_size: float = 0.2,
    random_state: int = 42
):
    """
    Train an XGBoost fraud detection model.

    - Expects engineered features + target column
    - Uses stratified train/validation split
    - Handles class imbalance via scale_pos_weight

    Returns:
    - trained model
    - validation dataframe (X_val, y_val) for evaluation
    """

    # Separate features and label
    X = df_features.drop(columns=[label_col])
    y = df_features[label_col]

    # Stratified split to preserve fraud ratio
    X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        test_size=test_size,
        stratify=y,
        random_state=random_state
    )

    # Handle class imbalance
    scale_pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)

    # Initialize XGBoost model
    model = XGBClassifier(
        n_estimators=500,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="auc",
        scale_pos_weight=scale_pos_weight,
        random_state=random_state,
        n_jobs=-1
    )

    # Train model
    model.fit(X_train, y_train)

    return model, X_val, y_val


# ------------------------------------------------------------------
# 2. Evaluate model and choose threshold for HIGH RECALL
# ------------------------------------------------------------------
def evaluate_model(
    model,
    X_val: pd.DataFrame,
    y_val: pd.Series
):
    """
    Evaluate model on validation data and select a threshold
    that balances precision and recall (max F1-score).

    Returns:
    - chosen threshold
    """

    # Predict fraud probabilities
    y_proba = model.predict_proba(X_val)[:, 1]

    # Threshold-independent metrics
    print("Validation ROC-AUC:", roc_auc_score(y_val, y_proba))
    print("Validation PR-AUC :", average_precision_score(y_val, y_proba))

    thresholds = np.linspace(0.01, 0.99, 50)

    best_f1 = -1
    best_threshold = None
    best_metrics = None

    # Evaluate all thresholds
    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)

        precision, recall, f1, _ = precision_recall_fscore_support(
            y_val, y_pred, average="binary", zero_division=0
        )

        # Select threshold with maximum F1
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = t
            best_metrics = (precision, recall, f1)

    precision, recall, f1 = best_metrics

    # Final evaluation at chosen threshold
    y_pred = (y_proba >= best_threshold).astype(int)

    print("\nChosen Threshold (max F1):", round(best_threshold, 4))
    print(f"Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f}")

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_val, y_pred))

    print("\nClassification Report:")
    print(classification_report(y_val, y_pred, digits=4))

    return best_threshold


# ------------------------------------------------------------------
# 3. Save / Load model
# ------------------------------------------------------------------
def save_model(model, threshold: float, path: str):
    """
    Save trained model and chosen threshold together.
    """
    joblib.dump(
        {"model": model, "threshold": threshold},
        path
    )


def load_model(path: str):
    """
    Load trained model and threshold.
    """
    obj = joblib.load(path)
    return obj["model"], obj["threshold"]


# ------------------------------------------------------------------
# 4. Inference: predict fraud probability
# ------------------------------------------------------------------
def predict_proba(model, X: pd.DataFrame):
    """
    Predict fraud probability for engineered features.
    """
    return model.predict_proba(X)[:, 1]
