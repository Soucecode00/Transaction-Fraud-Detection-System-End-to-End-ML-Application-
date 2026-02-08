import numpy as np
import pandas as pd

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
    precision_recall_fscore_support
)


def evaluate_model(
    model,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    threshold_grid: np.ndarray | None = None
):
    """
    Evaluate model on validation data and select optimal threshold
    (max F1-score by default).

    Args:
        model: trained classifier with predict_proba
        X_val (pd.DataFrame): validation features
        y_val (pd.Series): validation labels
        threshold_grid (np.ndarray): thresholds to test

    Returns:
        dict with metrics + best threshold
    """

    if threshold_grid is None:
        threshold_grid = np.linspace(0.01, 0.99, 50)

    # Probabilities
    y_proba = model.predict_proba(X_val)[:, 1]

    results = {
        "roc_auc": roc_auc_score(y_val, y_proba),
        "pr_auc": average_precision_score(y_val, y_proba),
    }

    best_f1 = -1
    best_threshold = None
    best_metrics = None

    for t in threshold_grid:
        y_pred = (y_proba >= t).astype(int)

        precision, recall, f1, _ = precision_recall_fscore_support(
            y_val, y_pred,
            average="binary",
            zero_division=0
        )

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = t
            best_metrics = {
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "confusion_matrix": confusion_matrix(y_val, y_pred),
                "classification_report": classification_report(
                    y_val, y_pred, digits=4
                )
            }

    results["best_threshold"] = best_threshold
    results.update(best_metrics)

    return results