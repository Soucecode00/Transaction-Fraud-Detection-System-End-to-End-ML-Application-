import joblib
from sklearn.model_selection import train_test_split

from src.data_loader import load_data
from src.split import split_dev_holdout
from features.Feature_Engineering import build_features
from Models.train import train_model
from Models.evaluate import evaluate_model
# from config import MODEL_PATH


def run():
    print("[run_experiments] run() starting")
    # 1. Load raw data
    df = load_data()
    print(f"[run_experiments] loaded data: {None if df is None else getattr(df, 'shape', None)}")

    df_train, df_holdout_test = split_dev_holdout(
    df,
    label_col="isFraud",
    holdout_size=0.2,
    random_state=42,
    stratify=True)                               

    # 2. Feature engineering
    df_features = build_features(df_train)

    X = df_features.drop(columns=["isFraud"])
    y = df_features["isFraud"]

    # 3. Train / validation split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        test_size=0.2,
        stratify=y,
        random_state=42
    )

    # 4. Train model
    params = {
        "n_estimators": 500,
        "max_depth": 4,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8
    }

    model = train_model(X_train, y_train, params)

    # 5. Evaluate
    eval_results = evaluate_model(model, X_val, y_val)

    # # 6. Save FINAL model
    # joblib.dump(
    #     {
    #         "model": model,
    #         "threshold": eval_results["best_threshold"]
    #     },
    #     MODEL_PATH
    # )

    print("Training complete. Model saved.")


if __name__ == "__main__":
    run()