from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
MODEL_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"

MODEL_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_data() -> pd.DataFrame:
    path = RAW_DIR / "venue_weekly_data.csv"
    df = pd.read_csv(path, parse_dates=["date"])
    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["venue_id", "date"])

    # time features
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month

    # lag feature: previous week revenue per venue
    df["revenue_lag_1"] = df.groupby("venue_id")["revenue"].shift(1)

    # moving average over last 4 weeks
    df["revenue_ma_4"] = (
        df.groupby("venue_id")["revenue"]
        .rolling(window=4, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
    )

    return df


def add_staffing_suggestion(df_results: pd.DataFrame) -> pd.DataFrame:
    """
    Simple staffing rule of thumb:
    - base: 1 staff hour per £500 of forecast revenue
    - enforce a minimum
    - create a small range (min / max) for flexibility
    """

    target_revenue_per_hour = 500  # £ per staff hour (arbitrary but reasonable)
    min_hours = 20

    suggested_hours = df_results["predicted_revenue"] / target_revenue_per_hour
    suggested_hours = suggested_hours.clip(lower=min_hours)

    df_results["suggested_staff_hours_centre"] = suggested_hours.round(1)
    df_results["suggested_staff_hours_min"] = (suggested_hours * 0.9).round(1)
    df_results["suggested_staff_hours_max"] = (suggested_hours * 1.1).round(1)

    return df_results


def train_and_evaluate(df: pd.DataFrame):
    # drop rows where lag features are missing
    df = df.dropna(subset=["revenue_lag_1", "revenue_ma_4"])

    feature_cols = [
        "year",
        "month",
        "week_of_year",
        "weather_score",
        "local_event",
        "promotion",
        "opening_hours",
        "staff_hours",
        "revenue_lag_1",
        "revenue_ma_4",
    ]
    target_col = "revenue"

    X = df[feature_cols]
    y = df[target_col]

    # simple time-based split: last 20% of rows (by date) as test
    df_sorted = df.sort_values("date")
    split_index = int(len(df_sorted) * 0.8)

    train_idx = df_sorted.index[:split_index]
    test_idx = df_sorted.index[split_index:]

    X_train, X_test = X.loc[train_idx], X.loc[test_idx]
    y_train, y_test = y.loc[train_idx], y.loc[test_idx]

    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=10,
        random_state=42,
        n_jobs=-1,
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred) ** 0.5  # manual square root
    mape = np.mean(np.abs((y_test - y_pred) / (y_test + 1e-6))) * 100

    print(f"Test MAE:  {mae:.2f}")
    print(f"Test RMSE: {rmse:.2f}")
    print(f"Test MAPE: {mape:.2f}%")

    # feature importance
    feature_importance = (
        pd.DataFrame(
            {"feature": feature_cols, "importance": model.feature_importances_}
        )
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )

    print("\nFeature importance:")
    print(feature_importance)

    # build a results frame for the test period
    results = df.loc[test_idx, ["date", "venue_id", "revenue", "staff_hours"]].copy()
    results["predicted_revenue"] = y_pred

    results = add_staffing_suggestion(results)

    return model, results, feature_importance, {"mae": mae, "rmse": rmse, "mape": mape}


def main():
    df = load_data()
    df = add_features(df)

    model, results, feature_importance, metrics = train_and_evaluate(df)

    # save model
    model_path = MODEL_DIR / "rf_venue_revenue.joblib"
    joblib.dump(model, model_path)
    print(f"\nSaved model to: {model_path}")

    # save test predictions with staffing suggestion
    predictions_path = OUTPUT_DIR / "test_predictions_with_staffing.csv"
    results.to_csv(predictions_path, index=False)
    print(f"Saved test predictions to: {predictions_path}")

    # save feature importance
    fi_path = OUTPUT_DIR / "feature_importance.csv"
    feature_importance.to_csv(fi_path, index=False)
    print(f"Saved feature importance to: {fi_path}")

    print("\nMetrics:")
    for k, v in metrics.items():
        print(f"{k.upper()}: {v:.2f}")


if __name__ == "__main__":
    main()
