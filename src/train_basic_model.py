from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
MODEL_DIR = PROJECT_ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)

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

    # simple time-based split: last 20% weeks as test
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
    rmse = mean_squared_error(y_test, y_pred) ** 0.5
    mape = np.mean(np.abs((y_test - y_pred) / (y_test + 1e-6))) * 100

    print(f"Test MAE:  {mae:.2f}")
    print(f"Test RMSE: {rmse:.2f}")
    print(f"Test MAPE: {mape:.2f}%")

    # show feature importance
    importance = (
        pd.DataFrame(
            {"feature": feature_cols, "importance": model.feature_importances_}
        )
        .sort_values("importance", ascending=False)
    )
    print("\nFeature importance:")
    print(importance)

def main():
    df = load_data()
    df = add_features(df)
    train_and_evaluate(df)

if __name__ == "__main__":
    main()

