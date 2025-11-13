from pathlib import Path

import numpy as np
import pandas as pd
import joblib


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
MODEL_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_historical_data() -> pd.DataFrame:
    path = RAW_DIR / "venue_weekly_data.csv"
    df = pd.read_csv(path, parse_dates=["date"])
    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["venue_id", "date"])

    # time features
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["week_of_year"] = df["date"].dt.isocalendar().week.astype(int)

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


def roll_forward_forecast(
    df_features: pd.DataFrame, model, horizon_weeks: int = 12
) -> pd.DataFrame:
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

    future_rows = []

    for venue_id, hist in df_features.groupby("venue_id"):
        hist = hist.sort_values("date")

        last_row = hist.iloc[-1]
        last_date = last_row["date"]
        last_revenue = last_row["revenue"]

        # last 4 actual revenues for moving average
        last_four = hist["revenue"].tail(4).tolist()

        # assume some simple, stable behaviour going forward
        base_opening_hours = last_row["opening_hours"]
        base_staff_hours = last_row["staff_hours"]

        for step in range(1, horizon_weeks + 1):
            new_date = last_date + pd.Timedelta(weeks=1)

            year = new_date.year
            month = new_date.month
            week_of_year = int(new_date.isocalendar().week)

            # simple placeholder assumptions for context features
            weather_score = 0.5      # neutral
            local_event = 0          # no event by default
            promotion = 0            # no promotion by default
            opening_hours = base_opening_hours
            staff_hours = base_staff_hours

            revenue_lag_1 = last_revenue
            revenue_ma_4 = float(np.mean(last_four))

            X_row = pd.DataFrame(
                [[
                    year,
                    month,
                    week_of_year,
                    weather_score,
                    local_event,
                    promotion,
                    opening_hours,
                    staff_hours,
                    revenue_lag_1,
                    revenue_ma_4,
                ]],
                columns=feature_cols,
            )

            predicted_revenue = float(model.predict(X_row)[0])

            future_rows.append(
                {
                    "date": new_date,
                    "venue_id": venue_id,
                    "predicted_revenue": round(predicted_revenue, 2),
                    "weather_score": weather_score,
                    "local_event": local_event,
                    "promotion": promotion,
                    "opening_hours": opening_hours,
                    "staff_hours": staff_hours,
                }
            )

            # update state for next iteration
            last_date = new_date
            last_revenue = predicted_revenue
            last_four = (last_four + [predicted_revenue])[-4:]

    future_df = pd.DataFrame(future_rows)
    future_df = add_staffing_suggestion(future_df)
    return future_df


def main():
    # load trained model
    model_path = MODEL_DIR / "rf_venue_revenue.joblib"
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found at {model_path}. "
            "Run src/train_basic_model.py first."
        )
    model = joblib.load(model_path)

    # load and prepare historical data
    df_hist = load_historical_data()
    df_features = add_features(df_hist)

    # generate future forecasts (e.g. next 12 weeks)
    future_df = roll_forward_forecast(df_features, model, horizon_weeks=12)

    # save to processed data folder
    output_path = OUTPUT_DIR / "future_forecasts_with_staffing.csv"
    future_df.to_csv(output_path, index=False)
    print(f"Saved future forecasts to: {output_path}")


if __name__ == "__main__":
    main()
