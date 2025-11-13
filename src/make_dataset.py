import numpy as np
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

def generate_synthetic_data(
    n_venues: int = 5,
    start_date: str = "2023-01-01",
    end_date: str = "2024-12-31",
    seed: int = 42,
) -> pd.DataFrame:
    np.random.seed(seed)

    dates = pd.date_range(start=start_date, end=end_date, freq="W")  # weekly
    venue_ids = [f"V{idx+1}" for idx in range(n_venues)]

    rows = []
    for venue in venue_ids:
        base_revenue = np.random.randint(15000, 30000)

        for date in dates:
            week_of_year = date.isocalendar().week

            # simple seasonality: higher in summer, lower in winter
            if date.month in [6, 7, 8]:
                seasonal_factor = 1.2
            elif date.month in [11, 12, 1]:
                seasonal_factor = 0.85
            else:
                seasonal_factor = 1.0

            # weather score (0 = poor, 1 = good)
            weather_score = np.random.choice([0, 0.5, 1.0], p=[0.3, 0.4, 0.3])

            # promotions and local events
            promotion = np.random.binomial(1, 0.2)
            local_event = np.random.binomial(1, 0.15)

            # opening hours per week
            opening_hours = np.random.randint(60, 90)

            # revenue influenced by these factors
            revenue = base_revenue * seasonal_factor
            revenue *= 1 + 0.15 * weather_score
            revenue *= 1 + 0.20 * promotion
            revenue *= 1 + 0.25 * local_event

            # some noise
            revenue *= np.random.normal(1.0, 0.08)

            # staff hours roughly linked to revenue
            staff_hours = (revenue / 500) + np.random.normal(0, 5)
            staff_hours = max(20, staff_hours)  # sensible lower bound

            rows.append(
                {
                    "date": date,
                    "venue_id": venue,
                    "revenue": round(revenue, 2),
                    "staff_hours": round(staff_hours, 1),
                    "weather_score": weather_score,
                    "local_event": local_event,
                    "promotion": promotion,
                    "opening_hours": opening_hours,
                    "week_of_year": week_of_year,
                }
            )

    df = pd.DataFrame(rows)
    return df

def main():
    df = generate_synthetic_data()
    output_path = RAW_DIR / "venue_weekly_data.csv"
    df.to_csv(output_path, index=False)
    print(f"Saved synthetic data to: {output_path}")
    print(df.head())

if __name__ == "__main__":
    main()
