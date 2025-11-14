# Venue revenue forecasting & staffing support

This project builds a weekly revenue forecasting model for a group of venues using synthetic but realistic data. It covers data creation, exploration, model training, evaluation, future forecasting and a simple staffing guide based on the predictions.

An insights notebook and a Tableau Public dashboard are included to help present the findings in a clear, practical way.

## Project structure

```
venue-revenue-forecasting/
├── data/
│   ├── raw/
│   └── processed/
├── models/
├── notebooks/
│   ├── 01_eda.ipynb
│   └── 02_insights.ipynb
├── reports/
│   └── summary.md
├── src/
│   ├── make_dataset.py
│   ├── train_basic_model.py
│   └── forecast_future.py
├── requirements.txt
└── README.md
```

## What the project does

### Synthetic data

`make_dataset.py` creates weekly data for several venues, including revenue, staff hours, weather, events, promotions and calendar fields.

### Exploration

`01_eda.ipynb` explores the structure of the data and looks at simple trends across venues and time.

### Model training

`train_basic_model.py` prepares features such as lagged revenue and short moving averages, then trains a random forest model to predict weekly revenue.
It prints performance metrics and saves the trained model, test predictions and feature importance.

### Insights

`02_insights.ipynb` focuses on simple, clear findings such as:

* how promotions and events relate to revenue
* which venues show more variation
* how model accuracy changes across venues
* where staffing differs from the suggested levels
* future predicted revenue and suggested staff hours

### Future forecasts

`forecast_future.py` rolls the model forward to create a 12-week forecast for every venue, along with a suggested staffing range based on predicted revenue.

### Tableau dashboard

A Tableau Public dashboard can be created using the three CSV files:

* `venue_weekly_data.csv`
* `test_predictions_with_staffing.csv`
* `future_forecasts_with_staffing.csv`

The dashboard includes:

1. Revenue trends
2. Actual vs predicted revenue
3. Staff comparisons
4. Future forecasts

The dashboard can be shared publicly through Tableau Public.

## How to run the project

### 1. Create and activate a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Generate the dataset

```bash
python src/make_dataset.py
```

### 4. Train the model

```bash
python src/train_basic_model.py
```

This saves:

* `models/rf_venue_revenue.joblib`
* `data/processed/test_predictions_with_staffing.csv`
* `data/processed/feature_importance.csv`

### 5. Produce future forecasts

```bash
python src/forecast_future.py
```

This saves:

* `data/processed/future_forecasts_with_staffing.csv`

### 6. Explore results in notebooks

Open:

* `notebooks/01_eda.ipynb`
* `notebooks/02_insights.ipynb`

## Next steps

Possible improvements include adding further models, estimating uncertainty, creating more detailed staffing rules or connecting the outputs to a small web app.
