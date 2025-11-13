# Venue Revenue Forecasting & Staffing Support

This project builds a weekly revenue forecasting model for a set of venues using synthetic data. It follows a simple end-to-end workflow: generate data, explore it, create features, train a model and review the results. The aim is to show how data science can help with planning and highlight which factors have the strongest influence on revenue.

The work mirrors tasks expected in a junior data science role: cleaning data, exploring patterns, creating and testing models and explaining the findings clearly.

## Project structure

```
venue-revenue-forecasting/
├── data/
│   └── raw/
├── notebooks/
│   └── 01_eda.ipynb
├── src/
│   ├── make_dataset.py
│   └── train_basic_model.py
├── models/
├── requirements.txt
└── README.md
```

## What the project does

### Synthetic data generation

`make_dataset.py` creates a weekly dataset for several venues. The data includes revenue, staff hours, weather score, local events, promotions, opening hours and calendar details. The synthetic data behaves in a realistic way, making it suitable for modelling and exploration.

### Exploratory work

`notebooks/01_eda.ipynb` looks at patterns in the data, including revenue over time, differences between venues and the effect of promotions, events and weather. Charts and summaries help build a picture of what drives demand.

### Model training

`train_basic_model.py` adds features such as lagged revenue, moving averages and calendar information. A random forest model is then trained to predict weekly revenue. Results include MAE, RMSE, MAPE and a feature importance table, which shows which inputs matter most.

## How to run the project

### 1. Install dependencies

```
pip install -r requirements.txt
```

### 2. Create the dataset

```
python src/make_dataset.py
```

This saves a CSV file to `data/raw/venue_weekly_data.csv`.

### 3. Train the model

```
python src/train_basic_model.py
```

This prints test metrics and feature importance in the terminal.

### 4. Explore the data

Open `notebooks/01_eda.ipynb` in Jupyter, VS Code or another notebook tool.

## Next steps

Possible extensions include adding more models, improving feature engineering, simulating more venues, creating a small dashboard or comparing forecasts by venue and season.

### 5. Generate future forecasts

After training the model, you can create forecasts for the next few weeks:

```bash
python src/forecast_future.py