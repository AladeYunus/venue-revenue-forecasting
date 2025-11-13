# Venue revenue forecasting – summary

This project uses synthetic data to predict weekly revenue for several venues and suggest staffing levels.

## Data

The dataset includes:

- weekly revenue per venue  
- staff hours  
- a basic weather score  
- local event flags  
- promotions  
- opening hours and calendar information  

The data is generated to behave in a realistic way, with higher demand in summer and around certain weeks.

## Model

A random forest model is trained on lagged revenue, moving averages and contextual features.  
The test performance is around:

- MAE: roughly in the low thousands of pounds  
- MAPE: roughly around 7%  

This means the model is usually within about 7 percent of the actual weekly revenue on the test period.

## Key signals

Feature importance suggests that:

- previous revenue values (lags and moving average) are strong predictors  
- local events and promotions have a clear effect on revenue  
- calendar information also helps capture regular patterns  

## Staffing suggestion

Based on the forecast, a simple rule of thumb converts predicted revenue into suggested staff hours per week. This gives a central estimate and a small range (minimum and maximum) so decision makers can adjust for local knowledge.

The approach is simple on purpose, but it shows how forecasting can support planning and provide a starting point for more detailed labour optimisation.
