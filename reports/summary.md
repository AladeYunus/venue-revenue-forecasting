# Venue revenue forecasting – summary

This project builds a weekly revenue forecasting model for a group of venues using synthetic but realistic data. It shows how forecasting can support planning, highlight patterns in demand and give a simple guide for staffing levels.

## Data

The dataset includes:

* weekly revenue per venue
* staff hours
* a basic weather score
* local event flags
* promotions
* opening hours and calendar details

The data follows patterns you would expect in real settings, such as seasonal changes and occasional demand spikes.

## Exploratory work

The notebook explores:

* revenue trends across time and by venue
* the effect of events and promotions
* variations linked to weather
* simple visual checks to understand each venue’s behaviour

This helps build a clear picture before modelling.

## Model

A random forest model is trained using:

* lagged revenue
* a short moving average
* weather, event and promotion indicators
* opening hours and basic calendar features

On the test period, performance is roughly:

* MAE in the low thousands
* MAPE around seven percent

This shows the model is generally close to the actual weekly values and captures the main drivers of demand.

Feature importance suggests that previous revenue, recent trends and event or promotion activity are the strongest signals.

## Staffing suggestion

Once revenue is predicted, the forecasts are translated into staffing suggestions using a simple rule of thumb based on revenue per staff hour. For each week, the output includes:

* a central suggested value
* a lower and upper range

This makes it easy to compare actual staffing levels with the suggested levels and spot weeks that may look over or under-staffed.

## Future forecasts

The project also generates forecasts for the next twelve weeks. These are produced by rolling the model forward week by week for each venue, using the most recent observations and simple assumptions.

The output includes:

* predicted revenue for each future week
* suggested staffing levels
* the venue and date for every row

These forecasts are shown in the notebook so you can see the expected trend over the coming weeks.

## Overall picture

The project covers:

* data creation
* exploration
* model training and evaluation
* revenue predictions
* staffing suggestions
* future forecasts

It offers a simple, practical example of how machine learning can support planning and provides a base that could be extended with richer data, alternative models or more detailed staffing rules.
