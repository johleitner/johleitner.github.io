---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: '0.8'
    jupytext_version: '1.4.1'
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# Demand forecasting: Deep learning vs. time series models

## Highlights

* 2 sales datasets are used for the comparison of state of the art deep learning models and linear time series models.
* Unexpectedly, ARIMA and ETS outperform the deep learning models for long time series and vice versa for short time series.
* The results are generated without hyperparameter tuning of the DL models but they indicate that simple models can be competitive. 

## Datasets

There are 2 datasets used as summarized in the table below: 


```{table} Datasets used in this comparison
:name: my-table-ref
|Dataset number | Dataset name | Type | Reference | Frequency | # Time Series | # Time series periods |
|---|---|---|---|---|---|---|
|1| Stallion (Kaggle) | Artificial |  [Source]( https://www.kaggle.com/utathya/future-volume-prediction)  | Month | 350 | 60 | 
|2| Walmart (M5 competition) | Real |  [Source](https://www.kaggle.com/c/m5-forecasting-accuracy/data) | Day | 1000 |  ~1910 |
```

All 350 time series in the Stallion dataset are used. The data is part of the pytorch forecasting module. 

The second dataset is the M5 competition data that originally contains more than 32k time series on the lowest of twelve hierarchies that were required to be forecasted.  It has to be downloaded from the [Kaggle](https://www.kaggle.com/c/m5-forecasting-accuracy/data) website. Only a random sample of 1,000 time series is used for fhe analysis here in order to limit the computation time.



## Models

There are three univariate time series models and two deep learning models used in the comparison. The Theta model won the M3 time series competition. 

AutoArima and AutoETS are not applied to the M5 competition data (dataset 2) because the runtime was too long. Theta is by far faster. 

```{table} Models used in this comparison
:name: my-table-ref

| Model name | Type | Reference | Covariates | Applied to Dataset 1|Applied to Dataset 2|
|---|---|---|---|---|---|
| Seasonal Naive | Univariate time series |    | No|Yes|Yes|
| Theta | Univariate time series |   {cite:p}`Assimakopoulos2000`  | No|Yes|Yes|
| AutoArima | Univariate time series |  |No|Yes|No|
| AutoETS | Univariate time series |  |No|Yes|No|
| Nbeats | Deep neural network | {cite:p}`oreshkin2020nbeats` |No|Yes|Yes|
| Temporal Fusion Transformer | Deep neural network | {cite:p}`lim2020temporal` | Yes |Yes|Yes|
```

## References

```{bibliography}
```
