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

# Walmart M5 competition (Kaggle)

The dataset contains 30,490 different time series of length 1941. Price data and data on calendar effects are available. Sales for different products start at different points in time. If the prices are 0 then this means that the product is not sold at this specific day. 

For the analysis not all time series are used but a sample of 1,000 time series. This is due to the fact that the computation time is very large. For this reason only Theta and seasonal naive are used as classic time series models. AutoArima and AutoETS estimated on time series of these lengths are too time consuming. 

For the temporal fusion transformer - the model that allows exogenous variables to be included - features are generated. 

A 10-fold rolling window cross validation is performed. The forecast horizon in each cross validation run is chosen to be 7 periods long. 




## Import of Data

The plot shows the complete history of all actuals of one randomly chosen time series that is used for this evaluation. Most time series of this dataset are erratic or intermittent. 

```{code-cell} ipython3
:tags: [hide-input]

import os
import warnings

warnings.filterwarnings("ignore")  # avoid printing out absolute paths

import copy
from pathlib import Path
import warnings
import gc

import numpy as np
import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
import torch

from pytorch_forecasting import Baseline, TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import SMAPE, PoissonLoss, QuantileLoss
from pytorch_forecasting.models.temporal_fusion_transformer.tuning import optimize_hyperparameters


from pytorch_forecasting.data.examples import get_stallion_data

from statsmodels.tsa.forecasting.theta import ThetaModel
from sktime.forecasting.arima import AutoARIMA
from sktime.forecasting.base import ForecastingHorizon
from sktime.forecasting.ets import AutoETS
from sktime.utils.plotting import plot_series
from sktime.forecasting.naive import NaiveForecaster

from tqdm import tqdm

from pytorch_forecasting import NBeats
from pytorch_forecasting.data import NaNLabelEncoder


kag_dat = pd.read_csv("D:/Kaggle/M5/RawDataJune/sales_train_evaluation.csv")
kag_calendar = pd.read_csv("D:/Kaggle/M5/RawDataJune/calendar.csv")
kag_prices = pd.read_csv("D:/Kaggle/M5/RawDataJune/sell_prices.csv")

kag_dat = kag_dat.sample(1000, random_state=42)

kag_calendar = kag_calendar[['date','d','wm_yr_wk','snap_CA', 'snap_TX','snap_WI','weekday']].rename(columns = {'d':'day'})   
kag_calendar['day'] = kag_calendar['day'].str.replace('d_', "") 
kag_calendar['day'] = kag_calendar['day'].astype(str).astype(int)    

kag_dat_long = pd.wide_to_long(kag_dat, 
                                  stubnames='d_', 
                                  i=['id','item_id','dept_id','cat_id','store_id','state_id'], 
                                  j='day').rename(
                                columns = {'d_':'actual'}).sort_values(['id','day']).reset_index()

kag_dat_long = kag_dat_long.merge(kag_calendar, on = ['day'], how = 'left')
kag_dat_long['date'] = pd.to_datetime(kag_dat_long['date'], errors='coerce')
kag_dat_long['actual'] = kag_dat_long['actual'].astype(float)


#when the price is missing the actuals are always 0. 
#The corresponding rows are removed for local time series models and TFT but not for N-beats
kag_dat_long = kag_dat_long.merge(kag_prices, on = ['store_id','item_id','wm_yr_wk'], how = 'left')

#these must be strings for the temporal fusion transformer
kag_dat_long['snap_CA'] = kag_dat_long['snap_CA'].astype(str)
kag_dat_long['snap_TX'] = kag_dat_long['snap_TX'].astype(str)
kag_dat_long['snap_WI'] = kag_dat_long['snap_WI'].astype(str)

max_prediction_length = 7
max_encoder_length = 35


id = kag_dat_long.id.head(1).squeeze()
item = kag_dat_long.item_id.head(1).squeeze()
store = kag_dat_long.store_id.head(1).squeeze()
state = kag_dat_long.state_id.head(1).squeeze()

timeseries = kag_dat_long.loc[kag_dat_long['id'] == id,['actual','date']]
timeseries['date'] = pd.to_datetime(timeseries['date'])
timeseries = timeseries.set_index('date').to_period('D')

plot_series(timeseries,labels = ['Sales for item ' + str(item) + ' in Store ' + str(store) + ' in state ' + str(state)])

```



## Time series models 

```{code-cell} ipython3
:tags: [hide-input, remove-output]


CV = 10 #number of CV runs, ie rolling time windows
results = []

timeseries_idx = sorted(kag_dat_long.id.unique())

for cv in tqdm(range(CV-1,-1,-1)):
    training_cutoff = kag_dat_long["day"].max() - max_prediction_length - cv

    fh = ForecastingHorizon(
            pd.PeriodIndex(pd.date_range(kag_dat_long.loc[kag_dat_long['day'] > training_cutoff,
                                                          'date'].min(), periods=max_prediction_length, freq="D"))
            , is_relative=False)
    for idx in timeseries_idx:     
        train_time_series = kag_dat_long.loc[(kag_dat_long['sell_price'].notna()) & 
                                             (kag_dat_long['id'] == idx) & 
                                             (kag_dat_long['day'] <= training_cutoff) , 
                                                ['date','actual']].set_index('date').squeeze()
        test_time_series = kag_dat_long.loc[(kag_dat_long['sell_price'].notna()) & 
                                            (kag_dat_long['id'] == idx) & 
                                            (kag_dat_long['day'] > training_cutoff),
                                                ['date','actual']].reset_index(drop = True).head(
                                                                                    max_prediction_length)
        train_time_series.index = pd.PeriodIndex(pd.date_range("2011-01-29", 
                                                               periods=len(train_time_series), freq="D"))

        #Sktime seasonal naive
        df_snaive = test_time_series.copy()
        df_snaive['Model'] = 'Seasonal Naive'
        df_snaive['CV'] = cv
        df_snaive['timeseries_idx'] = idx
        naive = NaiveForecaster(strategy = "last", sp = 7)
        naive.fit(train_time_series)
        df_snaive['Forecast'] = naive.predict(fh = fh).reset_index(drop = True)
        df_snaive['Replaced'] = 'No'
        results.append(df_snaive)
        
        #Statsmodels Theta
        df_theta = test_time_series.copy()
        df_theta['Model'] = 'Theta'
        df_theta['CV'] = cv
        df_theta['timeseries_idx'] = idx
        try:
            tm = ThetaModel(train_time_series, period = 7)
            tm_fitted = tm.fit()
            df_theta['Forecast'] = tm_fitted.forecast(max_prediction_length).reset_index(drop = True)
            df_theta['Replaced'] = 'No'
            results.append(df_theta)
        except ValueError:
            naive = NaiveForecaster(strategy = "mean")
            naive.fit(train_time_series)
            df_theta['Forecast'] = naive.predict(fh = fh).reset_index(drop = True)
            df_theta['Replaced'] = 'Yes'
            results.append(df_theta)            
        
        '''
        #Sktime Auto ARIMA
        df_arima = test_time_series.copy()
        df_arima['Model'] = 'AutoArima'
        df_arima['CV'] = cv
        df_arima['timeseries_idx'] = idx
        try:
            arima = AutoARIMA(sp=12, n_jobs=-1, max_d=2, max_p=2, max_q=2, suppress_warnings=True)
            arima_fitted = arima.fit(train_time_series)
            df_arima['Forecast'] = arima_fitted.predict(fh).reset_index(drop = True)
            df_arima['Replaced'] = 'No'
            results.append(df_arima)
        except ValueError:    
            naive = NaiveForecaster(strategy = "mean")
            naive.fit(train_time_series)
            df_arima['Forecast'] = naive.predict(fh = fh).reset_index(drop = True)
            df_arima['Replaced'] = 'Yes'
            results.append(df_arima)
        
        #Sktime Auto ETS
        df_ets = test_time_series.copy()
        df_ets['Model'] = 'AutoETS'
        df_ets['CV'] = cv
        df_ets['timeseries_idx'] = idx
        try:
            autoets = AutoETS(auto=True, additive_only = True, n_jobs=-1, sp=12)
            autoets_fitted = autoets.fit(train_time_series)
            df_ets['Forecast'] = autoets_fitted.predict(fh).reset_index(drop = True)
            df_ets['Replaced'] = 'No'
            results.append(df_ets)
        except ValueError:    
            naive = NaiveForecaster(strategy = "mean")
            naive.fit(train_time_series)
            df_ets['Forecast'] = naive.predict(fh = fh).reset_index(drop = True)
            df_ets['Replaced'] = 'Yes'
            results.append(df_ets)
        '''    

df_results_local2 = pd.concat(results) 
```




The plots show actuals of an example time series and forecasts by the Theta model, AutoArima and AutoETS. The forecasts  are generated by the 10-fold rolling window cross validation. 


```{code-cell} ipython3
:tags: [hide-input]

def generate_plot(df, raw_data, id, model):
  '''
  The function plots the historical actuals and the forecasts of all 
  cross validation runs. 
  
  Parameters: 
  
  df is a data frame of the form 
  Data columns (total 7 columns):
   #   Column          Non-Null Count  Dtype         
  ---  ------          --------------  -----         
   0   date            280 non-null    datetime64[ns]
   1   actual          280 non-null    float64       
   2   Model           280 non-null    object        
   3   CV              280 non-null    int64         
   4   timeseries_idx  280 non-null    object        
   5   Forecast        280 non-null    float64       
   6   Replaced        280 non-null    object     
  
  id: int that selects one of the time series in the Stallion dataset
  model: str. One of ['Theta', 'Seasonal Naive']
  '''
  forecasts = df.loc[(df['timeseries_idx'] == id) & (df['Model'] == model),['date','Forecast','CV']]
  forecasts['date'] = pd.to_datetime(forecasts['date'], errors='coerce').dt.to_period('D')
  forecasts['Data'] = 'Forecasts made in ' + forecasts.groupby('CV')['date'].transform('min').dt.strftime('%Y-%m-%d')
  forecasts = forecasts.rename(columns = {'Forecast':'value'})
  
  actuals = raw_data.loc[(raw_data['id'] == id),['date','actual']].rename(columns = {'actual':'value'})
  actuals['Data'] = "Actual"
  actuals['date'] =  actuals['date'].dt.to_period('D')
  
  total = pd.concat([forecasts[['date','value', 'Data']].set_index('date'),
              actuals.set_index('date')],axis = 0).reset_index().pivot(index='date',
                                                                       columns = "Data", values = 'value')
  
  total[total.index>'2016-03-01'].plot(figsize = (20,5), grid=True, 
      title = "Actuals and forecasts of " + model + " for all cross validation runs for time series id " + str(id))

generate_plot(df_results_local2, kag_dat_long, id, 'Theta')
```

```{code-cell} ipython3
:tags: [hide-input]
generate_plot(df_results_local2, kag_dat_long, id, 'Seasonal Naive')
```


## Temporal Fusion Transformer


The Temporal Fusion Transformer is trained here with code from the official [tutorial]( https://pytorch-forecasting.readthedocs.io/en/latest/tutorials/stallion.html). The learning rate was optimized but is not shown here. The error metric was set to RMSE, which showed better performance. 

```{code-cell} ipython3
:tags: [hide-input, remove-output]
from pytorch_forecasting.metrics import RMSE



from pytorch_forecasting.metrics import RMSE

CV = 10  #number of CV runs, ie rolling time windows
results_tft_kaggle = []
for cv in tqdm(range(CV-1,-1,-1)):    
    training_cutoff = kag_dat_long["day"].max() - max_prediction_length - cv

    training = TimeSeriesDataSet(
    kag_dat_long[(kag_dat_long['day'] <= training_cutoff) & (kag_dat_long['sell_price'].notna())],
    time_idx="day",
    target="actual",
    group_ids=["id"],
    min_encoder_length=max_encoder_length // 2,  
    max_encoder_length=max_encoder_length,
    min_prediction_length=1,
    max_prediction_length=max_prediction_length,
    static_categoricals=['id','item_id','dept_id','cat_id','store_id','state_id'],
    #static_reals=["avg_population_2017", "avg_yearly_household_income_2017"],
    time_varying_known_categoricals=["weekday", 'snap_CA', 'snap_TX', 'snap_WI'],
    #variable_groups={"special_days": special_days},  # group of categorical variables can be treated as one variable
    time_varying_known_reals=["day", "sell_price"],
    time_varying_unknown_categoricals=[],
    time_varying_unknown_reals=[],
    target_normalizer=GroupNormalizer(
        groups=["id"], transformation="softplus"
    ),  # use softplus and normalize by group
    add_relative_time_idx=True,
    add_target_scales=True,
    add_encoder_length=True,
    )

    # create validation set (predict=True) which means to predict the last max_prediction_length points in time
    # for each series
    validation = TimeSeriesDataSet.from_dataset(training, 
                                                kag_dat_long[(kag_dat_long['day'] <= training_cutoff + 
                                                              max_prediction_length)  & 
                                                             (kag_dat_long['sell_price'].notna())], 
                                                predict=True, stop_randomization=True)

    # create dataloaders for model
    batch_size = 128  
    train_dataloader = training.to_dataloader(train=True, batch_size=batch_size, num_workers=0)
    val_dataloader = validation.to_dataloader(train=False, batch_size=batch_size * 10, num_workers=0)

    # configure network and trainer
    pl.seed_everything(42)
    trainer = pl.Trainer(
        gpus=0,
        gradient_clip_val=0.1,
    )

    # configure network and trainer
    early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=1e-4, patience=10, verbose=False, mode="min")
    lr_logger = LearningRateMonitor()  
    logger = TensorBoardLogger("lightning_logs")  

    #Train model
    trainer = pl.Trainer(
        max_epochs=200,
        gpus=0,
        weights_summary="top",
        gradient_clip_val=0.1,
        limit_train_batches=30,  # coment in for training, running validation every 30 batches
        callbacks=[lr_logger, early_stop_callback],
        logger=logger,
    )

    tft = TemporalFusionTransformer.from_dataset(
    training,
    learning_rate=0.01,
    hidden_size=16,
    attention_head_size=1,
    dropout=0.1,
    hidden_continuous_size=8,
    #output_size=7,  # 7 quantiles by default
    #loss=QuantileLoss(),
    output_size=1,  
    loss=RMSE(),        
    log_interval=10,  
    reduce_on_plateau_patience=4,
    )

    # fit network
    trainer.fit(
        tft,
        train_dataloader=train_dataloader,
        val_dataloaders=val_dataloader,
    )

    # load the best model according to the validation loss
    # (given that we use early stopping, this is not necessarily the last epoch)
    best_model_path = trainer.checkpoint_callback.best_model_path
    best_tft = TemporalFusionTransformer.load_from_checkpoint(best_model_path)

    # calcualte mean absolute error on validation set
    actuals = torch.cat([y[0] for x, y in iter(val_dataloader)])
    predictions = best_tft.predict(val_dataloader)

    column_names =  list(pd.Series(sorted(kag_dat_long.loc[(kag_dat_long['day'] > training_cutoff) & 
                                                           (kag_dat_long['sell_price'].notna()),
                          'date'].unique())).dt.to_period('D').dt.strftime('%Y-%m-%d'))
    df_predictions = pd.DataFrame(predictions.numpy())
    df_predictions.columns = column_names[:max_prediction_length]
    df_predictions.index = kag_dat_long.id.unique()
    df_predictions = df_predictions.stack().reset_index().rename(columns = {'level_0':'id', 'level_1':'date',0:'forecasts'})

    df_actuals = pd.DataFrame(actuals.numpy())
    df_actuals.columns = column_names[:max_prediction_length]
    df_actuals.index = kag_dat_long.id.unique()
    df_actuals = df_actuals.stack().reset_index().rename(columns = {'level_0':'id', 'level_1':'date',0:'actuals'})

    df_result = df_predictions.merge(df_actuals, on = ['id', 'date'], how = 'left')
    df_result['CV'] = cv
    df_result['Model'] = 'Temporal Fusion Transformer'

    results_tft_kaggle.append(df_result)
    
    
df_results_tft_kaggle = pd.concat(results_tft_kaggle)   
```



```{code-cell} ipython3
:tags: [hide-input]
generate_plot(df_results_tft_kaggle.rename(columns = {'id':'timeseries_idx','actuals':'actual',
                                                      'forecasts':'Forecast'}), 
              kag_dat_long, id, 'Temporal Fusion Transformer')
```



## N-beats: neural basis expansion analysis


```{code-cell} ipython3
:tags: [hide-input, remove-output]

CV = 10 #number of CV runs, ie rolling time windows
results_nbeats_kaggle = []
for cv in tqdm(range(CV-1,-1,-1)): 
    training_cutoff = kag_dat_long["day"].max() - max_prediction_length - cv

    context_length = max_encoder_length
    prediction_length = max_prediction_length

    training = TimeSeriesDataSet(
        kag_dat_long[kag_dat_long['day'] <= training_cutoff],
        time_idx="day",
        target="actual",
        categorical_encoders={"id": NaNLabelEncoder().fit(kag_dat_long.id)},
        group_ids=["id"],
        # only unknown variable is "value" - and N-Beats can also not take any additional variables
        time_varying_unknown_reals=["actual"],
        max_encoder_length=context_length,
        max_prediction_length=prediction_length,
    )

    #validation = TimeSeriesDataSet.from_dataset(training, kag_dat_long, min_prediction_idx=training_cutoff + 1)

    validation = TimeSeriesDataSet.from_dataset(training, 
                                                kag_dat_long[(kag_dat_long['day'] <= training_cutoff + 
                                                              max_prediction_length)  & 
                                                             (kag_dat_long['sell_price'].notna())], 
                                                predict=True, stop_randomization=True)    
    
    batch_size = 128
    train_dataloader = training.to_dataloader(train=True, batch_size=batch_size, num_workers=0)
    val_dataloader = validation.to_dataloader(train=False, batch_size=batch_size, num_workers=0)

    early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=1e-4, patience=10, verbose=False, mode="min")
    trainer = pl.Trainer(
        max_epochs=200,
        gpus=0,
        weights_summary="top",
        gradient_clip_val=0.01,
        callbacks=[early_stop_callback],
        limit_train_batches=30,
    )

    net = NBeats.from_dataset(
        training,
        learning_rate=5e-3,
        log_interval=10,
        log_val_interval=1,
        weight_decay=1e-2,
        widths=[32, 512],
        backcast_loss_ratio=1.0,
    )

    trainer.fit(
        net,
        train_dataloader=train_dataloader,
        val_dataloaders=val_dataloader,
    )

    best_model_path = trainer.checkpoint_callback.best_model_path
    best_model = NBeats.load_from_checkpoint(best_model_path)

    actuals = torch.cat([y[0] for x, y in iter(val_dataloader)])
    predictions = best_model.predict(val_dataloader)


    column_names =  list(pd.Series(sorted(kag_dat_long.loc[kag_dat_long['day'] > training_cutoff,
                          'date'].unique())).dt.to_period('D').dt.strftime('%Y-%m-%d'))
    df_predictions = pd.DataFrame(predictions.numpy())
    df_predictions.columns = column_names[:max_prediction_length]
    df_predictions.index = kag_dat_long.id.unique()
    df_predictions = df_predictions.stack().reset_index().rename(columns = {'level_0':'id', 'level_1':'date',0:'forecasts'})

    df_actuals = pd.DataFrame(actuals.numpy())
    df_actuals.columns = column_names[:max_prediction_length]
    df_actuals.index = kag_dat_long.id.unique()
    df_actuals = df_actuals.stack().reset_index().rename(columns = {'level_0':'id', 'level_1':'date',0:'actuals'})

    df_result = df_predictions.merge(df_actuals, on = ['id', 'date'], how = 'left')
    df_result['CV'] = cv
    df_result['Model'] = 'Nbeats'

    results_nbeats_kaggle.append(df_result)
    import gc
    gc.collect()
    
df_results_nbeats_kaggle = pd.concat(results_nbeats_kaggle)     
```


```{code-cell} ipython3
:tags: [hide-input]
generate_plot(df_results_nbeats_kaggle.rename(columns = {'id':'timeseries_idx','actuals':'actual',
                                                      'forecasts':'Forecast'}), 
              kag_dat_long, id, 'Nbeats')
```




## Results 

For the definition of the metrics reported in the table below, see the Results section in the previous chapter. 

Nbeats is the only model that is superior to the naive benchmark in all metrics. Unlike the Temporal Fusion Transformer it does not use covariates and the computation time (not reported here) is much lower.  

Theta has the lowest Mean Absolute Percentage Error and Median Absolute Percentage Error but it has the highest MASE.  

There is no clearly superior model in this case since it depends on the choice of the metric. 


```{code-cell} ipython3
:tags: [hide-input]

df_results_local2 =  df_results_local2.rename(columns= {'Forecast':'forecasts', 
                                  'actual':'actuals', 
                                  'timeseries_idx':'id'}).drop(columns =  'Replaced').copy()
df_results_local2['date'] = df_results_local2['date'].apply(lambda x: x.strftime('%Y-%m-%d')) 

df_total1 = pd.concat([df_results_local2,    
                        df_results_tft_kaggle,
                        df_results_nbeats_kaggle])    


df_total1['AE'] = (df_total1['actuals']-df_total1['forecasts']).abs()
df_total1['PE'] = 100*(df_total1['actuals']-df_total1['forecasts']).abs()/(df_total1['actuals']).abs()


df_mae = df_total1.groupby(['Model'])['AE'].mean().reset_index(name = "Mean absolute error")

df_medape =df_total1[df_total1['actuals']>0].groupby(['Model'])['PE'].median().reset_index(
    name = "Median absolute percentage error")
    
df_mape =  df_total1[df_total1['actuals']>0].groupby(['Model'])['PE'].mean().reset_index(
    name = "Mean absolute percentage error")
    
df_mae0 = df_total1[df_total1['actuals']< 0.001].groupby(['Model'])['AE'].mean().reset_index(
    name = "Mean absolute error when actuals are 0")    


# Calculate MASE 

data_mase = kag_dat_long[['id','actual','date']].sort_values(['id','date']).copy()
data_mase['date'] = data_mase['date'].apply(lambda x: x.strftime('%Y-%m-%d')) 
df_date_cv = df_total1.groupby(['CV'])['date'].min().reset_index().rename(columns = {'date':'min_date'})

df_mase_denominator_list = []
for cv in df_date_cv['CV'].unique():
    tmp = data_mase[data_mase['date'] < df_date_cv.loc[df_date_cv['CV'] == cv,'min_date'].squeeze() ].copy()
    #we choose the forecast horizon < than the seasonal period, therefore a simple shift is ok. 
    tmp['naive_forecast'] = tmp.groupby(['id'])['actual'].shift(7)
    tmp = tmp[tmp['naive_forecast'].notna()]
    tmp['naive_forecast_AE'] = (tmp['naive_forecast']  - tmp['actual']).abs()
    df_mase_denominator = tmp.groupby(['id'])['naive_forecast_AE'].mean().reset_index()
    df_mase_denominator = df_mase_denominator[df_mase_denominator['naive_forecast_AE'].notna()]
    df_mase_denominator['CV'] = cv
    df_mase_denominator_list.append(df_mase_denominator)
    
df_mase_denominator = pd.concat(df_mase_denominator_list)
df_mase_denominator = df_mase_denominator.rename(columns = {'timeseries':'id'})


df_mase = df_total1.merge(df_mase_denominator, on = ['id','CV'], how = 'left')
df_mase['MASE'] = df_mase['AE'] / df_mase['naive_forecast_AE']
df_mase = df_mase.groupby('Model')['MASE'].mean().reset_index()


pd.concat([df_mae, 
            df_medape.drop(columns = ['Model']), 
            df_mape.drop(columns = ['Model']),
            df_mae0.drop(columns = ['Model']),
            df_mase.drop(columns = ['Model'])] , axis = 1).round(2)

```
