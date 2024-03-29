=============
Easy Forecast
=============


.. image:: https://img.shields.io/pypi/v/easyforecast.svg
        :target: https://pypi.python.org/pypi/easyforecast

.. image:: https://img.shields.io/travis/akai01/easyforecast.svg
        :target: https://travis-ci.com/akai01/easyforecast

.. image:: https://readthedocs.org/projects/easyforecast/badge/?version=latest
        :target: https://easyforecast.readthedocs.io/en/latest/?badge=latest
        :alt: Documentation Status




Forecasting methods and algorithms from:
 Introduction to Time Series and Forecasting by Peter J. BrockwellRichard A. Davis.

The state-space Arima models and other methods will be implemented in the future.

For now, only ARAR algorithm is implemented from Peter J. BrockwellRichard A. Davis[2016]

* Free software: MIT license
* Install from GitHub "pip install git+https://github.com/Akai01/easyforecast"
* Documentation: https://easyforecast.readthedocs.io/en/latest/
* The usage:

Forecasting using ARAR algorithm::

    from easyforecast.arar import ARAR
    import pandas as pd
    from prophet import Prophet
    from easyforecast.utils import accuracy
    import numpy as np
    
    
    df = pd.read_csv("https://raw.githubusercontent.com/Akai01/example-time-series-datasets/main/Data/retail.csv", sep= ",")
    df.head()
    
    df_sub = df[['date', 'series_38']] 
    df_sub.columns = ["ds", "y"] 
    df_sub.head()
    
    train = df_sub[:-12]
    test = df_sub[-12:]
    test_series = pd.DataFrame(pd.array(test['y']), index= pd.to_datetime( test["ds"]))
    test_series.columns = ["test"]
    train.head()
    
    # forecast using ARAR model
    arar = ARAR(train, h = 12, freq = "MS")
    arar.forecast() 
    arar_forecast = arar.get_forecast()
    arar.accuracy(test_set = test)
    
    test_series.plot()
    arar.plot()

Prophet forecast vs ARAR::

    m = Prophet()
    m.fit(train)
    future = m.make_future_dataframe(periods=12, freq = "MS", include_history= False)
    prophet_forecast = m.predict(future)
    prophet_forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
    
    accuracy(np.array(test["y"]), np.array(prophet_forecast["yhat"]))

    accuracy(np.array(test["y"]), np.array(arar_forecast["mean"]))

MAPE ARAR = 1.9 vs MAPE Prophet = 3.4
--------    
    
Forecast using auto_arima::

    
    from easyforecast.arima import ARIMA
    arima = ARIMA(train, freq = "MS") 
    arima.auto_arima(m = 12, ic = "aicc")
    arima.forecast(h = 12)
    arima.get_forecast()
    arima.accuracy(test_set = test)
    test_series.plot() 
    arima.plot()


Features
--------

* Automatic model selection
* Visualisation of the forecast models
* Risidual diagnostics
* Error diagnostics

