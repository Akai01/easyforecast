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




Time series forecastin in Python


* Free software: MIT license
* Documentation: https://easyforecast.readthedocs.io.

* The usage:

Example of ARAR and ARIMA.auto_arima::

    from easyforecast.arar import ARAR
    import pandas as pd
    
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
    arar.get_forecast()
    arar.accuracy(test_set = test)
    
    test_series.plot() 
    arar.plot()
    
    # forecast using auto_arima
    
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

Credits
-------

This package was created with Cookiecutter_.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
