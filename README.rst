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
    model1 = ARAR(train, h = 12, freq = "MS")
    model1 = model1.forecast()
    model1.get_forecast()
    model1.accuracy(test_set = test)
    
    test_series.plot() 
    model1.plot()
    
    # forecast using auto_arima
    
    from easyforecast.arima import ARIMA
    model2 = ARIMA(train, freq = "MS")
    model2 = model2.auto_arima(m = 12, ic = "aicc")
    
    model2 = model2.forecast(h = 12)
    model2.get_forecast()
    model2.accuracy(test_set = test)


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
