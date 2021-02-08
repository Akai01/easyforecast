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

The easyforecast is still under developement, however ARAR and ARIMA models are working.
ARIMA.auto_arima is an implementation of R`s forecast package auto.arima().

============================
Install directly from Github:
============================

    pip install git+https://github.com/Akai01/easyforecast.git

============================
ARAR example::
============================

    import pandas as pd//
    from easyforecast.arar import ARAR

    df = pd.read_csv("https://raw.githubusercontent.com/Akai01/example-time-series-datasets/main/Data/retail.csv", sep= ",")

    df_sub = df[['date', 'series_38']]

    df_sub.columns = ["ds", "y"]

    df_sub.head()

    train = df_sub[:-12]

    test = df_sub[-12:]

    test_series = pd.DataFrame(pd.array(test['y']), index= pd.to_datetime( test["ds"]))

    test_series.columns = ["test"]

    train.head()

    model = ARAR(train, h = 12, freq = "MS")

    model = model.forecast()

    model.get_forecast()
    
    test_series.plot()

    model.plot()

    model.accuracy(test_set = test)


About:
----------
* Free software: MIT license
* Documentation: https://easyforecast.readthedocs.io.


Features
--------

* TODO

Credits
-------

This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage
