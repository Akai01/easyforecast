#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  3 09:52:01 2021

@author: Resul Akay
"""
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA as stat_arima
import matplotlib.pyplot as plt
from easyforecast.utils import ndiffs
from easyforecast.utils import nsdiffs
from easyforecast.utils import accuracy as accuracy_util

class ARIMA:
    """
    Time Series forecasting using ARIMA model.
    
    It is a wrapper for the ARIMA class in statsmodels.tsa.arima.model module.
    
    easyforecast package ARIMA automates:
        * model selection beased of information cretaria,
        * forecasting,
        * residual diagnostics,
        * plotting
        * accuracy masurement.
    
    --------------------------
    Methods
    --------------------------
    auto_arima()
        Fit best ARIMA model to univariate time series.
    summary()
        ARIMA object summaries
    forecast()
        Forecast an ARIMA object of easyforecast.arima module
    residuals()
        Extract model residuals
    get_forecast()
        Extract forecast as a data frame
    accuracy()
        Accuracy measures
    plot()
        Plot n ARIMA object of easyforecast.arima module
    """
    def __init__(self, df, freq, xreg = None):
        """
        Initiates easyforecast ARIMA class to fit an automatic arima model 
        to a univariate time series.
        
        The model is initiated by providing the following arguments.
        
        Args:
            df: The observed time-series as a pandas data frame which has two 
            columns; ds the date column and y the univariate time series.
            
            freq : str, the frequency of the time-series which specifies  
            Pandas offset or offset string.
            
            xreg: Optional, Array of external regressors, 
            which must have the same number of rows as df.
            
        """
        self.y = np.array(df['y'])
        self.ds = pd.to_datetime(df['ds'])
        self.xreg = xreg
        self.freq = freq
        self.df = df
    
    def auto_arima(self, d = None, D = None, m = 1, max_p = 5, max_q = 5,
                   max_P = 2, max_Q = 2, max_d = 2, max_D = 1, 
                   start_p = 0, start_q = 0, start_P = 0, start_Q = 0,
                   alpha = 0.05, ic = "aic", trend=None, 
                   enforce_stationarity= False, 
                   enforce_invertibility=True, 
                   concentrate_scale=False, 
                   trend_offset=1, dates=None, 
                   missing='none', validate_specification=True,
                   start_params=None, transformed=True, includes_fixed=False, 
                   method=None, method_kwargs=None, gls=None, gls_kwargs=None, 
                   cov_type=None, cov_kwds=None, return_params=False, 
                   low_memory=False):
        """
        Fits best ARIMA model to univariate time series.
        
        It is a wrapper for the ARIMA class in statsmodels.tsa.arima.model modul
        
        Args: 
            d : Order of first-differencing. If missing, will choose a value 
            based on Kwiatkowski et al. Unit Root Test.
            
            D: Order of seasonal-differencing. If missing, will choose a value 
            based on Kwiatkowski et al. Unit Root Test. 
            
            m : Length of seasonal period, e.g. for monthly data 12 for weekly 
            data 52.
            
            max_p : Maximum value of p
            
            max_q : Maximum value of q
            
            max_P : Maximum value of P
            
            max_Q : Maximum value of Q
            
            max_d : Maximum value of d
            
            max_D : Maximum value of D
            
            start_p : Starting value of p. 
            
            start_q : Starting value of q. 
            
            start_P : Starting value of P. 
            
            start_Q : Starting value of Q. 
            
            alpha : Level of the Kwiatkowski et al. Unit Root test, possible 
            values range from 0.01 to 0.1.
            
            ic : Information criterion to be used in model selection.
            
            trend : str{'n','c','t','ct'} or iterable, optional Parameter 
            controlling the deterministic trend. Can be specified as a string 
            where 'c' indicates a constant term, 't' indicates a linear trend 
            in time, and 'ct' includes both. Can also be specified as an 
            iterable defining a polynomial, as in `numpy.poly1d`, where 
            `[1,1,0,1]` would denote :math:`a + bt + ct^3`. Default is 'c' for 
            models without integration, and no trend for models with 
            integration.
            
            enforce_stationarity : bool, optional 
            Whether or not to require the autoregressive parameters to 
            correspond to a stationarity process.
            
            enforce_invertibility : bool, optional
            Whether or not to require the moving average parameters to 
            correspond to an invertible process.
            
            concentrate_scale : bool, optional
            Whether or not to concentrate the scale (variance of the error term) 
            out of the likelihood. This reduces the number of parameters by one.
            This is only applicable when considering estimation by numerical 
            maximum likelihood.
            
            trend_offset : int, optional
            The offset at which to start time trend values. 
            Default is 1, so that if `trend='t'` the trend is equal to 1, 
            2, ..., nobs. Typically is only set when the model created by 
            extending a previous dataset.
            
            dates : array_like of datetime, optional
            If no index is given by `endog` or `exog`, an array-like object of 
            datetime objects can be provided.
            
            missing : str 
            Available options are 'none', 'drop', and 'raise'. 
            If 'none', no nan checking is done. If 'drop', any observations
            with nans are dropped. If 'raise', an error is raised. 
            Default is 'none'.

                
            validate_specification : bool, optional
            See from statsmodels.tsa.arima.model import ARIMA ?ARIMA
            
            start_params : : array_like, optional 
            Initial guess of the solution for the loglikelihood maximization.
            If None, the default is given by Model.start_params.
            
            transformed : bool, optional 
            Whether or not `start_params` is already transformed. 
            Default is True.
            
            includes_fixed : bool, optional
            If parameters were previously fixed with the `fix_params` method,
            this argument describes whether or not `start_params` also includes 
            the fixed parameters, in addition to the free parameters. 
            Default is False.
            
            method : str, optional 
            The method used for estimating the parameters of the model. 
            Valid options include 'statespace', 'innovations_mle', 
            'hannan_rissanen', 'burg', 'innovations', and 'yule_walker'. 
            Not all options are available for every specification 
            (for example 'yule_walker' can only be used with AR(p) models).
            
            method_kwargs : dict, optional
            Arguments to pass to the fit function for the parameter estimator 
            described by the `method` argument.
            
            gls: bool, optional 
            Whether or not to use generalized least squares (GLS) to estimate 
            regression effects. The default is False if `method='statespace'`
            and is True otherwise.
            
            gls_kwargs : dict, optional 
            Arguments to pass to the GLS estimation fit method. 
            Only applicable if GLS estimation is used 
            (see `gls` argument for details).
            
            cov_type : str, optional 
            The `cov_type` keyword governs the method for calculating the 
            covariance matrix of parameter estimates. Can be one of:

                - 'opg' for the outer product of gradient estimator
                - 'oim' for the observed information matrix estimator,
                calculated ing the method of Harvey (1989)
                - 'approx' for the observed information matrix estimator,
                calculated using a numerical approximation of the Hessian matrix.
                - 'robust' for an approximate (quasi-maximum likelihood) 
                covariance matrix that may be valid even in the presence of 
                some misspecifications. Intermediate calculations use the 
                'oim' method.
                - 'robust_approx' is the same as 'robust' except that the 
                intermediate calculations use the 'approx' method.
                - 'none' for no covariance matrix calculation.

            Default is 'opg' unless memory conservation is used to avoid
            computing the loglikelihood values for each observation, in which
            case the default is 'oim'.
            
            cov_kwds : dict or None, optional 
            A dictionary of arguments affecting covariance matrix computation.
            
            return_params : bool, optional 
            Whether or not to return only the array of maximizing parameters.
            Default is False.
            
            low_memory : bool, optional 
            If set to True, techniques are applied to substantially reduce 
            memory usage. If used, some features of the results object will 
            not be available (including smoothed results and in-sample 
                              prediction), 
            although out-of-sample forecasting is possible. Default is False.
    
    
    Returns:
        ARIMA object of easyforecast.arima module
        

            ---------------------------
                  Example:
            ---------------------------
            
    import pandas as pd
    from easyforecast.arima import ARIMA
    df = pd.read_csv("https://raw.githubusercontent.com/Akai01/example-time-series-datasets/main/Data/retail.csv", sep= ",")
    df_sub = df[['date', 'series_38']]

    df_sub.columns = ["ds", "y"]
    df_sub.head()
    
    train = df_sub[:-12]
    test = df_sub[-12:]
    test_series = pd.DataFrame(pd.array(test['y']), 
                               index= pd.to_datetime( test["ds"]))
    test_series.columns = ["test"]
    train.head()
    
    model = ARIMA(train, freq = "MS")
    model = model.auto_arima(m = 12, ic = "aicc")
    model = model.forecast(h = 12)
    fc =  model.get_forecast()
    
    test_series.plot()
    model.plot()

        """
        y = self.y 
        xreg = self.xreg
        freq = self.freq
        data = pd.DataFrame(self.y, index= self.ds)
        
        if not enforce_stationarity:
            if d == None: 
                d = ndiffs(y, alpha = alpha, max_lags = None, max_d = max_d)
            if D == None: 
                D = nsdiffs(y, alpha = alpha, m = m, max_D = max_D)
        if enforce_stationarity:
            if d == None: 
                d = 0
                print("d is not specified, it set to default value 0")
            if D == None: 
                D = 0
                print("D is not specified, it set to default value 0")
            if not D == None:
                print("Please consider re-specifying the parameter D...")
            if not d == None:
                print("Please consider re-specifying the parameter d...")
        
        best_ic = float('inf')
        
        for i in range(start_p, max_p):
            for k in range(start_q, max_q):
                for I in range(start_P, max_P):
                    for K in range(start_Q, max_Q):
                        model = stat_arima(
                            endog = data, exog = xreg, 
                            order=(i,d,k), 
                            seasonal_order=(I, D, K, m),
                            freq = freq, trend = trend, 
                            enforce_stationarity = enforce_stationarity, 
                            enforce_invertibility = enforce_invertibility, 
                            concentrate_scale = concentrate_scale, 
                            trend_offset = trend_offset, 
                            dates = dates, missing = missing, 
                            validate_specification = validate_specification)
                        fit = model.fit(start_params = start_params,
                                        transformed = transformed,
                                        includes_fixed = includes_fixed,
                                        method = method, 
                                        method_kwargs = method_kwargs,
                                        gls = gls,
                                        gls_kwargs = gls_kwargs,
                                        cov_type = cov_type,
                                        cov_kwds = cov_kwds,
                                        return_params = return_params,
                                        low_memory = low_memory)
                        if ic == "aic":
                            model_ic = fit.aic
                        elif ic == "aicc":
                            model_ic = fit.aicc
                        elif ic == "bic":
                            model_ic = fit.bic
                        else:
                            model_ic = fit.aic
                            print("spesification of ic was not clear, aic is selected")
                        if model_ic < best_ic:
                            best_ic = model_ic
                            best_fit = fit
        self.best_model = best_fit
        self.model = model
        self.ic = ic
        return self
    
    def summary(self):
        """
        Object Summaries.
        
        Summaries details of selected arima model.
        
        
        ---------------------------
                  Example:
        ---------------------------
    import pandas as pd
    from easyforecast.arima import ARIMA
    df = pd.read_csv("https://raw.githubusercontent.com/Akai01/example-time-series-datasets/main/Data/retail.csv", sep= ",")
    df_sub = df[['date', 'series_38']]

    df_sub.columns = ["ds", "y"]
    df_sub.head()
    
    train = df_sub[:-12]
    test = df_sub[-12:]
    test_series = pd.DataFrame(pd.array(test['y']), 
                               index= pd.to_datetime( test["ds"]))
    test_series.columns = ["test"]
    train.head()
    
    model = ARIMA(train, freq = "MS")
        
    model.summary()
    
        """
        best_model = self.best_model
        return best_model.summary()
    
    def forecast(self, h = 10, xreg = None):
        """ 
        Forecast an ARIMA object.
        
        Args:
            h : int
            Forecast horizon.
            xreg : array_like
            Future external data
        
        Returns:
            ARIMA object of easyforecast.arima module
            
    
   
        ---------------------------
                  Example:
        ---------------------------
    
    import pandas as pd
    from easyforecast.arima import ARIMA
    df = pd.read_csv("https://raw.githubusercontent.com/Akai01/example-time-series-datasets/main/Data/retail.csv", sep= ",")
    df_sub = df[['date', 'series_38']]

    df_sub.columns = ["ds", "y"]
    df_sub.head()
    
    train = df_sub[:-12]
    test = df_sub[-12:]
    test_series = pd.DataFrame(pd.array(test['y']), 
                               index= pd.to_datetime( test["ds"]))
    test_series.columns = ["test"]
    train.head()
    
    model = ARIMA(train, freq = "MS")
    model = model.auto_arima(m = 12, ic = "aicc")
    model = model.forecast(h = 12)
    
    model..accuracy()
    
        """
        y = self.y
        best_model = self.best_model
        mean = best_model.predict(start = len(y), end = len(y) + h - 1)
        self.fitted = best_model.predict()
        self.mean = mean
        self.upper_95 = mean + 1.96*np.std(y)/np.sqrt(len(y))
        self.upper_80 = mean + 1.28*np.std(y)/np.sqrt(len(y))
        self.lower_95 = mean - 1.96*np.std(y)/np.sqrt(len(y))
        self.lower_80 = mean - 1.96*np.std(y)/np.sqrt(len(y))
        return self
    def residuals(self):
        """
        Extract Model Residuals
        
        residuals method  extracts model residuals from an auto_arima model.
        
        Returns:
            A numpy array.
        
        """
        y = self.y
        y_head = self.fitted
        
        return np.array(y - y_head)
        
        
    def get_forecast(self):
        """
        Get Forecasted values and prediction intervals as a pandas data frame.
        
        Returns:
            A Pandas DataFrame
            
        ---------------------------
                  Example:
        ---------------------------
    import pandas as pd
    from easyforecast.arima import ARIMA
    df = pd.read_csv("https://raw.githubusercontent.com/Akai01/example-time-series-datasets/main/Data/retail.csv", sep= ",")
    df_sub = df[['date', 'series_38']]

    df_sub.columns = ["ds", "y"]
    df_sub.head()
    
    train = df_sub[:-12]
    test = df_sub[-12:]
    test_series = pd.DataFrame(pd.array(test['y']), 
                               index= pd.to_datetime( test["ds"]))
    test_series.columns = ["test"]
    train.head()
    
    model = ARIMA(train, freq = "MS")
    model = model.auto_arima(m = 12, ic = "aicc")
    model = model.forecast(h = 12)
    fc =  model.get_forecast()
    
        
        """
        out = pd.DataFrame(
            {
             "mean":self.mean, 
             "upper_95": self.upper_95, 
             "upper_80" : self.upper_80, 
             "lower_95":self.lower_95,
             "lower_80": self.lower_80},
            index= pd.date_range(start= max(self.ds), freq = self.freq,
                                 periods= len(self.mean))[1:])
        return out
    
    def accuracy(self,  test_set = None):
        """ 
        Accuracy measures for a easyforecast ARIMA model
        
        
        Args:
            test_set : array_like
                A dataframe same length as forecast horizon 'h' and same 
                structure as imput dataframe 'df'.
            
        Returns:
            Dict[str, str]: with following values:
            1. Mean error
            2. Root mean squared error
            3. mean absolute error
            4. Mean percentage error
            5. Mean absolute percentage error
            
            
        ---------------------------
                  Example:
        ---------------------------
        
    import pandas as pd
    from easyforecast.arima import ARIMA
    df = pd.read_csv("https://raw.githubusercontent.com/Akai01/example-time-series-datasets/main/Data/retail.csv", sep= ",")
    df_sub = df[['date', 'series_38']]

    df_sub.columns = ["ds", "y"]
    df_sub.head()
    
    train = df_sub[:-12]
    test = df_sub[-12:]
    test_series = pd.DataFrame(pd.array(test['y']), 
                               index= pd.to_datetime( test["ds"]))
    test_series.columns = ["test"]
    train.head()
    
    model = ARIMA(train, freq = "MS")
    model = model.auto_arima(m = 12, ic = "aicc")
    model = model.forecast(h = 12)
    
    model.accuracy()
    
        """
        Test_accuracy = "Please provide test set as a data frame"
        pred = self.mean.values
        fitted = self.fitted
        y = self.y

        if type(test_set) == type(self.df):
            actual = test_set.y
            Test_accuracy = accuracy_util(actual, pred)
        
        out = {"Training set": accuracy_util(y, fitted),
               "Test set" : Test_accuracy}
        
        return out
    
    def plot(self, y_axis = 'Value'):
        """
        
        Plot an auto_arima  object
        
        
        Args:
            y_axis: A string to specify y axis
            
    
    
    ----------------------------
            Example:
    ----------------------------
    import pandas as pd
    from easyforecast.arima import ARIMA
    df = pd.read_csv("https://raw.githubusercontent.com/Akai01/example-time-series-datasets/main/Data/retail.csv", sep= ",")
    df_sub = df[['date', 'series_38']]

    df_sub.columns = ["ds", "y"]
    df_sub.head()
    
    train = df_sub[:-12]
    test = df_sub[-12:]
    test_series = pd.DataFrame(pd.array(test['y']), 
                               index= pd.to_datetime( test["ds"]))
    test_series.columns = ["test"]
    train.head()
    
    model = ARIMA(train, freq = "MS")
    model = model.auto_arima(m = 12, ic = "aicc")
    model = model.forecast(h = 12)
    fc =  model.get_forecast()
    
    test_series.plot()
    model.plot()
            """
        df = pd.DataFrame({"y" : self.y}, index= self.ds)
        ax =  df["y"].plot(label='observed')
        fc = pd.DataFrame(
            {
             "mean":self.mean, 
             "upper_95": self.upper_95, 
             "upper_80" : self.upper_80, 
             "lower_95":self.lower_95,
             "lower_80": self.lower_80},
            index= pd.date_range(start= max(self.ds), freq= self.freq,
                                 periods= len(self.mean))[1:])
        fc["mean"].plot(ax=ax, label='Point forecast')
        ax.fill_between(fc.index, fc.iloc[:, 1], fc.iloc[:, 4], color='gray')
        ax.set_xlabel('Date')
        ax.set_ylabel(y_axis)
        plt.legend()  