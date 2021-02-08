#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  2 13:29:13 2021

@author: Resul Akay
"""
import sys
import numpy as np

def autocovariance(x, maxlag = 40):
    """
    Autocovariance of a time series.
    
    Args:
        x: A time series as a numpy array
        maxlag: Maximum lag 
    
    
    """
    n = len(x)
    if(maxlag>=n):
        sys.exit('maxlag > data. Data is too short')
    xbar = np.mean(x)
    def f(j):
        a1 = (x[0:(n - j)] - xbar)
        a2 = (x[(j):n]- xbar)
        return(sum(a1*a2)/n)
    out = [f(j) for j in range(0, maxlag+1)]
    return(out)

import pandas as pd
from statsmodels.tsa.stattools import kpss
import warnings


def is_constant(x):
    """ Is an numpy array constant?
    
    args:
        x : A numpy array.
    returns:
        True or False
    
    """
    x = np.array(x)
    result = np.all(x == x[1])
    return result


def ndiffs(x, alpha = 0.05, max_lags = None, max_d = 2):
    """ 
    Number of differences required for a stationary series
    
    Description: 
        Functions to estimate the number of differences required to make a 
        given time series stationary. ndiffs estimates the number of first 
        differences necessary.
        
    Args:
        x : A univariate time series
        alpha : Level of the test, possible values range from 0.01 to 0.1.
        test : Type of unit root test to use
        type : Specification of the deterministic component in the regression
        max.d : Maximum number of non-seasonal differences allowed
        """
    x = x[~pd.isnull(x)]
    d = 0
    if alpha < 0.01:
        print("Specified alpha value is less than the minimum, setting alpha=0.01")
        alpha = 0.01
    if  alpha > 0.1:
        print("Specified alpha value is larger than the maximum, setting alpha=0.1")
        alpha = 0.1
    if is_constant(x):
        final_d = d
    
    alpha = alpha
    if max_lags == None:
        max_lags = round(3 * np.sqrt(len(x))/13)
    warnings.simplefilter("ignore")
    kpss_stat, p_value, lags,  crit = kpss(x = x, lags = max_lags)
        
    while  (alpha < kpss_stat) & (d <= max_d):
        d = d + 1
        x = np.diff(x)
        kpss_stat, p_value, lags,  crit = kpss(x = x, lags = max_lags)
    if not is_constant(x): 
        final_d = d
    
    if final_d > max_d:
        final_d = max_d
        
    return final_d


def nsdiffs(x, alpha = 0.05, m = None, max_D = 1):
    D = 0
    if alpha < 0.01:
        print("Specified alpha value is less than the minimum, setting alpha=0.01")
        alpha = 0.01
    if  alpha > 0.1:
        print("Specified alpha value is larger than the maximum, setting alpha=0.1")
        alpha = 0.1
    if is_constant(x):
        final_D = D
    if m ==1:
        sys.exit("Non seasonal data")
    if m < 0 :
        print("I can't handle data with frequency less than 1.")
        print("Seasonality will be ignored.")
        final_D = 0
    if m > len(x):
        final_D = 0
    alpha = alpha
    warnings.simplefilter("ignore")
    kpss_stat, p_value, lags,  crit = kpss(x = x, lags = m)
        
    while  (alpha < kpss_stat) & (D <= max_D):
        D = D + 1
        x = np.diff(x)
        kpss_stat, p_value, lags,  crit = kpss(x = x, lags = m)
    if not is_constant(x) & m < 0 & m > len(x):
        final_D = D
    if final_D > max_D:
        final_D = max_D
        
    return final_D

def accuracy(actual, pred):
    """Accuracy measures for a forecast model
        
        
        Args:
            actual : A dataframe same length as forecast horizon 'h' and same 
            structure as imput dataframe 'df'.
            
        Returns:
            Dict[str, str]: with following values:
            1. Mean error
            2. Root mean squared error
            3. mean absolute error
            4. Mean percentage error
            5. Mean absolute percentage error
            """
    error = actual - pred
    pe = error/actual *100
    me = np.mean(error)
    mse = np.mean(np.power(error, 2))
    mae = np.mean(np.abs(error))
    mape = np.mean(np.abs(pe))
    mpe = np.mean(pe)
    rmse = np.sqrt(mse)
    out = {'Mean error' : me, 'Root mean squared error' : rmse,
           'Mean absolute error' : mae, 'Mean percentage error' : mpe,
           'Mean absolute percentage error' : mape}
    return out