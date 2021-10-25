import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

class ARAR:
    """
    ARAR is implementation of ararma algorithm in Python
        
        Attributes:
            df : pandas.DataFrame
                A pandas data frame which has two columns:
                ds the date column and y the univariate time series.
            h : int
                An integer to specify forecast horizon 
            freq: str
                The frequency of the data should be a pandas freq strings.
                See: <https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases>
            max_lag : int
                Maximum lag for autocorelation function
                
            
        
        References:
            Brockwell, Peter J., and Richard A. Davis. 
            Introduction to Time Series and Forecasting. 
            Chapter 10. 2nd ed. Springer, 2002
            
            Weigt George (2018), 
            itsmr: Time Series Analysis Using the Innovations Algorithm.
        Examples:
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
            
            model = ARAR(train, fh = 12, freq = "MS", max_lag= 26)
            model.forecast()
            model.get_forecast()
            model.accuracy(test_set = test)
            model.plot()
    """
    def __init__(self, df, h, freq, max_lag = 26):
        
        """
        Initiates ARAR algorithm to forecast a univariate time series
        automaticly without paramether tuning.
        
        Args:
            df: A pandas data frame which has two columns:
                ds the date column and y the univariate time series.
            h: An integer to specify forecast horizon
            freq: The frequency of the data
            max_lag: Maximum lag for autocorelation function
        
        Returns:
            h ahead forecast and prediction intervals at 95 and 80 confidence 
            interval.
            
            """
        self.y = np.array(df["y"])
        self.ds = pd.to_datetime(df["ds"])
        self.h = h
        self.freq = freq
        self.df = df
        self.max_lag = max_lag
        

    def forecast(self):
        """
        Forecast a univariate timeseries using ARAR algorithm.
        
        Args:
            No
        
        Returns:
            A ARAR model object.
        
        """
        h = self.h
        Y = y = self.y
        psi = [1]
        
        for k in np.array(range(0, 3)):
            n = len(y)
            phi =  [np.matmul(y[range(i+1,n)], y[
                range(0, n-i-1)])/sum(np.power(y[
                    range(0, n-i-1)], 2)) for i in range(0, 15)]
            err = [sum(np.power((y[range(i+1, n)]- phi[i]*y[
                range(0,(n - i-1))]), 2))/sum(np.power(y[
                    range(i,n)], 2))  for i in np.array(range(0, 15))]
            tau = np.array(np.where(err == np.min(err))).item()
            if(err[tau] <= 8/n or (phi[tau] >= 0.93 and tau > 3)):
                y = (y[tau+1:n] - phi[tau]*y[0:(n - tau-1)])
                tau2 = tau
                if tau==0:
                    tau2 = 1
                psi = np.concatenate((psi, np.zeros(tau2+1)))  - phi[
                    tau]*np.concatenate((np.zeros(tau2+1), psi))
            elif phi[tau] >= 0.93:
                A = np.zeros((2,2))
                A[0, 0] = sum(np.power(y[1:(n - 2)], 2))
                A[0, 1] = sum(np.array(y[0:(n - 3)]) * np.array(y[1:(n - 2)]))
                A[1, 0] = sum(np.array(y[0:(n - 3)]) * np.array(y[1:(n - 2)]))
                A[1, 1] = sum(np.power(y[0:(n - 3)], 2))
                b = (sum(np.array(y[2:(n-1)]) * np.array(y[
                    1:(n - 2)])), sum(np.array(y[
                        2:(n-1)]) * np.array(y[0:(n - 3)])))
                phi = np.linalg.lstsq(A, b, rcond = 1e-07)[0]
                y = np.array(y[2:(n-1)]) - np.array(phi[0]) * np.array(y[
                    1:(n - 2)]) - np.array(phi[1]) * np.array(y[0:(n - 3)])
                psi = np.append(np.append(psi, 0), 0) - np.array(phi[
                    1]) * np.append(np.append(0, psi),0)- np.array(phi[
                        1])* np.append(np.append(0, 0),psi)
            else: 
                break
            
        S = y
        Sbar = np.mean(S)
        X = S - Sbar
        gamma = self._autocovariance(X)
        y = Y
        np.array(gamma)
        A = np.zeros((4,4)) + gamma[0] 
        b = np.zeros(4)
        best_sigma2 = float('inf')
        m = (26)
        del err, k, n, phi, tau
        
        # fitting a subset autoregression ------------------
        for i in range(1,(m-2)):
            for j in range((i + 1),(m - 1)):
                for k in range((j +1), m):
                    A[0, 1] = A[1, 0] = gamma[i]
                    A[0, 2] = A[2, 0] = gamma[j]
                    A[1, 2] = A[2, 1] = gamma[j - i ]
                    A[0, 3] = A[3, 0] = gamma[k]
                    A[1, 3] = A[3, 1] = gamma[k - i ]
                    A[2, 3] = A[3, 2] = gamma[k - j ]
                    b[0] = gamma[1]
                    b[1] = gamma[i+1]
                    b[2] = gamma[j+1]
                    b[3] = gamma[k+1]
                    phi = np.linalg.inv(A.T @ A) @ A.T @ b
                    sigma2 = gamma[0] - phi[0] * gamma[1] - phi[1] * gamma[
                        i + 1] - phi[2] * gamma[j + 1] - phi[3] * gamma[k + 1]
                    if(sigma2 < best_sigma2):
                        best_sigma2 = sigma2
                        best_phi = phi
                        best_lag = (0,i,j,k)
        i = best_lag[1]
        j = best_lag[2]
        k = best_lag[3]
        phi = best_phi
        sigma2 = best_sigma2 
        del A, b, best_lag, best_phi, best_sigma2, gamma, m, S, X, Y
        
        xi = np.concatenate((psi, np.zeros(k+1))) - phi[0]*np.concatenate(
            ([0], psi, np.zeros(k))) - phi[1]*np.concatenate(
                (np.zeros(i+1), psi, np.zeros(k-i))) - phi[2]*np.concatenate(
                    (np.zeros(j+1), psi, np.zeros(k-j))) - phi[
                        3]*np.concatenate((np.zeros(k+1), psi))
        
        n = len(y)
        k = len(xi)
        y = np.concatenate((y, np.zeros(h)))
        c = (1-sum(phi))*Sbar
        
        for i in range(1, h+1):
            y[n-1+i] = c - sum(xi[range(1, k)]*y[np.array(n+i-1-np.array(
                range(1,k))).astype(int)])
        
        mean = y[n-1 + np.array(range(1,h+1))]
        self.mean = mean
        self.upper_95 = mean + 1.96*np.std(y)/np.sqrt(len(y))
        self.upper_80 = mean + 1.28*np.std(y)/np.sqrt(len(y))
        self.lower_95 = mean - 1.96*np.std(y)/np.sqrt(len(y))
        self.lower_80 = mean - 1.96*np.std(y)/np.sqrt(len(y))
        return self

    def get_forecast(self):
        out = pd.DataFrame(
            {
             "mean":self.mean, 
             "upper_95": self.upper_95, 
             "upper_80" : self.upper_80, 
             "lower_95":self.lower_95,
             "lower_80": self.lower_80},
            index= pd.date_range(start= max(self.ds), freq= self.freq,
                                 periods= self.h +1)[1:])
        return out
    
    def accuracy(self, test_set):
        """ 
        Accuracy measures for a forecast model
        
        
        Args:
            test_set : A dataframe same length as forecast horizon 'h' and same 
            structure as imput dataframe 'df'.
            
        Returns:
            Dict[str, str]: with following values:
            1. Mean error
            2. Root mean squared error
            3. mean absolute error
            4. Mean percentage error
            5. Mean absolute percentage error
        """
        
        pred = self.mean
        Test_accuracy = "Please provide test set as a data frame"
        if type(test_set) == type(self.df):
            actual = test_set['y']
            Test_accuracy = self._accuracy(actual, pred)
        return Test_accuracy
    
    def plot(self, y_axis = 'Value'):
        """Plot an ARAR  object
        Args:
            
            y_axis: A string to specify y axis"""
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
                                 periods= self.h +1)[1:])
        fc["mean"].plot(ax=ax, label='Point forecast')
        ax.fill_between(fc.index, fc.iloc[:, 1], fc.iloc[:, 4], color='gray')
        ax.set_xlabel('Date')
        ax.set_ylabel(y_axis)
        plt.legend()
    def _autocovariance(self, x):
        """ Autocovariance of a time series. 
        Args: 
            x: A time series as a numpy array
            """
        max_lag = self.max_lag
        n = len(x)
        if(max_lag < n):
            assert "The series too short"
        xbar = np.mean(x)
        def f(j):
            a1 = (x[0:(n - j)] - xbar)
            a2 = (x[(j):n]- xbar)
            return(sum(a1*a2)/n)
        out = [f(j) for j in range(0, max_lag+1)]
        return(out)
    
    def _accuracy(self, actual, pred):
        """Accuracy measures for a forecast model
        Args:
            actual : A dataframe same length as forecast horizon 'h' and same 
            structure as imput dataframe 'df'.
            pred : The mean forecast
            
        Returns:
            Dict[str, str]: with following values:
            1. Mean error
            2. Root mean squared error
            3. mean absolute error
            4. Mean percentage error
            5. Mean absolute percentage error """
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