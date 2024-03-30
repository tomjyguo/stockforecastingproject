import quandl
import numpy as np
from sklearn.impute import SimpleImputer
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt
import pandas as pd

def preprocessing(dataset):
    numerical_cols = ['Open', 'High', 'Low', 'Close', 'Adj. Open', 'Adj. Low', 'Adj. Close', 'Volume', 'MA_10', 'MA_50', 'Close Detrend', 'RSI', 'Volatility']

    dataset['MA_10'] = dataset['Close'].rolling(window = 10).mean()
    dataset['MA_50'] = dataset['Close'].rolling(window = 50).mean()

    exp12 = dataset['Close'].ewm(span = 12, adjust = False).mean()
    exp26 = dataset['Close'].ewm(span = 26, adjust = False).mean()
    dataset['MACD'] = exp12 - exp26

    dataset['Close Detrend'] = (dataset['Close'] - dataset['Close'].shift(30))

    delta = dataset['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    RS = gain / loss
    dataset['RSI'] = 100 - (100 / (1 + RS))

    dataset['Volatility'] = dataset['Close'].rolling(window=20).std()

    dataset['OBV'] = np.where(dataset['Close'] > dataset['Close'].shift(1), dataset['Volume'], 
                               np.where(dataset['Close'] < dataset['Close'].shift(1), -dataset['Volume'], 0)).cumsum()
    
    imputer = SimpleImputer(strategy='mean')
    dataset[numerical_cols] = imputer.fit_transform(dataset[numerical_cols])

    return dataset

def ADF_test(stock_close_price):
    adf_test = adfuller(stock_close_price, autolag = 'AIC')
    print('ADF Statistic: %f' % adf_test[0])
    print('p-value: %f' % adf_test[1])
    print('Critical Values:')
    for key, value in adf_test[4].items():
        print('\t%s: %.3f' % (key, value))
    
    if adf_test[1] < 0.05:
        print('Result: Series is Stationary')
    else:
        print('Result: Series is Not Stationary')
