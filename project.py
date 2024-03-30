import quandl
import os
from functions import preprocessing, ADF_test
import pandas as pd
import matplotlib.pyplot as plt
from pmdarima.arima import auto_arima

api_key = os.getenv('QUANDL_API_KEY')

def main(dataset, data, seasonal):
    stock_data = quandl.get(dataset, api_key=api_key)
    stock_data = preprocessing(stock_data)
    
    train_data, test_data = data_split(stock_data['Close Detrend'] * 0.8)

    if data == 'test':
        data = test_data
    else:
        data = train_data
    
    if seasonal:
        arima_model = auto_arima(
            data,
            start_p = 1, 
            start_q = 1,
            max_p = 3,
            max_q = 3,
            m = 12,
            start_P = 0, 
            seasonal = True,
            d = None,
            D = 1,
            trace = False,
            error_action = 'ignore',
            suppress_warnings = True,
            stepwise = True
        )
    else:
        arima_model = auto_arima(
            data,
            start_p=1, 
            start_q=1,
            max_p=5,
            max_q=5,
            m=1,
            d=None,
            test='adf',
            trace=False,
            alpha=0.05,
            scoring='mse',
            suppress_warnings=True,
            seasonal=False,
            stepwise=True
        )
    
    fitted_model = arima_model.fit(train_data)
    
    forecast_values = fitted_model.predict(len(test_data)) 
    fcv_series = pd.Series(forecast_values.values, index=test_data.index)
    
    plt.figure(figsize=(12, 5), dpi=100)
    plt.plot(train_data, label='Training')
    plt.plot(test_data, label='Actual Stock Price')
    plt.plot(fcv_series, label='Predicted Stock Price')
    plt.title('Auto ARIMA')
    plt.xlabel('Time')
    plt.ylabel('Close Price')
    plt.legend(loc='upper left', fontsize=8)
    plt.show()

if __name__ == "__main__":
    dataset = 'WIKI/AAPL' 
    data_type = 'train'  # Either 'train' or 'test'
    is_seasonal = True  # Either True or False
    main(dataset, data_type, is_seasonal)