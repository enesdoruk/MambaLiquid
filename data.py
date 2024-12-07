import pandas as pd 
import numpy as np
from sklearn.preprocessing import MinMaxScaler


def load_data(stock):
    data_x = stock[['Open', 'High', 'Low', 'Volume']].values
    data_y = stock[['Close']].values
    
    test_set_size = int(np.round(0.1 * len(data_x)))
    train_set_size = len(data_x) - test_set_size
    
    x_train = data_x[:train_set_size]
    y_train = data_y[:train_set_size]
    x_test = data_x[train_set_size:]
    y_test = data_y[train_set_size:]
    
    return x_train, y_train, x_test, y_test

dates = pd.date_range('2010-01-02', '2017-11-10', freq='B')
df1 = pd.DataFrame(index=dates)
df_ibm = pd.read_csv("data/StockPrice/Data/Stocks/ibm.us.txt", parse_dates=True, index_col=0)
df_ibm = df1.join(df_ibm)

df_ibm = df_ibm[['Open', 'High', 'Low', 'Close', 'Volume']]

df_ibm = df_ibm.fillna(method='ffill')

scaler = MinMaxScaler(feature_range=(-1, 1))
df_ibm[['Open', 'High', 'Low', 'Close', 'Volume']] = scaler.fit_transform(df_ibm[['Open', 'High', 'Low', 'Close', 'Volume']])

x_train, y_train, x_test, y_test = load_data(df_ibm)

print('x_train.shape =', x_train.shape)
print('y_train.shape =', y_train.shape)
print('x_test.shape =', x_test.shape)
print('y_test.shape =', y_test.shape)