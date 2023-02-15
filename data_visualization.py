import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas import datetime
import math, time
import itertools
from sklearn import preprocessing
import datetime

def normalize_data(df):
    min_max_scaler = preprocessing.MinMaxScaler()
    df['Open'] = min_max_scaler.fit_transform(df.Open.values.reshape(-1,1))
    df['High'] = min_max_scaler.fit_transform(df.High.values.reshape(-1,1))
    df['Low'] = min_max_scaler.fit_transform(df.Low.values.reshape(-1,1))
    df['Volume'] = min_max_scaler.fit_transform(df.Volume.values.reshape(-1,1))
    df['Close'] = min_max_scaler.fit_transform(df.Close.values.reshape(-1,1))
    return df

def visualize():
    df = pd.read_csv("prices_data/price_data.csv", index_col = 0)
    print(df)


    df.plot(figsize=(23,8),title = "Apple Stock Price Analysis")
    plt.subplot(411)
    plt.plot(df.Open, label='Original')
    plt.legend(loc='best')
    plt.subplot(412)
    plt.plot(df.Low, label='Trend')
    plt.legend(loc='best')
    plt.subplot(413)
    plt.plot(df.High,label='Seasonality')
    plt.legend(loc='best')
    plt.subplot(414)
    plt.plot(df.Volume, label='Residuals')
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()
    
    df = normalize_data(df)
    df.plot(figsize=(23,10))
    plt.show()
    plt.subplot(411)
    plt.plot(df.Open, label='Original')
    plt.legend(loc='best')
    plt.subplot(412)
    plt.plot(df.Low, label='Trend')
    plt.legend(loc='best')
    plt.subplot(413)
    plt.plot(df.High,label='Seasonality')
    plt.legend(loc='best')
    plt.subplot(414)
    plt.plot(df.Volume, label='Residuals')
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()
