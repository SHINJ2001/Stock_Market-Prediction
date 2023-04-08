import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas import datetime
import math, time
import itertools
from sklearn import preprocessing
import datetime
plt.style.use('seaborn-darkgrid')
import warnings
warnings.filterwarnings("ignore")

def normalize_data(df):
    min_max_scaler = preprocessing.MinMaxScaler()
    df['Open'] = min_max_scaler.fit_transform(df.Open.values.reshape(-1,1))
    df['High'] = min_max_scaler.fit_transform(df.High.values.reshape(-1,1))
    df['Low'] = min_max_scaler.fit_transform(df.Low.values.reshape(-1,1))
    df['Volume'] = min_max_scaler.fit_transform(df.Volume.values.reshape(-1,1))
    df['Close'] = min_max_scaler.fit_transform(df.Close.values.reshape(-1,1))
    return df

def visualize():
    df = pd.read_csv("prices_data/price_data.csv", parse_dates = ['Date'])
    df2 = df.truncate(before = '2023-01-01')
    
    df2.set_index("Date")
    plt.subplot(1,2,1)
    plt.plot(df2.Open, label='Open')
    plt.legend(loc='best')
    plt.plot(df2.Low, label='Low')
    plt.legend(loc='best')
    plt.plot(df2.High,label='High')
    plt.legend(loc='best')
    plt.plot(df2.Close, label='Close')
    plt.subplot(1,2,2)
    plt.plot(df2.Volume, label='Volume')
    plt.legend(loc='best')
    plt.tight_layout()
    plt.gcf().autofmt_xdate()
    plt.show()

    return df

visualize()
