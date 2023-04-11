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
import seaborn as sns

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
    
    #Plot the opening, closing, highest and lowest values of the day
    df2.set_index("Date")
    plt.plot(df2.Open, label='Open')
    plt.legend(loc='best')
    plt.plot(df2.Low, label='Low')
    plt.legend(loc='best')
    plt.plot(df2.High,label='High')
    plt.legend(loc='best')
    plt.plot(df2.Close, label='Close')
    plt.legend(loc = 'best')
    plt.tight_layout()
    plt.gcf().autofmt_xdate()
    plt.show()

    # Plot the volumes
    plt.plot(df2.Volume, label='Volume')
    plt.legend(loc='best')
    plt.tight_layout()
    plt.gcf().autofmt_xdate()
    plt.show()
    
    #Plot the daily returns, i.e. percent change for the day
    df2['Daily Return'] = df2['Close'].pct_change()

    plt.plot(df2['Daily Return'], label = 'Daily Return')
    plt.show()

    #Plot histogram to get an overall look at the average daily return
    df2['Daily Return'].hist(bins = 50)
    plt.xlabel('Daily Return')
    plt.ylabel('Counts')
    plt.tight_layout
    plt.show()

    # plot all the features of the dataset 
    features = ['Open', 'High', 'Low', 'Close', 'Volume']
    plt.subplots(figsize=(20,10))
    for i, col in enumerate(features):
        plt.subplot(2,3,i+1)
        sns.distplot(df[col])
    plt.show()
    
    #Boxplot
    plt.subplots(figsize=(20,10))
    for i, col in enumerate(features):
        plt.subplot(2,3,i+1)
        sns.boxplot(df[col])
    plt.show()
    
    plt.figure(figsize=(10, 10))

    # As our concern is with the highly
    # correlated features only so, we will visualize
    # our heatmap as per that criteria only. This shall allow us to locate
    # highly clustered areas
    sns.heatmap(df2.corr() > 0.9, annot=True, cbar=False)
    plt.show()

    #Scatter plot to depict the clustering in a better sense
    plt.scatter(df2['Date'], df2['Close'])
    # Adding Title to the Plot
    plt.title("Scatter Plot")
    # Setting the X and Y labels
    plt.xlabel('Date')
    plt.ylabel('Close')
    plt.show()
    
    return df

visualize()
