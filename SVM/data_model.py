from sklearn.svm import SVC
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics 
plt.style.use('seaborn-darkgrid')
from sklearn.model_selection import train_test_split

def svm(df):
    df.index = pd.to_datetime(df['Date'])
    df = df.drop(['Date'], axis='columns')

    df['Open-Close'] = df.Open - df.Close
    df['High-Low'] = df.High - df.Low
      
    # Store all predictor variables in a variable X
    X = df[['Open-Close', 'High-Low']]
    y = np.where(df['Close'].shift(-1) > df['Close'], 1, 0)
    split_percentage = 0.8
    split = int(split_percentage*len(df))
    
    # Train data set
    X_train = X[:split]
    y_train = y[:split]
    
    # Test data set
    X_test = X[split:]
    y_test = y[split:]

    cls = SVC().fit(X_train, y_train)
    df['Predicted_Signal'] = cls.predict(X)
    df['Return'] = df.Close.pct_change()
    df['Strategy_Return'] = df.Return *df.Predicted_Signal.shift(1)
    df['Cum_Ret'] = df['Return'].cumsum()
    df['Cum_Strategy'] = df['Strategy_Return'].cumsum()

    plt.plot(df['Cum_Ret'],color='red')
    plt.plot(df['Cum_Strategy'],color='blue')
    plt.show()

    mse = sklearn.metrics.mean_squared_error(x_test, y_test)

    return mse
#    x = df.drop('Close', axis = 1)
#    y = df['Close']
#
#    X_train = x[:, :-1]
#    y_train = y[:, :-1]
#    
#    x_test = []
#    model = SVR(kernel = 'linear')
#    print('model stored')
#    model.fit(X_train, y_train)
#    print('model fitted')
#    y_pred = model.predict(X_test)
#    print('model predicted')
#    mse = mean_squared_error(y_test, y_pred)
#
#    return mse

df = pd.read_csv("../prices_data/price_data.csv", parse_dates = ['Date'])
x = svm(df)
print(x)
