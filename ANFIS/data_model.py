import numpy as np
import pandas as pd
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt
import sklearn.metrics
import math 
import sys
sys.path.insert(0, '../')
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")

def anfis(df):
    # Define the input and output variables
    input_variable = ctrl.Antecedent(np.arange(0, 101, 1), 'input_variable')
    output_variable = ctrl.Consequent(np.arange(0, 101, 1), 'output_variable')
    
    # Define the membership functions for input and output variables
    input_variable['low'] = fuzz.trimf(input_variable.universe, [0, 0, 50])
    input_variable['medium'] = fuzz.trimf(input_variable.universe, [0, 50, 100])
    input_variable['high'] = fuzz.trimf(input_variable.universe, [50, 100, 100])
    
    output_variable['buy'] = fuzz.trimf(output_variable.universe, [0, 0, 50])
    output_variable['hold'] = fuzz.trimf(output_variable.universe, [0, 50, 100])
    output_variable['sell'] = fuzz.trimf(output_variable.universe, [50, 100, 100])
    
    # Define the rules for the fuzzy system
    rule1 = ctrl.Rule(input_variable['low'], output_variable['buy'])
    rule2 = ctrl.Rule(input_variable['medium'], output_variable['hold'])
    rule3 = ctrl.Rule(input_variable['high'], output_variable['sell'])
    
    # Create the control system and simulate it with the input df
    control_system = ctrl.ControlSystem([rule1, rule2, rule3])
    simulation = ctrl.ControlSystemSimulation(control_system)
    
    for i in range(len(df)):
        simulation.input['input_variable'] = df['Close'][i]
        simulation.compute()
        df.loc[i, 'predicted_output'] = simulation.output['output_variable']

    row = round(0.9 * len(df))
    x_test = df['Close'][int(row):]
    y_test = df['predicted_output'][int(row):]
    mse = sklearn.metrics.mean_squared_error(x_test,
            y_test)

    rmse = math.sqrt(mse)

    # Print the predicted outputs
    plt.figure(figsize = (25,8))
    plt.plot(x_test, color = 'blue', label = 'Actual')
    plt.plot(y_test, color = 'red', label = 'Prediction')
    plt.legend(loc = 'best')
    plt.show()

    return rmse
