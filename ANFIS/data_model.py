import pandas as pd
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

def anfis(df):
    # Define input and output variables
    stock_price = ctrl.Antecedent(np.arange(df['Price'].min(), df['Price'].max()+1, 1), 'stock_price')
    prediction = ctrl.Consequent(np.arange(0, 101, 1), 'prediction')

# Define fuzzy membership functions for input variable
    stock_price['low'] = fuzz.trimf(stock_price.universe, [df['Price'].min(), df['Price'].min(), df['Price'].mean()])
    stock_price['medium'] = fuzz.trimf(stock_price.universe, [df['Price'].min(), df['Price'].mean(), df['Price'].max()])
    stock_price['high'] = fuzz.trimf(stock_price.universe, [df['Price'].mean(), df['Price'].max(), df['Price'].max()])
    
    # Define fuzzy membership functions for output variable
    prediction['low'] = fuzz.trimf(prediction.universe, [0, 0, 50])
    prediction['medium'] = fuzz.trimf(prediction.universe, [0, 50, 100])
    prediction['high'] = fuzz.trimf(prediction.universe, [50, 100, 100])
    
    # Define fuzzy rules
    rule1 = ctrl.Rule(stock_price['low'], prediction['low'])
    rule2 = ctrl.Rule(stock_price['medium'], prediction['medium'])
    rule3 = ctrl.Rule(stock_price['high'], prediction['high'])
    
    # Create control system and add rules
    prediction_ctrl = ctrl.ControlSystem([rule1, rule2, rule3])
    
    # Create a simulation
    simulation = ctrl.ControlSystemSimulation(prediction_ctrl)
    
    # Set input values
    simulation.input['stock_price'] = df['Price'][0]
    
    # Compute output values
    simulation.compute()
    
    # Print prediction
    print(simulation.output['prediction'])
