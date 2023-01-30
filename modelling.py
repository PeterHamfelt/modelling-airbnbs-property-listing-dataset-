import numpy as np
import pandas as pd
import os
import itertools
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error,r2_score
from tabular_data import load_airbnb
from sklearn.preprocessing import normalize


def plot_prediction(y_pred,y_true):
    y_pred = y_pred
    y_true = y_true
    samples = len(y_pred)
    plt.figure()
    plt.scatter(np.arange(samples), y_pred, c='r', label='predictions')
    plt.scatter(np.arange(samples), y_true, c='b', label='true labels', marker='x')
    plt.legend()
    plt.xlabel('Sample numbers')
    plt.ylabel('Values')
    plt.show()   
    
def custom_tune_regression_model_hyperparameters(model_type, X,y, hyperparameter_grid: list):
    X = normalize(X)
    x_train, x_test, y_train, y_test = model_selection.train_test_split(X,y, test_size= 0.3)
    x_val, x_test, y_val, y_test = model_selection.train_test_split(x_test,y_test, test_size=0.5)
    
    best_performance_metrics = {"Training RMSE": 0,"Testing RMSE": 0, "Validation RMSE": 0}
    best_hyperparameter_combination = None
    best_model = None
        
    for combination in hyperparameter_grid:
        
        model = model_type(**combination, verbose = 2, max_iter = 20000, early_stopping = False)
        model.fit(x_train,y_train)
        
        y_train_pred = model.predict(x_train)
        y_test_pred = model.predict(x_test)
        y_val_pred = model.predict(x_val)
        
        y_train_RMSE = np.sqrt(mean_squared_error(y_train, y_train_pred))
        y_test_RMSE = np.sqrt(mean_squared_error(y_test, y_test_pred))
        y_val_RMSE = np.sqrt(mean_squared_error(y_val, y_val_pred))
        
        if y_val_RMSE < best_performance_metrics["Validation RMSE"]:
            best_model = model
            best_hyperparameter_combination = combination
            best_performance_metrics["Validation RMSE"] = y_val_RMSE
            best_performance_metrics["Training RMSE"] = y_train_RMSE
            best_performance_metrics["Testing RMSE"] = y_test_RMSE 
                     
    print(best_hyperparameter_combination)
                
    return best_model, best_hyperparameter_combination, best_performance_metrics

def create_hyperparameter_grid(hyperparameter_dict: dict):
    hyperparameters = hyperparameter_dict.keys()
    values = hyperparameter_dict.values()
    hyperparameter_combinations = itertools.product(*values)
    combination_grid = [dict(zip(hyperparameters,combinations)) for combinations in hyperparameter_combinations]
    
    print(combination_grid)
    
    return combination_grid   
    
script_dir = os.path.dirname(os.path.realpath(__file__))
df = pd.read_csv(os.path.join(script_dir,"data/tabular_data/clean_tabular_data.csv"))
X,y = load_airbnb(df,"Price_Night")
hyperparameters = {"learning_rate":["invscaling","adaptive"],"eta0":np.linspace(0.01,0.001,5)}

hyperparameters_combination = create_hyperparameter_grid(hyperparameters)

best_model = custom_tune_regression_model_hyperparameters(SGDRegressor,X,y,hyperparameters_combination)

# x_train, x_test, y_train, y_test = model_selection.train_test_split(X,y, test_size= 0.3)
# x_val, x_test, y_val, y_test = model_selection.train_test_split(x_test,y_test, test_size=0.5)

# model = SGDRegressor(verbose = 2)
# model.fit(x_train,y_train)

# y_pred = model.predict(x_test)

# RMSE = np.sqrt(mean_squared_error(y_test,y_pred))
# r2 = r2_score(y_test,y_pred)
# print(f"The model RMSE is {RMSE} and r2 score is {r2}")

# graph_1 = plot_prediction(y_pred,y_test)
