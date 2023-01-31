import numpy as np
import pandas as pd
import os
import itertools
import matplotlib.pyplot as plt
import joblib
import json
from sklearn import model_selection
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error,r2_score
from tabular_data import load_airbnb
from sklearn.preprocessing import normalize
from sklearn.model_selection import GridSearchCV


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

def tune_regression_model_hyperparameters(model_type,hyperparameter_dict:dict):
    model = model_type()
    
    gs = GridSearchCV(model,
                      hyperparameter_dict,
                      scoring = "neg_root_mean_squared_error")
    
    gs.fit(x_train,y_train)

    y_train_pred = gs.predict(x_train)
    y_test_pred = gs.predict(x_test)
    y_val_pred = gs.predict(x_val)
    
    y_train_RMSE = np.sqrt(mean_squared_error(y_train,y_train_pred))
    y_test_RMSE = np.sqrt(mean_squared_error(y_test, y_test_pred))
    y_val_RMSE = np.sqrt(mean_squared_error(y_val, y_val_pred))
    performance_metrics = {"Train RMSE": y_train_RMSE,"Test RMSE": y_test_RMSE, "Validation RMSE": y_val_RMSE}
    
    print(y_test_RMSE, y_val_RMSE)
    
    return gs, performance_metrics, gs.best_params_, gs.best_score_

def save_model(model,performance_metrics,hyperparameter_combination,folder):
    working_dir = os.path.dirname(os.path.realpath(__file__))
    save_dir = os.path.join(working_dir,folder)
    model_saved_path = os.path.join(save_dir,"model.joblib")
    hyperparameter_json = os.path.join(save_dir,"hyperparameters.json")
    performance_metrics_json = os.path.join(save_dir,"metrics.json")
    
    if os.path.exists(save_dir) == False:
        os.makedirs(save_dir)
        
    joblib.dump(model,model_saved_path)
    
    with open(hyperparameter_json,"w") as file:
        json.dump(hyperparameter_combination,file)
    
    with open(performance_metrics_json,"w") as file:
        json.dump(performance_metrics,file)
    
    
script_dir = os.path.dirname(os.path.realpath(__file__))
df = pd.read_csv(os.path.join(script_dir,"data/tabular_data/clean_tabular_data.csv"))
X,y = load_airbnb(df,"Price_Night")
X = normalize(X)
x_train, x_test, y_train, y_test = model_selection.train_test_split(X,y, test_size= 0.3, random_state= 42)
x_val, x_test, y_val, y_test = model_selection.train_test_split(x_test,y_test, test_size=0.5, random_state= 42)
hyperparameters = {"learning_rate":["invscaling","adaptive"],"eta0":np.linspace(0.01,0.001,5)}

# hyperparameters_combination = create_hyperparameter_grid(hyperparameters)
# best_model = custom_tune_regression_model_hyperparameters(SGDRegressor,X,y,hyperparameters_combination)

best_model, performance_metrics ,best_hyperparameter_combination, best_RMSE = tune_regression_model_hyperparameters(SGDRegressor,hyperparameters)
save_model_folder = "models/regression/linear_regression"
save_model(best_model,performance_metrics,best_hyperparameter_combination,save_model_folder)
