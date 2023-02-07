import numpy as np
import pandas as pd
import os
import itertools
import matplotlib.pyplot as plt
import joblib
import json
from sklearn import model_selection
from sklearn.linear_model import SGDRegressor, LogisticRegression
from sklearn.metrics import mean_squared_error,r2_score, accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, ConfusionMatrixDisplay
from tabular_data import load_airbnb
from sklearn.preprocessing import normalize
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor


def plot_prediction(y_pred,y_true):
    """Plot prediction vs actual

   Plot scatter plot using matplotlib to visualise and compare the values predicted by the model and 
   the actual value. 

    Args:
        y_pred (np.array): THe predicted values made by the model
        y_true (np.array): The actual values for the predicting label
    """
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
    """Tune regression model's hyperparameters

    This function will first normalise the features of the dataset before splitting them into training,
    testing and validation sets. It will then loop through all the possible combination of hyperparameters
    generated using the create_hyperparameter_grid() function to find the combination of hyperparameters which
    will produce the best performing model. 

    Args:
        model_type (_type_): The model to be tuned.  
        X (_type_): Features columns
        y (_type_): Label columns
        hyperparameter_grid (list): List of dictionary of all possible combination of hyperparameters.

    Returns:
        sklearn.linear_model: The best performing model 
        dict: A dictionary of the best combination of hyperparameters and its values
        dict: A dictionary which consist of the train, test and validation RMSE values
    """
    
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
    """Create hyperparameter grid

    The function takes in a hyperparameter dictionary which consist of all the different hyperparameters
    and the values to try then generate a list of dictionary where each dictionary is a different combination
    of hyperparameters.  
    
    Args:
        hyperparameter_dict (dict): A dictionary that consist of the different hyperparameters and its values to try.

    Returns:
        list: All possible combination of hyperparameters in each dictionary. 
    """
    hyperparameters = hyperparameter_dict.keys()
    values = hyperparameter_dict.values()
    hyperparameter_combinations = itertools.product(*values)
    combination_grid = [dict(zip(hyperparameters,combinations)) for combinations in hyperparameter_combinations]
    
    print(combination_grid)
    
    return combination_grid   

def tune_regression_model_hyperparameters(model_type,X,y,hyperparameter_dict:dict):
    """Tune model using GridSearchCV

    This function uses Sklearn's GridSearchCV to search for the best possible combination of hyperparameters to 
    achieve the best model. 

    Args:
        model_type (sklearn): The Sklearn model type to perform grid search on
        hyperparameter_dict (dict): The dictionary which consist of the hyperparameters and its values to be tried.

    Returns:
        sklearn.linear_model: The best performing model 
        dict: A dictionary which consist of the train, test and validation RMSE values
        dict: A dictionary of the best combination of hyperparameters and its values
        np.float64: The best score achieved during grid search
    """
    x_train, x_test, y_train, y_test = model_selection.train_test_split(X,y, test_size=0.3, random_state=42)
    x_test, x_val, y_test, y_val = model_selection.train_test_split(x_test, y_test, test_size=0.5, random_state=42)
    
    model = model_type(random_state = 42)
    
    gs = GridSearchCV(model,
                      hyperparameter_dict,
                      scoring = "neg_root_mean_squared_error")
    
    gs.fit(x_train,y_train)

    y_train_pred = gs.predict(x_train)
    y_test_pred = gs.predict(x_test)
    y_val_pred = gs.predict(x_val)
    
    train_RMSE = np.sqrt(mean_squared_error(y_train,y_train_pred))
    test_RMSE = np.sqrt(mean_squared_error(y_test, y_test_pred))
    val_RMSE = np.sqrt(mean_squared_error(y_val, y_val_pred))
    performance_metrics = {"Train RMSE": train_RMSE,"Test RMSE": test_RMSE, "Validation RMSE": val_RMSE}
    
    print(test_RMSE, val_RMSE)
    
    return gs.best_estimator_, performance_metrics, gs.best_params_, gs.best_score_

def tune_classification_model_hyperparameters(model_type,X,y,hyperparameter_dict):
    """Tune classification model's hyperparameters

    _extended_summary_

    Args:
        model_type (list): _description_
        X (_type_): _description_
        y (_type_): _description_
        hyperparameter_dict (_type_): _description_

    Returns:
        _type_: _description_
    """
    
    x_train, x_test, y_train, y_test = model_selection.train_test_split(X,y, test_size=0.3, random_state=42)
    x_test, x_val, y_test, y_val = model_selection.train_test_split(x_test,y_test, test_size=0.5, random_state=42)
    
    model = model_type(random_state= 42, max_iter = 10000)
    
    gs = GridSearchCV(model,
                      hyperparameter_dict)
    
    gs.fit(x_train, y_train)
    
    y_train_pred = gs.predict(x_train)
    y_test_pred = gs.predict(x_test)
    y_val_pred = gs.predict(x_val)
    
    train_acc = accuracy_score(y_train,y_train_pred)
    test_acc = accuracy_score(y_test,y_test_pred)
    val_acc = accuracy_score(y_val,y_val_pred)
    
    train_f1 = f1_score(y_train, y_train_pred, average="weighted")
    test_f1 = f1_score(y_test,y_test_pred, average="weighted")
    val_f1 = f1_score(y_val,y_val_pred, average="weighted")
    
    model_accuracy = {"Training Accuracy":train_acc, 
                      "Testing Accuracy":test_acc, 
                      "Validation Accuracy": val_acc}
    
    model_f1 = {"Training F1 Score":train_f1,
                 "Testing F1 Score":test_f1,
                 "Validation F1 Score":val_f1}
    
    performance_metrics = {"Model Accuracy":model_accuracy,"Model F1 Score":model_f1}
    
    return gs.best_estimator_, gs.best_params_, performance_metrics
    

def save_model(model,performance_metrics,hyperparameter_combination,folder):
    """Save model

    Save the best performing model together with the model's performance metrics and hyperparameter combination
    to the local folder name "models/regression/linear_regression". The function will first check if the local folder
    exist. If it doesn not exist it will make the folder before saving the model as a joblib file and the hyperparameter
    and performance metrics as json file.

    Args:
        model (Sklearn): The best performing model to be saved
        performance_metrics (dict): The performance metrics associated with the best performing model. It includes the
        training, testing and validation RMSE.
        hyperparameter_combination (dict): The combination of hyperparameter used to achieved the best performing model.
        folder (str): The folder directory.
    """
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
        
def evaluate_all_models(model_list,X,y,hyperparameter_list):
    """Evaluate different models

    Evaluate the performance of a list of different regression models by tunning their hyperparameters and comparing them
    to each other and the base linear regression model. At the same time create a folder for each of the models to save the
    model, its performance metrics and the hyperparameter combination used to achieved that result. 

    Args:
        model_list (list): A list of the sklearn model to be evaluated. 
        hyperparameter_list (list): A list of hyperparameter dictionary corresponding to each model. 
    """
    
    for model_type, hyperparameter_dict in zip(model_list,hyperparameter_list):
        
        model, performance_metrics, model_params, model_best_score = tune_regression_model_hyperparameters(model_type,X,y,hyperparameter_dict)
        
        save_path = "models/regression/{}".format(type(model).__name__)
        
        save_model(model,performance_metrics,model_params,save_path)
        
def find_best_model():
    """Find the best performing model

    This function finds the best performing model from all the saved models in the model/regression folder by firstly loading
    in and appending all the model's validation RMSE value into a list. From the list, the position of the lowest RMSE value can 
    be obtained and used to index the best model's path from the list of model's directory. From the best model's path, the best 
    model and its associated hyperparameters and performance metrics then can be load in. 
    
    Returns:
        _type_: The best sklearn model
        dict : The hyperparameters associated with the best model.
        dict : The performance metrics assocaited with the best model.  
    """
    model_folder = os.path.join(os.path.dirname(os.path.realpath(__file__)),"models/regression")
    different_models_folder = [folder[0] for folder in os.walk(model_folder)][1:]
    
    validation_RMSE_list = []
    
    for model_files in different_models_folder:
        
        performance_metrics_path = os.path.join(model_files,"metrics.json")
        
        with open(performance_metrics_path) as file:
            performance_metric = json.load(file)
            
        validation_RMSE_list.append(performance_metric["Validation RMSE"])
        
    lowest_RMSE = min(validation_RMSE_list)
    best_model_folder_index = validation_RMSE_list.index(lowest_RMSE)
    model_path = os.path.join(different_models_folder[best_model_folder_index],"model.joblib")
    hyperparameter_path = os.path.join(different_models_folder[best_model_folder_index],"hyperparameters.json")
    performance_metrics_path = os.path.join(different_models_folder[best_model_folder_index],"metrics.json")
    
    with open(model_path, "rb") as file:
        best_model = joblib.load(file)
    
    with open(hyperparameter_path) as file:
        best_model_hyperparameters = json.load(file)
        
    with open(performance_metrics_path) as file:
        best_model_performance_metrics = json.load(file)
    
    print(model_path,hyperparameter_path)
    
    return best_model, best_model_hyperparameters, best_model_performance_metrics

script_dir = os.path.dirname(os.path.realpath(__file__))
df = pd.read_csv(os.path.join(script_dir,"data/tabular_data/clean_tabular_data.csv"))
# hyperparameters = {"learning_rate":["invscaling","adaptive"],"eta0":np.linspace(0.01,0.001,5)}
gbr_hyperparameters = {"learning_rate": [0.1,0.01,0.001], 
                   "subsample": [1.0,0.1,0.01], 
                   "n_estimators":[10,50,100],
                   "max_depth": [4,6,8]}

dt_hyperparameters = {"splitter": ["best","random"],
                      "max_depth": [4,6,8],
                      }

rf_hyperparameters = {"n_estimators":[10,50,100],
                      "max_depth":[4,6,8],
                      }

log_hyperparameters = {"penalty":["l2","none"]}

# hyperparameters_combination = create_hyperparameter_grid(hyperparameters)
# best_model = custom_tune_regression_model_hyperparameters(SGDRegressor,X,y,hyperparameters_combination)

# best_model, performance_metrics ,best_hyperparameter_combination, best_RMSE = tune_regression_model_hyperparameters(SGDRegressor,hyperparameters)
# save_model_folder = "models/regression/linear_regression"
# save_model(best_model,performance_metrics,best_hyperparameter_combination,save_model_folder)

if __name__ == "__main__":
    X,y_regression = load_airbnb(df,"Price_Night")
    print(type(X), type(y_regression))
    model_list = [GradientBoostingRegressor, DecisionTreeRegressor, RandomForestRegressor]
    hyperparameter_list = [gbr_hyperparameters, dt_hyperparameters,rf_hyperparameters]
    evaluate_all_models(model_list,X,y_regression, hyperparameter_list)
    best_reg_model, best_reg_model_hyperparameters, best_reg_model_performance_metrics = find_best_model()
    
    
    X, y_classification = load_airbnb(df,"Category")
    best_clas_model, best_clas_model_hyperparameters, best_clas_model_performance_metrics = tune_classification_model_hyperparameters(LogisticRegression,X,y_classification,log_hyperparameters)
    
    
    