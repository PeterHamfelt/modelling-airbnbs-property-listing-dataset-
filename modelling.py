import numpy as np
import pandas as pd
import os
import itertools
import matplotlib.pyplot as plt
import joblib
import json
import shutil
import glob
import yaml
import torch
from sklearn import model_selection
from sklearn.linear_model import SGDRegressor, LogisticRegression
from sklearn.metrics import mean_squared_error,r2_score, accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, ConfusionMatrixDisplay
from tabular_data import load_airbnb
from sklearn.preprocessing import normalize
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, RandomForestClassifier, GradientBoostingClassifier
from torch.utils.tensorboard import SummaryWriter

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
        X (pandas.DataFrame): The feature columns which will be used to predict the labels.
        y (pandas.Series): The sample's target the model is predicting for. 
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
        model_type (list): Classification models to be tried. 
        X (pandas.DataFrame): The feature columns which will be used to predict the labels.
        y (pandas.Series): The categorical column the model is trying to predict using the features. 
        hyperparameter_dict (dict): The dictionary which consist of the hyperparameters and values associated to each model type
        which will be tried during the grid search. 

    Returns:
        _type_: Best performing model from obtained from grid search
        dict: A dictionary of the best combination of hyperparameters and its values
        dict: The models training, testing and validation accuracy values. 
    """
    
    x_train, x_test, y_train, y_test = model_selection.train_test_split(X,y, test_size=0.3, random_state=42)
    x_test, x_val, y_test, y_val = model_selection.train_test_split(x_test,y_test, test_size=0.5, random_state=42)
    
    model = model_type(random_state= 42)
    
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
    
    print(val_acc, val_f1)
    
    return gs.best_estimator_, performance_metrics, gs.best_params_, gs.best_score_
    
        
def evaluate_all_models(model_list,X,y,hyperparameter_list,reg_or_class):
    """Evaluate different models

    Evaluate the performance of a list of different regression models by tunning their hyperparameters and comparing them
    to each other and the base linear regression model. At the same time create a folder for each of the models to save the
    model, its performance metrics and the hyperparameter combination used to achieved that result. 

    Args:
        model_list (list): A list of the sklearn model to be evaluated. 
        X (pandas.DataFrame): The feature columns which will be used to predict the labels.
        y (pandas.Series): The labels the model is predicting.
        hyperparameter_list (list): A list of hyperparameter dictionary corresponding to each model. 
        reg_or_class (str): To evaluate regression models, use reg_or_class = "reg" and for classification models, use
        reg_or_class = "class".
    """
    
    if reg_or_class.lower() == "reg":
        save_upper_path = "models/regression"
        hyperparameter_tuner = tune_regression_model_hyperparameters
    elif reg_or_class.lower() == "class":
        save_upper_path = "models/classification"
        hyperparameter_tuner = tune_classification_model_hyperparameters
    
    for model_type, hyperparameter_dict in zip(model_list,hyperparameter_list):
        
        model, performance_metrics, model_params, model_best_score = hyperparameter_tuner(model_type,X,y,hyperparameter_dict)
        
        save_path = "{}/{}".format(save_upper_path ,type(model).__name__)
        full_save_path = os.path.join(working_dir,save_path)
        
        save_model(model,performance_metrics,model_params,full_save_path)
        
def find_best_model(reg_or_class:str):
    """Find the best performing model

    This function finds the best performing model from all the saved models in the model/regression folder by firstly loading
    in and appending all the model's validation RMSE value into a list. From the list, the position of the lowest RMSE value can 
    be obtained and used to index the best model's path from the list of model's directory. From the best model's path, the best 
    model and its associated hyperparameters and performance metrics then can be load in. 
    
    Args:
        reg_or_class (str) = Find which type of model, the best model for classification or regression. reg_or_class = "reg" for 
        regression model and reg_or_class = "class" for classification model. 
    
    Returns:
        _type_: The best sklearn model
        dict : The hyperparameters associated with the best model.
        dict : The performance metrics assocaited with the best model.  
    """
    if reg_or_class.lower() == "reg":
        model_type = "regression"
    elif reg_or_class.lower() == "class":
        model_type = "classification"
        
    model_folder = os.path.join(working_dir,f"models/{model_type}")
    different_models_folder = [folder[0] for folder in os.walk(model_folder)][1:]
    
    model_performance_metric_list = []
    
    for model_files in different_models_folder:
        
        performance_metrics_path = os.path.join(model_files,"metrics.json")
        
        with open(performance_metrics_path) as file:
            performance_metric_dict = json.load(file)
            
        
        if reg_or_class.lower() == "reg":    
            model_performance_metric_list.append(performance_metric_dict["Validation RMSE"])
        elif reg_or_class.lower() == "class":
            model_performance_metric_list.append(performance_metric_dict["Model F1 Score"]["Validation F1 Score"])
    
    if reg_or_class.lower() == "reg":
        performance_metric = min(model_performance_metric_list)
    elif reg_or_class.lower() == "class":
        performance_metric = max(model_performance_metric_list)
            
    best_model_folder_index = model_performance_metric_list.index(performance_metric)
    model_path = os.path.join(different_models_folder[best_model_folder_index],"model.joblib")
    hyperparameter_path = os.path.join(different_models_folder[best_model_folder_index],"hyperparameters.json")
    performance_metrics_path = os.path.join(different_models_folder[best_model_folder_index],"metrics.json")
    
    with open(model_path, "rb") as file:
        best_model = joblib.load(file)
    
    with open(hyperparameter_path) as file:
        best_model_hyperparameters = json.load(file)
        
    with open(performance_metrics_path) as file:
        best_model_performance_metrics = json.load(file)
    
    return best_model, best_model_hyperparameters, best_model_performance_metrics

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
    save_dir = os.path.join(working_dir,folder)
    model_saved_path = os.path.join(save_dir,"model.joblib")
    hyperparameter_json = os.path.join(save_dir,"hyperparameters.json")
    performance_metrics_json = os.path.join(save_dir,"metrics.json")
    
    if os.path.exists(save_dir) == False:
        os.mkdir(save_dir)
        
    joblib.dump(model,model_saved_path)
    
    with open(hyperparameter_json,"w+") as file:
        json.dump(hyperparameter_combination,file)
    
    with open(performance_metrics_json,"w+") as file:
        json.dump(performance_metrics,file)
        
def get_nn_config():
    """ Get Neural Network Configuration

    This function loads in the configuration of the neural network from nn_config.yaml file and returns a dictionary of 
    the parameters and values in a dictionary.

    Returns:
        dict: Neural netowkr's hyperparameters and the value associated with each hyperparameters. 
    """
    
    nn_yaml_file = os.path.join(working_dir,"nn_config.yaml")
    
    with open(nn_yaml_file,"r") as yaml_file:
        nn_config_dict = yaml.safe_load(yaml_file)
        
    return nn_config_dict

global working_dir

working_dir = os.path.dirname(os.path.realpath(__file__))
df = pd.read_csv(os.path.join(working_dir,"data/tabular_data/clean_tabular_data.csv"))

# Regression model hyperparameter dictionaries
sgd_hyperarameters = {"penalty": ["l1","l2","elasticnet"],
                      "alpha": [0.01, 0.001, 0.0001]}

gbr_regression_hyperparameters = {"learning_rate": [0.1,0.01,0.001], 
                   "subsample": [1.0,0.1,0.01], 
                   "n_estimators":[10,50,100],
                   "max_depth": [4,6,8]}

dt_regression_hyperparameters = {"splitter": ["best","random"],
                      "max_depth": [4,6,8],
                      }

rf_regression_hyperparameters = {"n_estimators":[10,50,100],
                      "max_depth":[4,6,8],
                      }

regression_model_list = [SGDRegressor,
                         GradientBoostingRegressor, 
                         DecisionTreeRegressor, 
                         RandomForestRegressor
                         ]
regression_hyperparameter_list = [sgd_hyperarameters, 
                                  gbr_regression_hyperparameters, 
                                  dt_regression_hyperparameters,
                                  rf_regression_hyperparameters
                                  ]

# Classification model hyperparameter dictionaries
log_hyperparameters = {"penalty":["l2","none"],
                       "solver":["lbfgs","newton-cg","sag","saga"]}

dt_classification_hyperparameters = {"splitter":["best","random"],
                                     "min_samples_split":[2,10,50,100],
                                     "max_depth":[2,4,7,10],
                                     "min_samples_leaf":[2,10,50,100],
                                     "max_features":[None,"auto","sqrt","log2"]
                                     }

rf_classification_hyperparameters = {"n_estimators":[10,50,100],
                                     "max_depth":[2,4,7,10],
                                     "min_samples_split":[2,10,50,100],
                                     "min_samples_leaf":[2,10,50,100],
                                     "max_features":[None,"auto","sqrt","log2"]
                                     }

gbr_classification_hyperparameters = {"learning_rate": [0.1, 0.01, 0.001, 0.0001],
                                      "n_estimators": [10,50,75,100],
                                      "min_samples_leaf": [2,10,50,100],
                                      "max_depth": [2,4,7,10],
                                      "max_features": [None,"auto","sqrt","log2"]}

classification_model_list = [LogisticRegression, 
                             DecisionTreeClassifier, 
                             RandomForestClassifier,
                             GradientBoostingClassifier
                             ]

classification_hyperparameter_list =[log_hyperparameters, 
                                     dt_classification_hyperparameters, 
                                     rf_classification_hyperparameters,
                                     gbr_classification_hyperparameters
                                     ]

if __name__ == "__main__":
    # Regression section
    print("Regression Models")
    reg_var = "reg"
    X,y_regression = load_airbnb(df,"Price_Night")
    evaluate_all_models(regression_model_list,X,y_regression, regression_hyperparameter_list,reg_var)
    best_reg_model, best_reg_model_hyperparameters, best_reg_model_performance_metrics = find_best_model(reg_var)
    
    print("\n")
    
    # Classification section
    print("Classification Models")
    class_var = "class"
    X, y_classification = load_airbnb(df,"Category")
    evaluate_all_models(classification_model_list,X,y_classification,classification_hyperparameter_list,class_var)
    best_class_model, best_class_model_hyperparameters, best_class_model_performance_metrics = find_best_model(class_var)
    
    print("The best regression model type is {} with the following hyperparameters {} and the model's performance metrics are {}".format(type(best_reg_model).__name__, best_reg_model_hyperparameters, best_reg_model_performance_metrics))
    
    print("\n")
    
    print("The best classification model type is {} with the following hyperparameters {} and the model's performance metrics are {}".format(type(best_class_model).__name__, best_class_model_hyperparameters, best_class_model_performance_metrics))
    
    nn_config_dict = get_nn_config()
