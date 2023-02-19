import os
import torch 
import pandas as pd
import numpy as np
import yaml
import datetime
import time
import json
import joblib
import itertools
from tabular_data import load_airbnb
from sklearn import model_selection
from sklearn.linear_model import SGDRegressor, LogisticRegression
from sklearn.preprocessing import normalize
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score, f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, RandomForestClassifier, GradientBoostingClassifier
from torch.utils.tensorboard import SummaryWriter

# SKlearn's Machine learning regression and classification models
def model_hyperparameter_tuner(model_type,X,y,hyperparameter_dict):
    
    x_train, x_test, y_train, y_test = model_selection.train_test_split(X,y, test_size=0.3, random_state=42)
    x_test, x_val, y_test, y_val = model_selection.train_test_split(x_test, y_test, test_size=0.5, random_state=42)
    
    model = model_type(random_state = 42)
    
    gs = GridSearchCV(model,hyperparameter_dict)
    gs.fit(x_train,y_train)
    
    performance_metrics = {}

    modes = ["Training","Testing","Validation"]
    x_data = [x_train,x_test,x_val]
    Criteron_2 = None
    
    for mode, data in zip(modes,x_data):
        y_pred = gs.predict(data)
        
        if mode.lower() == "training":
            target = y_train
        elif mode.lower() == "testing":
            target = y_test
        elif mode.lower() == "validation":
            target = y_val
    
        if "regressor" in model.__class__.__name__.lower():
            
            # Calculate RMSE value
            Criteron_1 = "RMSE"
            metric_1 = np.sqrt(mean_squared_error(target,y_pred)) 
        
        elif "classifier" in model.__class__.__name__.lower() or model.__class__.__name__.lower() == "logisticregression":
            
            Criteron_1 = "Accuracy"
            metric_1 = accuracy_score(target,y_pred)
            
            Criteron_2 = "F1 Scre"
            metric_2 = f1_score(target,y_pred,average = "weighted")
            
        performance_metrics[f"Model {Criteron_1}"] = {}
        performance_metrics[f"Model {Criteron_1}"][f"{mode} {Criteron_1}"] = metric_1
        
        if Criteron_2 != None:
            performance_metrics[f"Model {Criteron_2}"] = {}
            performance_metrics[f"Model {Criteron_2}"][f"{mode} {Criteron_2}"] = metric_2
            
    return gs.best_estimator_, performance_metrics, gs.best_params_
            

def evaluate_all_models(model_list,X,y,hyperparameter_list):
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
    #  TODO: Add evaluate_all_models and add find_best_model for SKlearn models.
    
    for model_type, hyperparameter_dict in zip(model_list,hyperparameter_list):
        
        model, performance_metrics,model_params = model_hyperparameter_tuner(model_type,X,y,hyperparameter_dict)
        
        save_model(model,metrics_dict=performance_metrics, model_type= model_type, hyperparameter_dict=model_params) 
            

# PyTorch linear Regression Neural Network
class AirbnbNightlyPriceImageDataset(torch.utils.data.Dataset):
    
    def __init__(self):
        super().__init__()
        clean_data_df = pd.read_csv(os.path.join(working_dir,"data/tabular_data/clean_tabular_data.csv"))
        load_df = load_airbnb(clean_data_df,"Price_Night")
        self.data = pd.concat([load_df[0],load_df[1]], axis = 1)
        self.x = pd.DataFrame(normalize(load_df[0]))
        self.y = load_df[1]
        
    def __getitem__(self,index):
        feature = torch.tensor(self.x.iloc[index])
        label = torch.tensor(self.y.iloc[index])
        
        return (feature, label)
    
    def __len__(self):
        
        return len(self.y)
    
    def data_splitter(self):
        validation_size = 0.3
        test_size = 0.2
        np.random.seed(42)
        dataset_size = len(self.y)
        sample_indices_1 = list(range(dataset_size))
        split_loc_1 = int(np.floor(dataset_size*(1-validation_size)))
        np.random.shuffle(sample_indices_1)
        
        training_sample_indices , validation_n_testing_indices = sample_indices_1[:split_loc_1], sample_indices_1[split_loc_1:]
    
        split_loc_2 = int(np.floor(len(validation_n_testing_indices)*(1-test_size)))
        
        validation_sample_indices,  testing_sample_indices = validation_n_testing_indices[:split_loc_2], validation_n_testing_indices[split_loc_2:]
        
        training_sampler = torch.utils.data.SubsetRandomSampler(training_sample_indices)
        testing_sampler = torch.utils.data.SubsetRandomSampler(testing_sample_indices)
        validation_sampler = torch.utils.data.SubsetRandomSampler(validation_sample_indices)
        
        return training_sampler, testing_sampler, validation_sampler
    
class LinearRegression(torch.nn.Module):
    
    def __init__(self,data,nn_config):
        super(LinearRegression,self).__init__()
        n_input = data.x.shape[1]
        n_output = data.y.ndim
        layers = []
        n_hidden_layers = nn_config["depth"] + 2
        n_neurons = nn_config["hidden_layer_width"]
        for hidden_layer in range(n_hidden_layers):
            if hidden_layer == 0: # Input layer
                layers.append(torch.nn.Linear(n_input,n_neurons))
            elif hidden_layer == range(n_hidden_layers)[-1]: # Output layer
                layers.append(torch.nn.ReLU())
                layers.append(torch.nn.Linear(n_neurons,n_output))
            else: # Hidden Layers
                layers.append(torch.nn.ReLU())
                layers.append(torch.nn.Linear(n_neurons,n_neurons))
                
        self.linear = torch.nn.Sequential(*layers)
        
    def forward(self,features):
        prediction = self.linear(features)
        return prediction
    
def training(model,train_loader,test_loader,validation_loader,n_epochs=10,optimiser_type="SGD",learning_rate = 0.001):
    
    writer = SummaryWriter()
    
    init_optimiser = getattr(torch.optim,optimiser_type)
    optimiser = init_optimiser(model.parameters(),lr = learning_rate)
    
    total_loss = 0
    total_r_square = 0
    
    performance_metrics = {"RMSE_loss":{}, "R_squared":{}}
    
    start_time = time.time()
    
    for epochs in range(n_epochs):
        
        for train_batch in train_loader:
            
            feature, label = train_batch
            
            feature, label = feature.to(torch.float32), label.to(torch.float32)
            
            train_prediction = model(feature)
            
            loss= torch.nn.functional.mse_loss(train_prediction.squeeze(),label)
            total_loss += loss.item()

            r_squared = r2_score(label,train_prediction.detach().numpy())
            total_r_square += r_squared
            
            # Backpropagate the train_loss into hidden layers
            loss.backward()
            
            optimiser.step()
            
            # Reset the gradient before computing the next loss fir every req_grad = True parameters. 
            optimiser.zero_grad()
            
        avg_training_loss = np.mean(total_loss)
        training_RMSE = np.sqrt(avg_training_loss)
        print(training_RMSE)
        avg_training_r_squared = np.mean(r_squared)
        print(avg_training_r_squared)
        writer.add_scalar("Average Training Loss",training_RMSE,epochs)
        writer.add_scalar("Aveage R Squared",avg_training_r_squared,epochs)
        
        testing_RMSE, avg_testing_r_squared = eval_model(model, test_loader)
        validation_RMSE, avg_validation_r_squared = eval_model(model, validation_loader)
        
            
    end_time = time.time()
    training_duration = end_time - start_time    
    inference_latency = training_duration / ((len(train_loader)*8)*n_epochs)
    losses = [training_RMSE, testing_RMSE, validation_RMSE]
    r_squared_scores = [avg_training_r_squared, avg_testing_r_squared, avg_validation_r_squared]
    modes = ["Training","Testing","Validation"]
    
    for mode, loss_value, r_score in zip(modes,losses,r_squared_scores):
        performance_metrics["RMSE_loss"][mode] = loss_value
        performance_metrics["R_squared"][mode] = r_score
        
    performance_metrics["training_duration"] = round(end_time - start_time,3)
    performance_metrics["inference_latency"] = inference_latency
    
    print(performance_metrics)
    
    return model, performance_metrics
    
def eval_model(model,dataloader):
    loss = 0 
    r_squared = 0
    
    for batch in dataloader:
        feature, label = batch
        feature, label = feature.to(torch.float32), label.to(torch.float32)
        
        y_pred = model(feature)
        
        loss += torch.nn.functional.mse_loss(y_pred.squeeze(),label).item()
        r_squared += r2_score(label,y_pred.detach().numpy())
        
    avg_loss = np.mean(loss)
    avg_r2 = np.mean(r_squared)
    RMSE = np.sqrt(avg_loss)
    
    return RMSE, avg_r2
          
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

def save_model(model,metrics_dict, nn_config=None, model_type = None, hyperparameter_dict = None):
    
    current_time = datetime.datetime.now().replace(microsecond=0).isoformat().replace(":","-")
    
    if isinstance(model,torch.nn.Module):
        
        if 'regression' in model.__class__.__name__.lower():
            save_path = os.path.join(working_dir,"models","regression","neural_networks",f"{model.__class__.__name__} ({current_time})")
            
            if os.path.exists(save_path) == False:
                os.makedirs(save_path)
                
            torch.save(model.state_dict(),os.path.join(save_path,"model.pt"))
            
            with open(os.path.join(save_path,"performance_metrics.json"),"w+") as f:
                json.dump(metrics_dict,f)
                
            with open(os.path.join(save_path,"hyperparanmeter.json"),"w+") as f:
                json.dump(nn_config,f)    

    elif hyperparameter_dict != None:  
        
        model_name = type(model_type()).__name__.lower()
        
        if 'regressor' in model_name.lower():
            upper_save_folder = os.path.join(working_dir,"models/regression",model_name)
        else: 
            upper_save_folder = os.path.join(working_dir,"models/classification",model_name)           
            
        if os.path.exists(upper_save_folder) == False:
            os.mkdir(upper_save_folder)
            
        joblib.dump(model,os.path.join(upper_save_folder,"model.joblib"))
        
        with open(os.path.join(upper_save_folder,"hyperparameters.json"),"w+") as jfile:
            json.dump(hyperparameter_dict, jfile)
            
        with open(os.path.join(upper_save_folder,"metrics.json"),"w+") as jfile:
            json.dump(metrics_dict, jfile)

def find_best_nn(data,n_model=1):
    
    train_sampler, testing_sampler, validation_sampler = data.data_splitter()
    train_loader = torch.utils.data.DataLoader(data,batch_size = 8, sampler = train_sampler)
    test_loader = torch.utils.data.DataLoader(data,batch_size = 8, sampler = testing_sampler)
    validation_loader = torch.utils.data.DataLoader(data,batch_size = 8, sampler = validation_sampler)
    
    min_validation_loss = np.inf
    
    for _ in range(n_model):
        
        generate_nn_configs()
        
        nn_config = get_nn_config()
        optimiser_type = nn_config["optimiser"]
        learning_rate = nn_config["Learning_rate"]

        model = LinearRegression(data,nn_config)
        
        model,performance_metrics = training(model,train_loader,test_loader,validation_loader,25,optimiser_type,learning_rate)
        
        save_model(model,metrics_dict=performance_metrics,nn_config=nn_config)
        
        if performance_metrics["RMSE_loss"]["Validation"] < min_validation_loss:
            best_model = model
            best_model_performance_metrics = performance_metrics
            best_model_hyperparameters = nn_config
       
    return best_model, best_model_performance_metrics, best_model_hyperparameters

def generate_nn_configs():
    
    nn_config_path = os.path.join(working_dir,"nn_config.yaml")
    
    nn_config = {
        'optimiser':"SGD",
        'Learning_rate':np.random.uniform(0.0001,0.01),
        'hidden_layer_width':np.random.randint(12,20),
        'depth':np.random.randint(1,4)
    }
    
    with open(nn_config_path,"w") as config_file:
        yaml.dump(nn_config,config_file)

if __name__ == "__main__":
    global working_dir
    working_dir = os.path.dirname(os.path.realpath(__file__))
    os.chdir(working_dir)
    
    # SKlearn Regression Machine learning section 
    label_column_name = "Price_Night"
    df = pd.read_csv(os.path.join(working_dir,"data/tabular_data/clean_tabular_data.csv"))
    X, y_regression = load_airbnb(df,label_column_name)
    
    sgd_hyperarameters = {"penalty": ["l1","l2","elasticnet"],
                      "alpha": [0.01, 0.001, 0.0001]}
    
    regression_model_list = [SGDRegressor]
    
    regression_model_hyperparameter_list = [sgd_hyperarameters]
    
    evaluate_all_models(regression_model_list,X,y_regression,regression_model_hyperparameter_list)
    
    # Initialise airbnb property torch datatset and train n number of models to determine which is the best performing model.
    # It will then return the best performin model, its performance metrics and hyperparameters. 
    data = AirbnbNightlyPriceImageDataset()
    n_models = 4
    best_model, best_model_performance_metrics, best_model_hyperparameters =find_best_nn(data,n_models)
    print(f"The best model is achieved using the following hyperparameters {best_model_hyperparameters} and its performance metrics are as follow {best_model_performance_metrics}")

    