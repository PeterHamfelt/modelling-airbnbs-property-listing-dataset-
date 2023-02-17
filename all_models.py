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

def save_model(model,metrics_dict, nn_config=None, hyperparameter_dict = None):
    
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
        
        model_name = str(type(model())).split(".")[-1]
        
        if 'regressor' in model_name.lower():
            upper_save_folder = os.path.join(working_dir,"models/regression",{}.format(type(model).__name__))
        else: 
            upper_save_folder = os.path.join(working_dir,"models/classification",{}.format(type(model).__name__))            
            
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
    
    
global working_dir

working_dir = os.path.dirname(os.path.realpath(__file__))

if __name__ == "__main__":
    os.chdir(working_dir)
    data = AirbnbNightlyPriceImageDataset()
    best_model, best_model_performance_metrics, best_model_hyperparameters =find_best_nn(data,4)
    print(f"The best model is achieved using the following hyperparameters {best_model_hyperparameters} and its performance metrics are as follow {best_model_performance_metrics}")

    