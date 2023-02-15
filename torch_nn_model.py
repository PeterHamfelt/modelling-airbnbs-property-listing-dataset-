import os
import torch 
import pandas as pd
import numpy as np
import yaml
from sklearn.preprocessing import normalize
from tabular_data import load_airbnb
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
    
    def train_val_sampler(self):
        validation_size = 0.2
        np.random.seed(42)
        dataset_size = len(self.y)
        sample_indices = list(range(dataset_size))
        split_loc = int(np.floor(dataset_size*validation_size))
        np.random.shuffle(sample_indices)
        
        print(split_loc)
        training_sample_indices , validation_sample_indices = sample_indices[:split_loc], sample_indices[split_loc:]
        
        training_sampler = torch.utils.data.SubsetRandomSampler(training_sample_indices)
        validation_sampler = torch.utils.data.SubsetRandomSampler(validation_sample_indices)
        
        return training_sampler, validation_sampler
    
class LinearRegression(torch.nn.Module):
    
    def __init__(self,data,nn_config):
        super().__init__()
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
    
def training(model,train_loader,validation_loader,n_epochs=10):
    
    writer = SummaryWriter()
    
    optimizer  = torch.optim.SGD(model.parameters(), lr = 0.001)
    
    for epochs in range(n_epochs):
        
        for train_batch, val_batch in zip(train_loader, validation_loader):
            
            train_feature, train_label = train_batch
            val_feature, val_label = val_batch
            
            train_feature, val_feature = train_feature.to(torch.float32), val_feature.to(torch.float32)
            train_label, val_label = train_label.to(torch.float32), val_label.to(torch.float32)
            
            train_prediction = model(train_feature)
            validation_prediciton = model(val_feature)
            
            train_loss = torch.nn.functional.mse_loss(train_prediction,train_label)
            validation_loss = torch.nn.functional.mse_loss(validation_prediciton,val_label)
            
            # Backpropagate the train_loss into hidden layers
            train_loss.backward()
            print(train_loss, validation_loss)
            optimizer.step()
            
            writer.add_scalar("Training loss",train_loss.item(),epochs)
            writer.add_scalar("Validation loss", validation_loss.item(),epochs)
            
            # Reset the gradient before computing the next loss fir every req_grad = True parameters. 
            optimizer.zero_grad()
            
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

if __name__ == "__main__":
    os.chdir(working_dir)
    data = AirbnbNightlyPriceImageDataset()
    train_sampler, validation_sampler = data.train_val_sampler()

    train_loader = torch.utils.data.DataLoader(data,batch_size = 8, sampler = train_sampler)
    validation_loader = torch.utils.data.DataLoader(data,batch_size = 8, sampler = validation_sampler)
    
    nn_config = get_nn_config()

    model = LinearRegression(data,nn_config)

    training(model,train_loader,validation_loader,10)