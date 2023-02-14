import os
import torch 
import pandas as pd
import numpy as np
from sklearn.preprocessing import normalize
from tabular_data import load_airbnb
from torch.utils.tensorboard import SummaryWriter

class AirbnbNightlyPriceImageDataset(torch.utils.data.Dataset):
    
    def __init__(self):
        super().__init__()
        clean_data_df = pd.read_csv(os.path.join(os.path.dirname(os.path.realpath(__file__)),"data/tabular_data/clean_tabular_data.csv"))
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
    
class LinearRegression(torch.nn.Module):
    
    def __init__(self,input_size,output_size):
        super().__init__()
        # Number of nodes in respective hidden layers
        hidden_layer_1 = 16
        hidden_layer_2 = 10
        # Initialise neural network layers
        self.linear = torch.nn.Sequential(
            # Input Layer to hidden layer 1
            torch.nn.Linear(input_size,hidden_layer_1),
            # Hidden layer 1 activation function
            torch.nn.ReLU(),
            # Hidden layer 1 to hidden layer 2
            torch.nn.Linear(hidden_layer_1,hidden_layer_2),
            # Hidden layer 2 activation function
            torch.nn.ReLU(),
            # Hidden layer 2 to output layer
            torch.nn.Linear(hidden_layer_2,output_size)
        )
        
    def forward(self,features):
        prediction = self.linear(features)
        return prediction
    
def training(model,train_loader,n_epochs=10):
    
    writer = SummaryWriter()
    
    optimizer  = torch.optim.SGD(model.parameters(), lr = 0.001)
    
    for epochs in range(n_epochs):
        
        for batch in train_loader:
            feature, label = batch
            feature = feature.to(torch.float32)
            label = label.to(torch.float32)
            prediction = model(feature)
            loss = torch.nn.functional.mse_loss(prediction,label)
            
            # Backpropagate the loss into hidden layers
            loss.backward()
            print(loss)
            optimizer.step()
            
            writer.add_scalar("loss",loss.item(),epochs)
            
            # Reset the gradient before computing the next loss fir every req_grad = True parameters. 
            optimizer.zero_grad()
            

if __name__ == "__main__":
    working_dir = os.path.dirname(os.path.realpath(__file__))
    os.chdir(working_dir)
    data = AirbnbNightlyPriceImageDataset()
    train_loader = torch.utils.data.DataLoader(data, batch_size = 32, shuffle = True)

    for batch in train_loader:
        feature_cols , label_cols = batch
        print(feature_cols,label_cols)
    
    features = feature_cols.to(torch.float32)
    labels = label_cols.to(torch.float32)
    input_shape = len((feature_cols[1]))
    output_shape = 1  
    model = LinearRegression(input_size = input_shape,output_size = output_shape)

    training(model,train_loader,10)