import os
import torch 
import pandas as pd
import numpy as np
from tabular_data import load_airbnb

class AirbnbNightlyPriceImageDataset(torch.utils.data.Dataset):
    def __init__(self):
        super().__init__()
        clean_data_df = pd.read_csv(os.path.join(os.path.dirname(os.path.realpath(__file__)),"data/tabular_data/clean_tabular_data.csv"))
        load_df = load_airbnb(clean_data_df,"Price_Night")
        self.data = pd.concat([load_df[0],load_df[1]], axis = 1)
        self.x = load_df[0]
        self.y = load_df[1]
        
    def __getitem__(self,index):
        feature = torch.tensor(self.x.iloc[index])
        label = torch.tensor(self.y.iloc[index])
        
        return (feature, label)
    
    def __len__(self):
        
        return len(self.y)

data = AirbnbNightlyPriceImageDataset()

train_loader = torch.utils.data.DataLoader(data, batch_size = 32, shuffle = True)

for batch in train_loader:
    feature_cols , label_cols = batch
    print(feature_cols,label_cols)