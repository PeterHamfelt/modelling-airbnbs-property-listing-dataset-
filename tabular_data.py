import pandas as pd
import numpy as np
import os

def remove_rows_with_missing_ratings(dataframe,column_name):
    """Remove rows with missing ratings

    Remove rows from the given dataframe that contains missing values in the specified
    columns.

    Args:
        dataframe (pandas.DataFrame): The dataframe to remove the rows from
        column_name (list): List of column names to check for missing values. 

    Returns:
        pandas.DataFrame: The dataframe with the rows that contains missing values in 
        specified columns removed. 
    """
    
    dataframe.dropna(subset = column_name,inplace = True)
    
    return dataframe

def combine_description_strings(dataframe):
    pass

def set_default_feature_values(dataframe):
    col_names = ["guests","beds","bathrooms","bedrooms"]
    
    for column in col_names:
        
        if dataframe[column].dtype == object:
            dataframe[column] = dataframe[column].replace(r"[a-zA-Z]+",1,regex = True)
        
        
        dataframe[column] = dataframe[column].replace(np.nan,1)
        
        dataframe[column] = dataframe[column].astype(float)
        
        print(sorted(list(dataframe[column].unique())))
            
        
    return dataframe


if __name__ == "__main__":

    working_dir = os.path.dirname(__file__)
    data_path = "data/tabular_data/listing.csv"
    path = os.path.join(working_dir, data_path)
    listing_df = pd.read_csv(path, na_values=np.nan)
    column_names = listing_df.columns
    print(column_names)
    rating_columns = column_names[10:16]
    listing_df = remove_rows_with_missing_ratings(listing_df,rating_columns)
    listing_df = set_default_feature_values(listing_df)



