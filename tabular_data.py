import pandas as pd
import numpy as np
import os
import ast

def remove_rows_with_missing_ratings(df,column_name):
    """Remove rows with missing ratings

    Remove rows from the given dataframe that contains missing values in the specified
    columns.

    Args:
        df (pandas.DataFrame): The dataframe to remove the rows from
        column_name (list): List of column names to check for missing values. 

    Returns:
        pandas.DataFrame: The dataframe with the rows that contains missing values in 
        specified columns removed. 
    """
    
    df.dropna(subset = column_name,inplace = True)
    
    return df

def combine_description_strings(df):
    
    df.dropna(subset = ["Description"], inplace = True)
    df["Description"] = df["Description"].apply(lambda x: x.replace("About this space"," "))
    
    return df
            

def set_default_feature_values(df):
    """Set a default value for features

    Replace missing and string values in the specified feature columns of the dataframe 
    with the default value of 1 and at the same time change the data type of the columns
    to float. 

    Args:
        df (pandas.DataFrame): The dataframe which contains the columns which requires
        modification.

    Returns:
        pandas.DataFrame: The dataframe which has been modified. 
    """
    
    col_names = ["guests","beds","bathrooms","bedrooms"]
    
    for column in col_names:
        
        if df[column].dtype == object:
            df[column] = df[column].replace(r"[a-zA-Z]+",1,regex = True)
        
        df[column] = df[column].replace(np.nan,1)
        
        df[column] = df[column].astype(float)
        
        print(sorted(list(df[column].unique())))
            
    return df


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
    listing_df = combine_description_strings(listing_df)

    print(listing_df["Description"])
