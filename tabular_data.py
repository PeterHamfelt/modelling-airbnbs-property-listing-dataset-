import pandas as pd
import numpy as np
import os
import ast

def remove_rows_with_missing_ratings(df):
    """Remove rows with missing ratings

    Remove rows from the given dataframe that contains missing values in the specified
    columns.

    Args:
        df (pandas.DataFrame): The dataframe to remove the rows from

    Returns:
        pandas.DataFrame: The dataframe with the rows that contains missing values in 
        specified columns removed. 
    """
    
    rating_column_names = ["Cleanliness_rating","Accuracy_rating",
                           "Communication_rating","Location_rating",
                           "Check-in_rating","Value_rating"]
    df.dropna(subset = rating_column_names,inplace = True)
    
    return df

def combine_description_strings(df):
    """Combine description list into string

    Firstly remove "About this space" from the list of items before combining the remaining 
    items in the list into a singular string.    

    Args:
        df (pandas.DataFrame): The dataframe which requires modification.

    Returns:
        pandas.DataFrame: The modified dataframe
    """
    
    df.dropna(subset = ["Description"], inplace = True)
    df["Description"] = df["Description"].apply(lambda x: x.replace("About this space",""))
    df["Description"] = df["Description"].apply(lambda x: " ".join(ast.literal_eval(x)) if x.startswith("[") and x.endswith("]") else x)
    df["Description"] = df["Description"].apply(lambda x: " ".join(x.split()))
    
    return df
            

def set_default_feature_values(df):
    """Set a default value for features

    Replace missing and string values in the specified feature columns of the dataframe 
    with the default value of 1 and at the same time change the data type of the columns
    to float. 

    Args:
        df (pandas.DataFrame): The dataframe which requires modification.

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

def clean_tabular_data(df):
    """Clean tabular data

    Cleans the tabular data by calling the three other functions defined above.

    Args:
        df (pandas.DataFrame): The dataframe which contains dirty data

    Returns:
        pandas.DataFrame: Processed and clean dataframe
    """
    
    df = remove_rows_with_missing_ratings(df)
    df = combine_description_strings(df)
    df = set_default_feature_values(df)
    
    df.drop(df.columns[-1], axis = 1, inplace = True)
    
    return df

def load_airbnb(df, label):
    """Split clean data 

    Split the clean data into features and label dataframe. The features dataframe
    will only contain numerical data that will be used to predict the label. 

    Args:
        df (pandas.DataFrame): The dataframe which contains clean data.
        label (str): The column name which the model will be predicting

    Returns:
       tuple : A tuple which contains the features and labels.
    """
    
    labels = df[label]
    features = df.drop(label, axis = 1)
    
    column_list = list(features.columns.values)
    
    for column in column_list:
        
        if features[column].dtype != "float64":
            features.drop(column, axis = 1, inplace = True)
    
    if label == "bedrooms":
        features["Category"] = df["Category"].astype("category").cat.codes
        
    return (features,labels)
    

if __name__ == "__main__":

    working_dir = os.path.dirname(__file__)
    data_path = "data/tabular_data/listing.csv"
    path = os.path.join(working_dir, data_path)
    listing_df = pd.read_csv(path, na_values=np.nan)
    listing_df = clean_tabular_data(listing_df)
    save_path = os.path.join(working_dir, "data/tabular_data/clean_tabular_data.csv")
    
    listing_df.to_csv(save_path, index = False)
    
    features, labels = load_airbnb(listing_df,"Price_Night")
    
    print(features.head(10))
    print(labels.head(10))
