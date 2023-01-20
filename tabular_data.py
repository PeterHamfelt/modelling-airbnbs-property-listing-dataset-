import pandas as pd
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
        
    """
    
    dataframe.dropna(subset = column_name,inplace = True)
    
    return dataframe
    

if __name__ == "__main__":

    working_dir = os.path.dirname(__file__)
    data_path = "data/tabular_data/listing.csv"
    path = os.path.join(working_dir, data_path)
    listing_df = pd.read_csv(path)
    column_names = listing_df.columns
    print(column_names)
    rating_columns = column_names[10:16]
    listing_df = remove_rows_with_missing_ratings(listing_df,rating_columns)
    print(type(listing_df))

