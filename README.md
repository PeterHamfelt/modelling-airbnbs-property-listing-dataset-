# Modelling-Airbnbs-property-listings
Airbnb is a widely used online marketplace that allows propety owners to offer their accommodations to travellers from around the globe. With millions of listings avaiable on the platform, setting competitive prices for listings is a key factor for hosts attract bookings while maximising revenue. In this repository, we developed a framework to systematically train, tune, and evaluate different machine learning models ability to accurately predict the price per night of Airbnb property listings using various property-related features. 

## Dependencies
- Python 3.x
- pandas
- numpy
- Sklearn
- pyTorch
- pyYAML
- joblib
- TensorBoard

## How to use
To evaluate the framework, follow the following steps:

1) Clone this repository onto your local machine.

    ```bash
    git clone hhttps://github.com/TyW-98/modelling-airbnbs-property-listing-dataset-
    cd modelling-airbnbs-property-listing-dataset
    ```

2) Install the necessary dependencies using the provided configuration file. Run the following commands:

    ```console
    conda create --name env_name python=3.8
    conda activate env_name
    pip install -r requirements.txt
    ```

3) Run the `all_model` script which contains of all the machine learning model used in this framework which includes:

    **<u>For predicting price per night:</u>**
    - SGDRegressor
    - DecisionTreeRegression
    - RandomForestRegressor
    - LinearRegression
    - GradientBoostingRegressor
    - Neural Network

    **<u>For predicting property's category:</u>**
    - LogisticRegression
    - GradientBoostingClassifier
    - RandomForestClassifier
    - DecisionTreeClassifier

    ```console
    python all_models.py
    ```
## Contributing
Contributions to this framework are welcome. Feel free to open a pull request or submit an issue if you have any suggestions or improvements to make. 

## Milestone 1 (Repository Setup)
- Establish a Github repository to version control project files. 

## Mileston 2 (Understand the framework)
- Review the overview of the framework from the video provided.

## Milestone 3 (Data Preparation)
- The Airbnb listing dataset requires preprocessing steps to be carried out before it can be used for machine learning purposes. 
- One of the key steps is to handle missing values in the dataset as this could negatively impact the performance of the machine learning models. 
- The remove_rows_with_missing_ratings function plays a critical role in handling missing data in the Airbnb listing dataset. Specifically, this function removes any rows from the dataset that contain missing values in the columns related to ratings for cleanliness, accuracy, communication, location, check-in, and value. By removing these rows, the function helps to ensure that the dataset is consistent and accurate for subsequent analysis and modeling tasks.

    ```python
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
    ```
- The set_default_feature_values function is a significant data preprocessing step that addresses missing values and replaces string values in specific columns of the Airbnb listing dataset. This function sets a default value of 1 for any missing values in the specified columns and converts the data type of the column to float. This ensures that the data is clean and uniform, making it easier to work with during machine learning modeling.

    ```python
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
    ```

- The combine_description_strings function is used to clean and modify the Description column in the Airbnb listing dataset. It removes the string "About this space" from the list of items in the column before combining the remaining items in the list into a single string. The function then applies several lambda functions to clean up the text further, including removing any remaining brackets, joining the list into a string, and removing any extra whitespace. Finally, the modified dataframe is returned.

    ```python
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
    ```

- A wrapper function `clean_tabular_data` is used to call the three functions described above and drops the last column of the dataset which is the image URL column before returning the processed dataset. 

    ```python
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
    ```
  
- Lastly the `load_airbnb` function splits the clean dataset into a features and label dataframe. The features dataframe contains only numerical data that will be used to predict the label. The function first assigns the label column to the labels variable and then drops the label column from the features dataframe. It then iterates over all columns in the features dataframe and drops any non-numerical column. If the label is 'bedrooms', it converts the 'Category' column in the features dataframe to categorical using the astype method and assigns the codes to the 'Category' column. The function returns a tuple containing the features and labels dataframes.
    ```python
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
    ```
