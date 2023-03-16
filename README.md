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