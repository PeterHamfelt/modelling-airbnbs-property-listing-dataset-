import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error,r2_score
from tabular_data import load_airbnb

def plot_prediction(y_pred,y_true):
    y_pred = y_pred
    y_true = y_true
    samples = len(y_pred)
    plt.figure()
    plt.scatter(np.arange(samples), y_pred, c='r', label='predictions')
    plt.scatter(np.arange(samples), y_true, c='b', label='true labels', marker='x')
    plt.legend()
    plt.xlabel('Sample numbers')
    plt.ylabel('Values')
    plt.show()   
    
def custom_tune_regression_model_hyperparameters():
    pass

script_dir = os.path.dirname(os.path.realpath(__file__))

df = pd.read_csv(os.path.join(script_dir,"data/tabular_data/clean_tabular_data.csv"))

X,y = load_airbnb(df,"Price_Night")

x_train, x_test, y_train, y_test = model_selection.train_test_split(X,y ,test_size = 0.15)

model = SGDRegressor(learning_rate= "adaptive")
model.fit(x_train,y_train)

y_pred = model.predict(x_test)

RMSE = mean_squared_error(y_test,y_pred,squared=False)
r2 = r2_score(y_test,y_pred)
print(f"The model RMSE is {RMSE} and r2 score is {r2}")

graph_1 = plot_prediction(y_pred,y_test)
