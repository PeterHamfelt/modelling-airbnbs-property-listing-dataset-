import numpy as np
import pandas as pd
import os
import joblib
import json
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from tabular_data import load_airbnb

script_dir = os.path.dirname(os.path.realpath(__file__))
df = pd.read_csv(os.path.join(script_dir,"data/tabular_data/clean_tabular_data.csv"))
X, y = load_airbnb(df,"Category")

x_train, x_test, y_train, y_test = model_selection.train_test_split(X,y, test_size=0.3)
x_test, x_val, y_test, y_val = model_selection.train_test_split(x_test,y_test,test_size=0.5)

log_reg = LogisticRegression()
log_reg.fit(x_train,y_train)

y_pred_label = log_reg.predict(x_test)
y_pred_percentage = log_reg.predict_proba(x_test)

print(y_pred_label)
print(y_pred_percentage)
