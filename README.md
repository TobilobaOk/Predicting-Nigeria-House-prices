# Predicting-Nigeria-House-prices

This documentation outlines the development and evaluation of a Ridge regression model to predict Nigeria house prices using real-life data. The model is evaluated using Mean Absolute Error (MAE) as the primary performance metric.


#Problem Statements:

•	The goal is to create a model that can predict house prices in Nigeria using machine learning (Ridge_Regression_Model)
•	Real estate agents, buyers, and investors needs have access to market-driven pricing that can assist them in making informed decisions.
•	The model takes values of property features such as numbers of bedrooms, bathrooms, parking space and various type of houses to predict apartment prices in Nigeria.


# Data Description
This dataset contains Houses listings in Nigeria and their prices based on Location and other parameters such as:<br>
•	bedrooms: number of bedrooms in the houses
•	bathrooms: number of bathrooms in the houses
•	toilets: number of toilets 
•	parking space
•	title: house type


# Model Training

**Import Statements and Dataset**
To begin with, the necessary libraries and modules are imported, and the dataset is loaded into the model.<br>
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from category_encoders import OneHotEncoder
from sklearn.utils.validation import check_is_fitted
from ipywidgets import Dropdown, FloatSlider, IntSlider, interact
import streamlit as st
import pickle

# Data Splitting
The independent variables (X) for the model are: "Bedrooms", “Parking Space”, “Title” columns, while the dependent variable (y) was the "Price
	X (features): Independent Variables<br>
  y (target): House Price<br>

  The data is split into training and test sets, with an 80%/20% ratio for training and testing, respectively:<br>
X = data["Bedrooms", “Parking Space”, “Title”]<br>
y = data['Price']<br>
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model Training
model = make_pipeline(OneHotEncoder(), SimpleImputer(), Ridge())

model.fit(X_train, y_train)

# Model Evaluation

After training the model, predictions are made on both the training and test sets. The performance is evaluated using the Mean Absolute Error (MAE) metric, which measures the average magnitude of errors in a set of predictions. A lower MAE indicates better model accuracy.<br>

•	Baseline Model MAE: 77,032,009  (This is the performance of a simple guess, like predicting the average value of the target variable for all data points.)<br>

•	Model (Train) MAE: 62,352,238 (The MAE on the training data, showing how well the model fits the training set.)<br>

•	Model (Test) MAE: 62,422,800 (The MAE on the test data, showing how well the model generalizes to unseen data.)<br>

The test MAE of 62,422,800 demonstrates that the model’s predictions are closer to the train MAE compared to the baseline model (MAE of 77,032,009 ), indicating an improvement in predictive accuracy.

# Calculate MAE for training and testing sets

train_mae = mean_absolute_error(y_train, train_predictions)<br>
test_mae = mean_absolute_error(y_test, test_predictions)<br>

# Make predictions on train and test sets

train_predictions = model.predict(X_train.values.shape)<br>
test_predictions = model.predict(X_test.values.shape)<br>

# make_prediction Function
def make_prediction(bedrooms, parking_space, house_type):
    data = {"bedrooms":bedrooms,
            "parking_space":parking_space,
            "title":house_type
    }
    df = pd.DataFrame(data, index=[0])
    prediction = model.predict(df).round(2)[0]
    return f"Predicted apartment price: ₦{prediction}"

# Slider Widget
interact(
    make_prediction,
    bedrooms=IntSlider(
        min=X_train["bedrooms"].min(),
        max=X_train["bedrooms"].max(),
        value=X_train["bedrooms"].mean(),
    ),
    parking_space=IntSlider(
        min=X_train["parking_space"].min(),
        max=X_train["parking_space"].max(),
        step=1,
        value=X_train["parking_space"].mean(),
    ),
    house_type=Dropdown(options=sorted(X_train["title"].unique())),
);

# Conclusion
The test MAE of 62,422,800 indicates that the model’s predictions are closer to the actual values than the baseline model (77,032,009),demonstrating that this model can be effectively used to predict house prices in Nigeria.

However, further improvements can be made by:
