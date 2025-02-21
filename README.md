# Predicting-Nigeria-House-prices

This documentation outlines the development and evaluation of a Ridge regression model to predict Nigeria house prices using real-life data. The model is evaluated using Mean Absolute Error (MAE) as the primary performance metric.


# Problem Statements:

•	The goal is to create a model that can predict house prices in Nigeria using machine learning (Ridge_Regression_Model)<br>
•	Real estate agents, buyers, and investors needs have access to market-driven pricing that can assist them in making informed decisions.<br>
•	The model takes values of property features such as numbers of bedrooms, bathrooms, parking space and various house type to predict apartment prices in Nigeria.<br>


# Data Description
This dataset contains Houses listings in Nigeria and their prices based on Location and other parameters such as:<br>
•	bedrooms: number of bedrooms in the houses<br>
•	bathrooms: number of bathrooms in the houses<br>
•	toilets: number of toilets<br>
•	parking space<br>
•	title: house type<br>


# Model Training

**Import Statements and Dataset**
To begin with, the necessary libraries and modules are imported, and the dataset is loaded into the model.<br>
import pandas as pd<br>
import seaborn as sns<br>
import matplotlib.pyplot as plt<br>
import numpy as np<br>
from sklearn.model_selection import train_test_split<br>
from sklearn.metrics import mean_absolute_error<br>
from sklearn.linear_model import LinearRegression, Ridge<br>
from sklearn.impute import SimpleImputer<br>
from sklearn.pipeline import make_pipeline<br>
from category_encoders import OneHotEncoder<br>
from sklearn.utils.validation import check_is_fitted<br>
from ipywidgets import Dropdown, FloatSlider, IntSlider, interact<br>
import streamlit as st<br>
import pickle<br>

# Data Splitting
The independent variables (X) for the model are: "Bedrooms", “Parking Space”, “Title” columns, while the dependent variable (y) was the "Price<br>
X (features): Independent Variables<br>
 y (target): House Price<br>

  The data is split into training and test sets, with an 80%/20% ratio for training and testing, respectively:<br>
X = data["Bedrooms", “Parking Space”, “Title”]<br>
y = data['Price']<br>
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)<br>

# Model Training
model = make_pipeline(OneHotEncoder(), SimpleImputer(), Ridge())<br>

model.fit(X_train, y_train)<br>

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
def make_prediction(bedrooms, parking_space, house_type):<br>
    data = {"bedrooms":bedrooms,<br>
            "parking_space":parking_space,<br>
            "title":house_type<br>
    }<br>
    df = pd.DataFrame(data, index=[0])<br>
    prediction = model.predict(df).round(2)[0]<br>
    return f"Predicted apartment price: ₦{prediction}"<br>

# Slider Widget
interact(<br>
    make_prediction,<br>
    bedrooms=IntSlider(<br>
        min=X_train["bedrooms"].min(),<br>
        max=X_train["bedrooms"].max(),<br>
        value=X_train["bedrooms"].mean(),<br>
    ),<br>
    parking_space=IntSlider(<br>
        min=X_train["parking_space"].min(),<br>
        max=X_train["parking_space"].max(),<br>
        step=1,<br>
        value=X_train["parking_space"].mean(),<br>
    ),<br>
    house_type=Dropdown(options=sorted(X_train["title"].unique())),<br>
);<br>

# Conclusion
The test MAE of 62,422,800 indicates that the model’s predictions are closer to the actual values than the baseline model (77,032,009),demonstrating that this model can<br> be effectively used to predict house prices in Nigeria.<br>

However, further improvements could be made by:<br>
1.	Incorporate Additional Features – Including economic indicators (e.g., inflation rate, exchange rate) and location-specific attributes could enhance<br>
   prediction accuracy.<br>
2.	Regular Model Updates – The real estate market changes over time, so periodic retraining of the model with updated data will improve accuracy.<br>


# Future Work
•	Expanding the feature set to include more economic indicators.<br>
•	Exploring Other Regression Models may yield better predictive performance.
