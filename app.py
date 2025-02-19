
import pickle
import streamlit as st
import numpy as np
import pandas as pd

# Load your model file
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

st.title('Nigeria House Price Predictor App')

# Add input widgets for user inputs
house_type = st.selectbox(
    "title",
    ['Detached Duplex', 'Terraced Duplexes', 'Semi Detached Duplex',
       'Detached Bungalow', 'Block of Flats', 'Semi Detached Bungalow',
       'Terraced Bungalow']
)
bedrooms = st.slider("bedrooms", min_value=1, max_value=9, value=3)
parking_space = st.slider("parking_space", min_value=1, max_value=9, value=2)

# When the 'Predict' button is clicked
if st.button("Predict"):
    # Prepare the input data as a DataFrame (since pipelines often expect a DataFrame)
    input_data = pd.DataFrame({
        'House_type': [house_type],
        'Bedrooms': [bedrooms],
        'Parking_space': [parking_space]
    })
    prediction = model.predict(input_data)[0].round(2)
    st.write(f'The predicted value is: {prediction} Naira')
