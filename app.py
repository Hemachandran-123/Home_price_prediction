import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json

json_file = '/content/columns (1).json'
pickle_file = '/content/banglore_home_prices_model (1).pickle'

with open(json_file, 'r') as f:
    data_columns = json.load(f)['data_columns']
locations = data_columns[3:]

with open(pickle_file, 'rb') as f:
    model = pickle.load(f)

def predict_price(location, sqft, bath, bhk):
    loc_index = data_columns.index(location.lower())
    x = np.zeros(len(data_columns))
    x[0] = sqft
    x[1] = bath
    x[2] = bhk
    if loc_index >= 0:
        x[loc_index] = 1
    return model.predict([x])[0]

def main():
    st.title("Bangalore Home Price Prediction")

    st.write("Enter the details to predict the home price in Bangalore:")

    location = st.selectbox("Location", locations)
    sqft = st.number_input("Total Square Feet", min_value=300, max_value=10000, step=1)
    bath = st.number_input("Number of Bathrooms", min_value=1, max_value=10, step=1)
    bhk = st.number_input("Number of Bedrooms (BHK)", min_value=1, max_value=10, step=1)

    if st.button("Predict"):
        price = predict_price(location, sqft, bath, bhk)
        st.success(f"The predicted price is ₹ {price:.2f} Lakhs")

if __name__ == "__main__":
    main()
