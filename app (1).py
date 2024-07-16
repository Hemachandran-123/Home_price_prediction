import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
import os

# Define the paths for the JSON and pickle files
json_file = 'columns.json'  # Update if needed
pickle_file = 'banglore_home_prices_model.pickle'  # Update if needed

@st.cache
def load_data_columns(json_file):
    if os.path.exists(json_file):
        with open(json_file, 'r') as f:
            data_columns = json.load(f)['data_columns']
        return data_columns
    else:
        st.error("JSON file not found!")
        return []

@st.cache
def load_model(pickle_file):
    if os.path.exists(pickle_file):
        with open(pickle_file, 'rb') as f:
            model = pickle.load(f)
        return model
    else:
        st.error("Pickle file not found!")
        return None

data_columns = load_data_columns(json_file)
locations = data_columns[3:] if data_columns else []

model = load_model(pickle_file)

def predict_price(location, sqft, bath, bhk):
    try:
        loc_index = data_columns.index(location.lower())
    except ValueError:
        loc_index = -1

    x = np.zeros(len(data_columns))
    x[0] = sqft
    x[1] = bath
    x[2] = bhk
    if loc_index >= 0:
        x[loc_index] = 1

    return model.predict([x])[0] if model else None

def main():
    st.title("Bangalore Home Price Prediction")

    st.write("Enter the details to predict the home price in Bangalore:")

    if data_columns and locations:
        location = st.selectbox("Location", locations)
        sqft = st.number_input("Total Square Feet", min_value=300, max_value=10000, step=1)
        bath = st.number_input("Number of Bathrooms", min_value=1, max_value=10, step=1)
        bhk = st.number_input("Number of Bedrooms (BHK)", min_value=1, max_value=10, step=1)

        if st.button("Predict"):
            if model:
                price = predict_price(location, sqft, bath, bhk)
                if price is not None:
                    st.success(f"The predicted price is ₹ {price:.2f} Lakhs")
                else:
                    st.error("Prediction failed!")
            else:
                st.error("Model not loaded!")
    else:
        st.error("Data columns not loaded!")

if __name__ == "__main__":
    main()
