import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os

cwd = os.getcwd()

# Load trained model and encoders
Model = joblib.load(os.path.join(cwd, "model.joblib"))
make_enc = joblib.load(os.path.join(cwd, "make_enc.joblib"))
model_enc = joblib.load(os.path.join(cwd, "model_enc.joblib"))
scaler = joblib.load(os.path.join(cwd, "scaler.joblib"))

# Encoding function
def encoder(df):
    df['make'] = make_enc.transform(df['make'])
    df["model"] = model_enc.transform(df["model"])
    return df

# Streamlit UI
st.title("🏦 Car Price Prediction")
st.write("Enter car details below:")

with st.form("car_price_form"):
    year = st.number_input("Year", min_value=1990, max_value=2025, step=1)
    km_driven = st.number_input("Distance Driven (km)", min_value=0, step=1000)
    mileage = st.number_input("Mileage (kmpl)", min_value=0.0, step=0.1, format="%.2f")
    engine = st.number_input("Engine CC", min_value=0, step=10)
    max_power = st.number_input("Max Power (BHP/HP)", min_value=0.0, step=0.1, format="%.2f")
    age = st.number_input("Car Age (Years)", min_value=0, step=1)

    make = st.selectbox(
        "Company Name",
        ['MARUTI', 'HYUNDAI', 'HONDA', 'MAHINDRA', 'TOYOTA', 'TATA', 'FORD',
         'VOLKSWAGEN', 'RENAULT', 'MERCEDES-BENZ', 'BMW', 'SKODA', 'CHEVROLET',
         'AUDI', 'NISSAN', 'DATSUN', 'FIAT', 'JAGUAR', 'LAND', 'VOLVO', 'JEEP',
         'MITSUBISHI', 'KIA', 'PORSCHE', 'MINI', 'MG', 'ISUZU', 'LEXUS', 'FORCE',
         'BENTLEY', 'AMBASSADOR', 'OPELCORSA', 'DAEWOO', 'PREMIER', 'MASERATI',
         'DC', 'LAMBORGHINI', 'FERRARI', 'MERCEDES-AMG', 'ROLLS-ROYCE', 'OPEL'],
        index=None,  # No default selection
        placeholder="Select a car brand"
    )

    model = st.text_input("Model Name", placeholder="Enter model name")

    individual = st.selectbox('Individual Seller?', ['Yes', 'No'])
    trustmark_dealer = st.selectbox('Trustmark Dealer?', ['Yes', 'No'])
    fuel_type = st.selectbox('Fuel Type', ['Diesel', 'Electric', 'LPG', 'Petrol'])
    gearbox_type = st.selectbox('Transmission Type', ['Manual', 'Automatic'])
    greater_5 = st.selectbox('Has More Than 5 Gears?', ['Yes', 'No'])

    submitted = st.form_submit_button("Predict Price")

if submitted:
    # Validate input fields
    if not make or not model:
        st.error("Please enter both the company name and model name.")
    else:
        # Prepare input data
        data = {
            "year": year,
            "km_driven": km_driven,
            "mileage": mileage,
            "engine": engine,
            "max_power": max_power,
            "age": age,
            "make": make,
            "model": model,
            "Individual": 1 if individual == 'Yes' else 0,
            "Trustmark Dealer": 1 if trustmark_dealer == 'Yes' else 0,
            "Diesel": 1 if fuel_type == 'Diesel' else 0,
            "Electric": 1 if fuel_type == 'Electric' else 0,
            "LPG": 1 if fuel_type == 'LPG' else 0,
            "Petrol": 1 if fuel_type == 'Petrol' else 0,
            "Manual": 1 if gearbox_type == 'Manual' else 0,
            "5": 1 if greater_5 == 'Yes' else 0,
            ">5": 1 if greater_5 == 'No' else 0,
        }
        
        # Convert to DataFrame
        dataframe = pd.DataFrame([data])

        # Encode categorical features
        encoded_data = encoder(dataframe)

        # Scale the data
        scaled_data = scaler.transform(encoded_data)

        # Make prediction
        scaled_price = Model.predict(scaled_data)[0]

        # Convert from Lakhs to INR
        actual_price = scaled_price * 100000  

        st.success(f'Predicted Price: ₹ {actual_price:,.2f}')



    