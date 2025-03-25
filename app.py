import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os

cwd = os.getcwd()
Model = joblib.load(os.path.join(cwd, "model.joblib"))
make_enc = joblib.load("make_enc.joblib")
model_enc = joblib.load(os.path.join(cwd, "model_enc.joblib"))
scaler = joblib.load(os.path.join(cwd, "scaler.joblib"))

def encoder(df):
    df['make'] = make_enc.transform(df['make'])
    df["model"] = model_enc.transform(df["model"])
    return df

st.set_page_config(page_title="Car Price Prediction 🚗💰", page_icon="🚗", layout="wide")

st.markdown(
    "<h1 style='text-align: center; color: #3498db;'>🚗 Car Price Prediction App</h1>",
    unsafe_allow_html=True,
)
st.markdown(
    "<p style='text-align: center; color: #2c3e50; font-size: 18px;'>Enter car details from the side bar to get an estimated selling price.</p>",
    unsafe_allow_html=True,
)
st.markdown("---")

st.sidebar.header("🔍 Enter Car Details")
year = st.sidebar.number_input("📅 Year", min_value=1990, max_value=2025, step=1)
km_driven = st.sidebar.number_input("🚗 Distance Driven (km)", min_value=0, step=1000)
mileage = st.sidebar.number_input("⛽ Mileage (kmpl)", min_value=0.0, step=0.1, format="%.2f")
engine = st.sidebar.number_input("⚙ Engine CC", min_value=0, step=10)
max_power = st.sidebar.number_input("🔥 Max Power (BHP/HP)", min_value=0.0, step=0.1, format="%.2f")
age = st.sidebar.number_input("📆 Car Age (Years)", min_value=0, step=1)

make = st.sidebar.selectbox(
    "🏭 Company Name",
    ['MARUTI', 'HYUNDAI', 'HONDA', 'MAHINDRA', 'TOYOTA', 'TATA', 'FORD',
     'VOLKSWAGEN', 'RENAULT', 'MERCEDES-BENZ', 'BMW', 'SKODA', 'CHEVROLET',
     'AUDI', 'NISSAN', 'DATSUN', 'FIAT', 'JAGUAR', 'LAND', 'VOLVO', 'JEEP',
     'MITSUBISHI', 'KIA', 'PORSCHE', 'MINI', 'MG', 'ISUZU', 'LEXUS', 'FORCE',
     'BENTLEY', 'AMBASSADOR', 'OPELCORSA', 'DAEWOO', 'PREMIER', 'MASERATI',
     'DC', 'LAMBORGHINI', 'FERRARI', 'MERCEDES-AMG', 'ROLLS-ROYCE', 'OPEL'],
    index=None,  
    placeholder="Select a car brand"
)

model = st.sidebar.text_input("🚘 Model Name", placeholder="Enter model name")
individual = st.sidebar.selectbox('👤 Individual Seller?', ['Yes', 'No'])
trustmark_dealer = st.sidebar.selectbox('🏢 Trustmark Dealer?', ['Yes', 'No'])
fuel_type = st.sidebar.selectbox('⛽ Fuel Type', ['Diesel', 'Electric', 'LPG', 'Petrol'])
gearbox_type = st.sidebar.selectbox('⚙ Transmission Type', ['Manual', 'Automatic'])
greater_5 = st.sidebar.selectbox('🔧 Has More Than 5 Gears?', ['Yes', 'No'])

st.sidebar.markdown("---")
submitted = st.sidebar.button("🚀 Predict Price")

if submitted:
    if not make or not model:
        st.error("❌ Please enter both the company name and model name.")
    else:
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

        dataframe = pd.DataFrame([data])

        encoded_data = encoder(dataframe)

        scaled_data = scaler.transform(encoded_data)

        scaled_price = Model.predict(scaled_data)[0]

        actual_price = scaled_price * 100000  

        st.markdown("---")
        st.markdown(
            f"<h2 style='text-align: center; color: #27ae60;'>💰 Estimated Price: ₹ {actual_price:,.2f}</h2>",
            unsafe_allow_html=True,
        )

