#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import r2_score
import pickle
import seaborn as sns
import matplotlib.pyplot as plt

# -----------------------------------
# 1. Data Loading and Initial Exploration
# -----------------------------------
print("Step 1: Loading and initial exploration...")
data = pd.read_csv('laptop_data.csv')
df = pd.DataFrame(data)

# Dropping the 'Unnamed: 0' column which is an index column
df = df.drop(['Unnamed: 0'], axis=1)

# -----------------------------------
# 2. Data Cleaning and Preprocessing
# -----------------------------------
print("Step 2: Cleaning and preprocessing data...")
# Removing 'GB' and 'kg' units and converting to numeric types
df['Ram'] = df['Ram'].str.replace("GB", "").astype('int32')
df['Weight'] = df['Weight'].str.replace("kg", "").astype('float32')

# -----------------------------------
# 3. Feature Engineering
# -----------------------------------
print("Step 3: Feature Engineering...")

# CPU Name Engineering
# Create a simplified CPU category based on the first few words
def get_simplified_cpu(name):
    if "Intel Core i5" in name:
        return "Intel Core i5"
    elif "Intel Core i7" in name:
        return "Intel Core i7"
    elif "Intel Core i3" in name:
        return "Intel Core i3"
    elif "AMD Ryzen" in name:
        return "AMD Ryzen"
    elif "Intel Pentium Quad" in name:
        return "Intel Pentium Quad"
    else:
        return "Other" # Consolidate less common CPUs
        
df['cpu'] = df['Cpu'].apply(get_simplified_cpu)
df.drop(columns=['Cpu'], inplace=True)

# Screen Resolution Engineering
df['y_res'] = df['ScreenResolution'].apply(lambda x: x.split()[-1])
df["x_res"] = df["y_res"].apply(lambda x: x.split("x")[0]).astype('int')
df["y_res"] = df["y_res"].apply(lambda x: x.split("x")[-1]).astype('int')
df['pixel_count'] = df['x_res'] * df['y_res']
df.drop(columns=['x_res', 'y_res'], inplace=True)

# Touchscreen Feature
df['touchscreen'] = df['ScreenResolution'].apply(lambda x: 1 if "Touchscreen" in x else 0)
df.drop(columns=['ScreenResolution'], inplace=True)

# OS Consolidation
def opsys(name):
    if name in ['Windows 10', 'Windows 7', 'Windows 10 S']:
        return 'Windows'
    elif name in ["macOS", 'Mac OS X']:
        return 'Mac'
    else:
        return 'others'
df["os"] = df['OpSys'].apply(opsys)
df.drop(columns=['OpSys'], inplace=True)

# GPU Brand Extraction
df["GPU"] = df['Gpu'].apply(lambda x: x.split(" ")[0])
df.drop(columns=['Gpu'], inplace=True)

# Memory Type Extraction
df['Memory_type'] = df['Memory'].apply(lambda x: x.split(" ")[-1])
df['Memory_type'] = df['Memory_type'].str.replace('Storage', 'SSD')
df.drop(columns=['Memory'], inplace=True)

# Additional Engineered Features
df['Ram_weight'] = df['Ram'] * df['Weight']
df['Inches_weight'] = df['Inches'] * df['Weight']

# Log transformation on Price and pixel_count for better model performance
df['Price'] = np.log(df['Price'])
df['pixel'] = np.log(df['pixel_count'])

# Finalizing the features to be used for the model
X = df.drop(columns=['Price', 'Ram', 'Weight', 'Inches', 'pixel_count'])
y = df['Price']

# -----------------------------------
# 4. Model Training
# -----------------------------------
print("Step 4: Training the model...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=2)

# Defining the preprocessing steps with ColumnTransformer and Pipeline
categorical_features = ['Company', 'TypeName', 'os', 'GPU', 'Memory_type', 'cpu']
numerical_features = ['touchscreen', 'Ram_weight', 'Inches_weight', 'pixel']

step1 = ColumnTransformer(transformers=[
    ('col_tnf', OneHotEncoder(handle_unknown='ignore', sparse_output=False, drop='first'), categorical_features)
], remainder='passthrough')

step2 = RandomForestRegressor(n_estimators=100, random_state=42)

pipe = Pipeline([
    ('step1', step1),
    ('step2', step2)
])

pipe.fit(X_train, y_train)

y_pred = pipe.predict(X_test)
score = r2_score(y_test, y_pred)
print(f"R-squared score on test data: {score}")

# -----------------------------------
# 5. Saving the Trained Model
# -----------------------------------
print("Step 5: Saving the trained model...")
with open('laptop_price_model.pkl', 'wb') as f:
    pickle.dump(pipe, f)
print("Model saved successfully as 'laptop_price_model.pkl'.")
print("-" * 30)
print("Training script finished.")

import streamlit as st
import numpy as np
import pandas as pd
import pickle

# --- Custom CSS for a clean and professional look ---
st.markdown("""
<style>
/* Streamlit app container with a clean background color */
[data-testid="stAppViewContainer"] {
    background-color: #f0f2f6;
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
}

/* Custom font and text color for a modern feel */
h1, h2, h3, h4, h5, h6 {
    color: #2c3e50;
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    font-weight: 700;
}

/* Semi-transparent sidebar for readability */
[data-testid="stSidebar"] {
    background-color: rgba(255, 255, 255, 0.9);
    border-radius: 10px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.1);
}

/* Styling for the prediction output box */
.prediction-box {
    background: #2c3e50; /* A sleek, dark background for the result */
    color: #ffffff;
    padding: 30px;
    border: 3px solid #3498db; /* A blue border for a modern touch */
    border-radius: 12px;
    font-size: 2.25rem;
    font-weight: bold;
    text-align: center;
    margin-top: 30px;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2);
}

/* General styling for widgets to improve readability */
.st-bf, .st-br, .st-bu, .st-bv, .st-bw, .st-bx, .st-by, .st-bz {
    color: #333;
}
</style>
""", unsafe_allow_html=True)


# --- Load the pre-trained model ---
try:
    with open('laptop_price_model.pkl', 'rb') as f:
        pipe = pickle.load(f)
except FileNotFoundError:
    st.error("Error: The model file 'laptop_price_model.pkl' was not found. Please ensure it's in the same directory.")
    st.stop()


# --- Helper function to extract a simplified CPU identifier ---
def extract_cpu(cpu_full):
    """Maps a user-selected CPU string to a simplified category."""
    if "i5" in cpu_full.lower():
        return "Intel Core i5"
    elif "i7" in cpu_full.lower():
        return "Intel Core i7"
    elif "i3" in cpu_full.lower():
        return "Intel Core i3"
    elif "quad" in cpu_full.lower():
        return "Intel Pentium Quad"
    elif "ryzen" in cpu_full.lower():
        return "AMD Ryzen"
    else:
        return "Other"


# --- Streamlit App UI ---
st.title("Laptop Price Predictor")
st.subheader("Select the features of the laptop to get a price prediction.")
st.markdown("---")

# --- User Input Features ---
st.sidebar.header("Laptop Features")

company_options = ['Apple', 'HP', 'Lenovo', 'Acer', 'Asus', 'others', 'Toshiba']
company = st.sidebar.selectbox("Brand", company_options)

type_name_options = ['Ultrabook', 'Notebook', '2 in 1 Convertible']
type_name = st.sidebar.selectbox("Type", type_name_options)

os_options = ['Windows', 'Mac', 'others']
os_value = st.sidebar.selectbox("OS", os_options)

gpu_options = ['Intel', 'AMD', 'Nvidia']
gpu = st.sidebar.selectbox("GPU Brand", gpu_options)

memory_type_options = ['SSD', 'HDD']
memory_type = st.sidebar.selectbox("Memory Type", memory_type_options)

cpu_options = ['Intel Core i5', 'Intel Core i7', 'Intel Core i3', 'AMD Ryzen', 'Intel Pentium Quad']
cpu_full = st.sidebar.selectbox("CPU", cpu_options)

touchscreen = st.sidebar.checkbox("Touchscreen")
ram = st.sidebar.slider("RAM (GB)", min_value=4, max_value=32, value=8, step=4)
inches = st.sidebar.number_input("Screen Size (Inches)", min_value=10.0, max_value=20.0, value=15.0, step=0.1)
weight = st.sidebar.number_input("Weight (kg)", min_value=0.5, max_value=5.0, value=2.0, step=0.1)
pixel_count = st.sidebar.number_input("Total Pixels", min_value=1000000, max_value=10000000, value=2073600, step=100000)

st.markdown("---")


# --- Prediction Logic ---
if st.sidebar.button("Predict Price"):
    try:
        # Create a DataFrame from the user's input.
        input_data = {
            'Company': [company],
            'TypeName': [type_name],
            'touchscreen': [int(touchscreen)],
            'os': [os_value],
            'GPU': [gpu],
            'Memory_type': [memory_type],
            'Ram_weight': [ram * weight],
            'Inches_weight': [inches * weight],
            'cpu': [extract_cpu(cpu_full)],
            'pixel': [np.log(pixel_count)] # Apply the log transformation
        }
        
        # Create a DataFrame from the input data
        input_df = pd.DataFrame(input_data)
        
        # Make the prediction using the loaded pipeline
        prediction_log = pipe.predict(input_df)
        
        # Reverse the log transformation to get the actual price
        predicted_price = np.exp(prediction_log)

        st.subheader("Predicted Laptop Price")
        st.markdown(f'<div class="prediction-box">${predicted_price[0]:,.2f}</div>', unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Prediction failed. An error occurred: {e}")
