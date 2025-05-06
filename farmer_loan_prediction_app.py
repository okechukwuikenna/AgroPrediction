import pandas as pd
import pickle
import streamlit as st

# Load the pre-trained model
model = pickle.load(open('loan_repayment_model.pkl', 'rb'))

# Load the dataset (Ensure the dataset exists and matches your model input features)
df = pd.read_csv('farmer_data.csv')  # Make sure 'farmer_data.csv' exists in your working directory

# If you don't have a CSV file and want to define the data manually (like we did earlier):
# Define your dummy data
data = {
    'Farmer_ID': [1, 2, 3, 4, 5],
    'Age': [27, 24, 29, 22, 30],
    'Education': ['Secondary', 'None', 'Tertiary', 'Primary', 'Secondary'],
    'Experience_Years': [3, 1, 5, 2, 7],
    'Crop_Type': ['Maize', 'Tomato', 'Soybean', 'Onion', 'Wheat'],
    'Region': ['West Zone', 'East Zone', 'North Zone', 'South Zone', 'West Zone'],
    'Land_Size_Acres': [3.5, 1.2, 5.0, 2.0, 6.0],
    'Annual_Revenue_USD': [6000, 2000, 10000, 2500, 12000],
    'Loan_Amount_USD': [3000, 1000, 5000, 1500, 6000],
    'Loan_Approved': ['Yes', 'No', 'Yes', 'Yes', 'Yes'],
    'Repaid_On_Time': ['Yes', 'No', 'Yes', 'No', 'Yes'],
    'Past_Defaults': [0, 1, 0, 1, 0],
    'Weather_Risk_Index': [0.2, 0.6, 0.1, 0.5, 0.3],
    'Market_Stability_Score': [0.8, 0.4, 0.9, 0.6, 0.7]
}

# Convert to a pandas DataFrame
df = pd.DataFrame(data)

# Preprocess the data (same preprocessing steps as used when training the model)
df['Education'] = df['Education'].map({'None': 0, 'Primary': 1, 'Secondary': 2, 'Tertiary': 3})
df['Crop_Type'] = df['Crop_Type'].map({'Maize': 0, 'Tomato': 1, 'Soybean': 2, 'Onion': 3, 'Wheat': 4})
df['Region'] = df['Region'].map({'West Zone': 0, 'East Zone': 1, 'North Zone': 2, 'South Zone': 3})
df['Loan_Approved'] = df['Loan_Approved'].map({'Yes': 1, 'No': 0})
df['Repaid_On_Time'] = df['Repaid_On_Time'].map({'Yes': 1, 'No': 0})

# Define features and target
X = df.drop(['Farmer_ID', 'Repaid_On_Time'], axis=1)
y = df['Repaid_On_Time']

# If you're running this as a web app, you'll use Streamlit inputs to get new farmer data
# Input fields for the farmer data in Streamlit

st.title("Farmer Loan Repayment Prediction")

# Input fields for new farmer data
age = st.number_input("Age", min_value=18, max_value=100, value=26)
education = st.selectbox("Education Level", ['None', 'Primary', 'Secondary', 'Tertiary'])
experience_years = st.number_input("Years of Experience", min_value=0, max_value=50, value=3)
crop_type = st.selectbox("Crop Type", ['Maize', 'Tomato', 'Soybean', 'Onion', 'Wheat'])
region = st.selectbox("Region", ['West Zone', 'East Zone', 'North Zone', 'South Zone'])
land_size = st.number_input("Land Size (Acres)", min_value=0.1, max_value=100.0, value=2.5)
annual_revenue = st.number_input("Annual Revenue (USD)", min_value=0, max_value=100000, value=5500)
loan_amount = st.number_input("Loan Amount (USD)", min_value=0, max_value=50000, value=2500)
loan_approved = st.selectbox("Loan Approved", ['Yes', 'No'])
past_defaults = st.selectbox("Past Defaults", [0, 1])
weather_risk = st.slider("Weather Risk Index", 0.0, 1.0, 0.2)
market_stability = st.slider("Market Stability Score", 0.0, 1.0, 0.75)

# Create DataFrame for the new farmer's input
new_farmer = pd.DataFrame([{
    'Age': age,
    'Education': education,
    'Experience_Years': experience_years,
    'Crop_Type': crop_type,
    'Region': region,
    'Land_Size_Acres': land_size,
    'Annual_Revenue_USD': annual_revenue,
    'Loan_Amount_USD': loan_amount,
    'Loan_Approved': loan_approved,
    'Past_Defaults': past_defaults,
    'Weather_Risk_Index': weather_risk,
    'Market_Stability_Score': market_stability
}])

# Process input data (same encoding as used in the training phase)
new_farmer['Education'] = new_farmer['Education'].map({'None': 0, 'Primary': 1, 'Secondary': 2, 'Tertiary': 3})
new_farmer['Crop_Type'] = new_farmer['Crop_Type'].map({'Maize': 0, 'Tomato': 1, 'Soybean': 2, 'Onion': 3, 'Wheat': 4})
new_farmer['Region'] = new_farmer['Region'].map({'West Zone': 0, 'East Zone': 1, 'North Zone': 2, 'South Zone': 3})
new_farmer['Loan_Approved'] = new_farmer['Loan_Approved'].map({'Yes': 1, 'No': 0})

# Ensure columns match the model's expected features
missing_cols = set(X.columns) - set(new_farmer.columns)
for col in missing_cols:
    new_farmer[col] = 0

new_farmer = new_farmer[X.columns]

# Predict repayment likelihood
predicted = model.predict(new_farmer)

# Display prediction result
if predicted[0] == 1:
    st.success("Loan Repayment Prediction: **Yes**, the farmer is likely to repay the loan on time.")
else:
    st.error("Loan Repayment Prediction: **No**, the farmer is not likely to repay the loan on time.")
