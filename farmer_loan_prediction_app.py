import streamlit as st
import pandas as pd
import pickle

# Load the pre-trained model
model = pickle.load(open('loan_repayment_model.pkl', 'rb'))

# Streamlit UI
st.title('Farmer Loan Repayment Prediction')
st.write("Please enter the details of the new farmer to predict loan repayment likelihood.")

# Input fields for the farmer data
age = st.number_input('Age', min_value=18, max_value=100, value=26)
education = st.selectbox('Education Level', ['None', 'Primary', 'Secondary', 'Tertiary'])
experience_years = st.number_input('Years of Experience', min_value=0, max_value=50, value=3)
crop_type = st.selectbox('Crop Type', ['Maize', 'Tomato', 'Soybean', 'Onion', 'Wheat'])
region = st.selectbox('Region', ['West Zone', 'East Zone', 'North Zone', 'South Zone'])
land_size = st.number_input('Land Size (Acres)', min_value=0.1, max_value=100.0, value=2.5)
annual_revenue = st.number_input('Annual Revenue (USD)', min_value=0, max_value=100000, value=5500)
loan_amount = st.number_input('Loan Amount (USD)', min_value=0, max_value=50000, value=2500)
loan_approved = st.selectbox('Loan Approved', ['Yes', 'No'])
past_defaults = st.selectbox('Past Defaults', [0, 1])
weather_risk = st.slider('Weather Risk Index', 0.0, 1.0, 0.2)
market_stability = st.slider('Market Stability Score', 0.0, 1.0, 0.75)

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

# Process input data (encode categorical variables)
new_encoded = pd.get_dummies(new_farmer)

# Load the original training data to get column names for encoding
# Assuming you have the original dataframe df
df = pd.DataFrame(data)  # Replace with the actual dataset you used to train the model
X = df.drop(['Farmer_ID', 'Repaid_On_Time'], axis=1)  # Adjust based on actual features

# Ensure all columns are present in new_encoded
missing_cols = set(X.columns) - set(new_encoded.columns)
for col in missing_cols:
    new_encoded[col] = 0  # Add missing columns with value 0

# Ensure column order matches the training set
new_encoded = new_encoded[X.columns]

# Predict repayment likelihood
predicted = model.predict(new_encoded)

# Display prediction result
if predicted[0] == 1:
    st.success("Loan Repayment Prediction: **Yes**, the farmer is likely to repay the loan on time.")
else:
    st.error("Loan Repayment Prediction: **No**, the farmer is not likely to repay the loan on time.")
