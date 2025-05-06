import streamlit as st
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Dummy data for the farmer dataset
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

# Create DataFrame
df = pd.DataFrame(data)

# Encode categorical columns
label_encoder = LabelEncoder()
df['Education'] = label_encoder.fit_transform(df['Education'])
df['Crop_Type'] = label_encoder.fit_transform(df['Crop_Type'])
df['Region'] = label_encoder.fit_transform(df['Region'])
df['Loan_Approved'] = df['Loan_Approved'].map({'Yes': 1, 'No': 0})
df['Repaid_On_Time'] = df['Repaid_On_Time'].map({'Yes': 1, 'No': 0})

# Features and target
X = df.drop(['Farmer_ID', 'Repaid_On_Time'], axis=1)
y = df['Repaid_On_Time']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save the trained model to a .pkl file
with open('loan_repayment_model.pkl', 'wb') as f:
    pickle.dump(model, f)

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
missing_cols = set(X.columns) - set(new_encoded.columns)
for col in missing_cols:
    new_encoded[col] = 0  # Add missing columns as 0

# Ensure column order matches training set
new_encoded = new_encoded[X.columns]

# Predict repayment likelihood
predicted = model.predict(new_encoded)

# Display prediction result
if predicted[0] == 1:
    st.success("Loan Repayment Prediction: **Yes**, the farmer is likely to repay the loan on time.")
else:
    st.error("Loan Repayment Prediction: **No**, the farmer is not likely to repay the loan on time.")
