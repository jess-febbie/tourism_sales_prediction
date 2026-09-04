import os
import streamlit as st
import pandas as pd
import joblib

# Load the model committed by the pipeline (sits next to this file)
model_path = os.path.join(os.path.dirname(__file__), "best_tourism_sales_prediction_model_v1.joblib")
model = joblib.load(model_path)

st.title("Tourism Sales Prediction App")
st.write("""
This application predicts whether a customer will purchase the Wellness Tourism Package.
Enter customer details below to get a prediction.
""")

# Input fields for the model features
# Customer Details
Age = st.number_input("Age", 18, 90, 30)
TypeofContact = st.selectbox("Type of Contact", ["Self Inquiry", "Company Invited", "Unknown"])
CityTier = st.selectbox("City Tier", [1, 2, 3])
Occupation = st.selectbox("Occupation", ["Salaried", "Small Business", "Large Business", "Free Lancer", "Unemployed"])
Gender = st.selectbox("Gender", ["Male", "Female", "Other"])
NumberOfPersonVisiting = st.number_input("Number of Persons Visiting", 0, 10, 1)
PreferredPropertyStar = st.selectbox("Preferred Property Star", [3, 4, 5])
MaritalStatus = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
NumberOfTrips = st.number_input("Number of Trips Annually", 0, 50, 5)
Passport_str = st.selectbox("Has Passport?", ["Yes", "No"])
Passport = 1 if Passport_str == "Yes" else 0
OwnCar_str = st.selectbox("Owns Car?", ["Yes", "No"])
OwnCar = 1 if OwnCar_str == "Yes" else 0
NumberOfChildrenVisiting = st.number_input("Number of Children Visiting", 0, 5, 0)
Designation = st.selectbox("Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP", "Officer"])
MonthlyIncome = st.number_input("Monthly Income", 0.0, 1000000.0, 50000.0, 1000.0)

# Customer Interaction Data
PitchSatisfactionScore = st.selectbox("Pitch Satisfaction Score (1-5)", [1, 2, 3, 4, 5])
ProductPitched = st.selectbox("Product Pitched", ["Basic", "Standard", "Deluxe", "Super Deluxe", "King"])
NumberOfFollowups = st.number_input("Number of Follow-ups", 0, 20, 3)
DurationOfPitch = st.number_input("Duration of Pitch (minutes)", 0.0, 120.0, 10.0, 0.5)

input_data = pd.DataFrame([{
    "Age": Age,
    "TypeofContact": TypeofContact,
    "CityTier": CityTier,
    "DurationOfPitch": DurationOfPitch,
    "Occupation": Occupation,
    "Gender": Gender,
    "NumberOfPersonVisiting": NumberOfPersonVisiting,
    "NumberOfFollowups": NumberOfFollowups,
    "ProductPitched": ProductPitched,
    "PreferredPropertyStar": PreferredPropertyStar,
    "MaritalStatus": MaritalStatus,
    "NumberOfTrips": NumberOfTrips,
    "Passport": Passport,
    "PitchSatisfactionScore": PitchSatisfactionScore,
    "OwnCar": OwnCar,
    "NumberOfChildrenVisiting": NumberOfChildrenVisiting,
    "Designation": Designation,
    "MonthlyIncome": MonthlyIncome,
}])

if st.button("Predict Purchase"):  
    # The model expects feature names to be consistent with training
    prediction = model.predict(input_data)[0]
    result = "Customer will purchase the package!" if prediction == 1 else "Customer will NOT purchase the package."
    st.subheader("Prediction Result:")
    if prediction == 1:
        st.success(f"The model predicts: **{result}**")
    else:
        st.info(f"The model predicts: **{result}**")
