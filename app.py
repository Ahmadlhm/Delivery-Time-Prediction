# streamlit_app/app.py
import streamlit as st
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import OrdinalEncoder


traffic_categories = [['Low', 'Medium', 'High']]  

trafic_encoder = OrdinalEncoder(categories=traffic_categories)



# Load the trained model and encoders
model = joblib.load('best_tree_xgboost.pkl')    
le_weather = joblib.load('le_weather_final_project.pkl')
le_vehicle = joblib.load('le_vehicle_final_project.pkl')
#trafic_encoder = joblib.load('ord_final_project.pkl')


st.title("Delivery Time Prediction")
st.write("Enter the details to predict the delivery time.")
# Input fields
distance = st.number_input("Distance (km)", min_value=0.0, step=0.1)
prep_time = st.number_input("Preparation Time (min)", min_value=0.0, step=1.0)
experience = st.number_input("Courier Experience (years)", min_value=0.0, step=0.1)
traffic = st.selectbox("Traffic Condition", ["Low", "Medium", "High"])      
weather = st.selectbox("Weather Condition", ["Clear", "Rainy", "Windy", "Foggy", "Snowy"])
vehicle = st.selectbox("Vehicle Type", ["Bike", "Car", "Scooter"])
time_of_day = st.selectbox("Time of Day", ["Morning", "Afternoon", "Evening", "Night"])

# Encode categorical inputs
traffic_encoded = trafic_encoder.fit_transform([[traffic]])[0][0]
weather_encoded = le_weather.transform([weather])[0]    
vehicle_encoded = le_vehicle.transform([vehicle])[0]
df_time = pd.DataFrame({'Time_of_Day': [time_of_day]})

time_mapping = {
    'Morning': 0,
    'Afternoon': 1,
    'Evening': 2,
    'Night': 3
}
df_time['Time_numeric'] = df_time['Time_of_Day'].map(time_mapping)
max_val = 4
df_time['Time_sin'] = np.sin(2 * np.pi * df_time['Time_numeric'] / max_val)
df_time['Time_cos'] = np.cos(2 * np.pi * df_time['Time_numeric'] / max_val)
time_sin = df_time['Time_sin'].iloc[0]
time_cos = df_time['Time_cos'].iloc[0]

# Prepare the input DataFrame
input_data = pd.DataFrame({
    'Distance_km': [distance],
    'Preparation_Time_min': [prep_time],
    'Courier_Experience_yrs': [experience],
    'Traffic_encoded': [traffic_encoded],
    'Time_sin': [time_sin],
    'Time_cos': [time_cos],
    'Weather': [weather_encoded],
    'Vehicle_Type': [vehicle_encoded]
    
})
# Predict button
if st.button("Predict Delivery Time"):
    prediction = model.predict(input_data)
    st.success(f"Predicted Delivery Time: {prediction[0]:.2f} minutes")
