
import streamlit as st
import pickle
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Load the trained Lasso model
with open('best_model (2).pkl', 'rb') as file:
    model = pickle.load(file)

# Initialize a label encoder for categorical variables
magType_encoder = LabelEncoder()
magType_encoder.fit(["ML", "MW", "MB", "MS", "MH", "MWC", "MI"])

# Function to make predictions
def predict_earthquake(features):
    # Create a full feature array of size 22 (as expected by the model)
    full_features = np.zeros(22)
    
    # Assign values to the corresponding feature indices
    full_features[1] = features[0]  # latitude
    full_features[2] = features[1]  # longitude
    full_features[3] = features[2]  # depth
    full_features[4] = features[3]  # magType_encoded
    full_features[17] = features[4] # month
    full_features[20] = features[5] # day
    
    # Reshape to fit model input and predict
    input_data = full_features.reshape(1, -1)
    prediction = model.predict(input_data)
    return prediction[0]

# Function to categorize earthquake magnitude
def categorize_magnitude(magnitude):
    if magnitude < 4.0:
        return "Low Risk"
    elif 4.0 <= magnitude <= 6.0:
        return "Moderate Risk"
    else:
        return "High Risk"

# Streamlit app layout
st.title("Earthquake Magnitude Prediction")
st.write("Advancing Earthquake Risk Reduction through Machine Learning-Enhanced Early Warning Systems")

# User inputs for selected features
st.header("Enter Earthquake Data")

latitude = st.number_input("Latitude", format="%.6f")
longitude = st.number_input("Longitude", format="%.6f")
depth = st.number_input("Depth (in km)", min_value=0.0, format="%.2f")
magType = st.selectbox("Magnitude Type", ["ML", "MW", "MB", "MS", "MH", "MWC", "MI"])
month = st.number_input("Month", min_value=1, max_value=12, step=1)
day = st.number_input("Day", min_value=1, max_value=31, step=1)

# Convert categorical feature 'magType' into numerical value using label encoder
magType_encoded = magType_encoder.transform([magType])[0]

# Collect all inputs into a list
features = [
    latitude, longitude, depth, magType_encoded, month, day
]

# Predict button
if st.button("Predict Magnitude"):
    try:
        # Make the prediction
        magnitude = predict_earthquake(features)
        category = categorize_magnitude(magnitude)

        # Display the result
        st.success(f"Predicted Magnitude: {magnitude:.2f}")
        st.info(f"Risk Category: {category}")
    except Exception as e:
        st.error(f"Error in prediction: {e}")
