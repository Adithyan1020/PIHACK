import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
import os
from twilio.rest import Client

# Load the trained model
MODEL_PATH = 'models/new/lstm_6_elderly_body.h5'
FALL_DETECTION_DIR = 'fall_detections'

# Ensure the fall detection directory exists
os.makedirs(FALL_DETECTION_DIR, exist_ok=True)

# Add custom CSS for styling
st.markdown("""
    <style>
        div.stButton > button {
            width: 100%;
        }
        .emergency-button div.stButton > button {
            background-color: #d32f2f;
            color: white;
            font-weight: bold;
            padding: 1em 2em;
        }
        .alert-box {
            padding: 20px;
            background-color: #ffebee;
            border-radius: 5px;
            text-align: center;
            margin: 10px 0;
        }
        .alert-text {
            color: #d32f2f;
            font-size: 1.2em;
            font-weight: bold;
        }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_trained_model():
    try:
        model = load_model(MODEL_PATH)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        raise

@st.cache_resource
def get_scaler():
    try:
        training_data = np.array([
            [-1.68553727835932, -0.237128818628498, 0.573137607959227, 86.91671498764, 22.9499191259498, -54.200872829371],
            [-1.59306619464705, -0.436719870601519, -0.511795403912472, -117.313150425733, 7.20236823633533, -36.5001373332926],
            [-2.86294137394329, 0.42939542832728, 0.761741996520889, -75.075533310953, 92.4100466933195, -107.913449507126],
            [-1.847590564, 0.032044435, 0.250862148, 19.53184606, -43.45835749, 11.10873745]
        ])
        scaler = StandardScaler()
        scaler.fit(training_data)
        return scaler
    except Exception as e:
        st.error(f"Error initializing scaler: {e}")
        raise

def predict_activity(model, scaler, sample):
    try:
        sample_scaled = scaler.transform([sample])
        timestamps = 66
        sequence = np.tile(sample_scaled, (timestamps, 1))
        sequence = sequence.reshape(1, timestamps, sequence.shape[1])
        pred = model.predict(sequence, verbose=0)
        activities = ['Falling', 'Running', 'Sitting', 'Stairs', 'Walking']
        predicted_index = np.argmax(pred)
        confidence = pred[0][predicted_index] * 100
        return activities[predicted_index], confidence
    except Exception as e:
        st.error(f"Error during prediction: {e}")
        raise

def display_status_indicator(status, confidence):
    """Display a colored status indicator with confidence level"""
    if status == "High Risk":
        st.markdown(f"""
            <div class="alert-box" style="background-color: #ffebee;">
                <div class="alert-text" style="color: #d32f2f;">
                    🚨 HIGH RISK FALL DETECTED!
                    MESSAGE SENT TO YOUR CARETAKER<br>
                    Confidence: {confidence:.2f}%
                </div>
            </div>
        """, unsafe_allow_html=True)
    elif status == "Warning":
        st.markdown(f"""
            <div class="alert-box" style="background-color: #fff3e0;">
                <div class="alert-text" style="color: #ef6c00;">
                    ⚠️ POTENTIAL FALL DETECTED!<br>
                    Confidence: {confidence:.2f}%
                </div>
            </div>
        """, unsafe_allow_html=True)
    elif status == "Normal":
        st.markdown(f"""
            <div class="alert-box" style="background-color: #e8f5e9;">
                <div class="alert-text" style="color: #2e7d32;">
                    ✓ Normal Activity Detected<br>
                    Confidence: {confidence:.2f}%
                </div>
            </div>
        """, unsafe_allow_html=True)

# Streamlit UI
st.title("Fall Detection System")

# Sensor data input fields
st.subheader("Sensor Data Input")
x_acc = st.number_input("X-Axis Acceleration", value=-0.172124393444624)
y_acc = st.number_input("Y-Axis Acceleration", value=-0.894497512741477)
z_acc = st.number_input("Z-Axis Acceleration", value=1.71025727103488)
x_gyr = st.number_input("X-Axis Gyroscope", value=-0.244148075807977)
y_gyr = st.number_input("Y-Axis Gyroscope", value=-83.9869380779442)
z_gyr = st.number_input("Z-Axis Gyroscope", value=83.1324198126163)

#62815	-0.500808740501113	1.05563524277474	1.81920834986419	-27.5887325663014	-95.2177495651112	117.069002349925
analyze_button = st.button("Analyze Movement", key="analyze_movement")
if analyze_button:
    try:
        # Load resources
        model = load_trained_model()
        scaler = get_scaler()

        # Get predictions
        sample = [x_acc, y_acc, z_acc, x_gyr, y_gyr, z_gyr]
        activity, confidence = predict_activity(model, scaler, sample)

        # Display results with appropriate warnings
        st.subheader("Analysis Results")
        
        if activity == 'Falling':
            account_sid = 'ACb4d7d687235b0351c7336c501136b7d2'
            auth_token = '9d896feb89b0258552a3ce5ef092215e'
            client = Client(account_sid, auth_token)

            message = client.messages.create(
                from_='+12185797555',
                body='This is an emergency',
                to='+918111838121'
            )

            print(message.sid)
            if confidence >= 75:
                display_status_indicator("High Risk", confidence)
            else:
                display_status_indicator("Warning", confidence)
        else:
            display_status_indicator("Normal", confidence)
            st.write(f"Detected Activity: {activity}")

    except Exception as e:
        st.error(f"An error occurred: {e}")
        st.error("Please check your sensor connections and try again.")
