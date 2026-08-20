import streamlit as st
import joblib
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model = joblib.load(os.path.join(BASE_DIR, "attention_span_model.pkl"))
scaler = joblib.load(os.path.join(BASE_DIR, "attention_span_scaler.pkl"))

st.title("Attention Span Predictor")

age                  = st.number_input("Age", 1, 100, 40)
reels_watch_time     = st.number_input("Daily Reels Time (hours)", 0.0, 24.0, 1.0)
daily_screen_time    = st.number_input("Daily Screen Time (hours)", 0.0, 24.0, 2.0)
sleep_hours          = st.number_input("Average Sleep Hours per Night", 0.0, 24.0, 7.0)
focus_level          = st.selectbox("Focus Level", ["Low", "Medium", "High"])
task_completion_rate = st.number_input("Task Completion Rate (%)", 0, 100, 80)
stress_level         = st.selectbox("Stress Level", ["Low", "Medium", "High"])
platform             = st.selectbox("Platform Used", ["Instagram Reels", "TikTok", "YouTube Shorts"])

focus_map    = {"Low": 0, "Medium": 1, "High": 2}
stress_map   = {"Low": 2, "Medium": 1, "High": 0}
platform_map = {"Instagram Reels": 0, "TikTok": 1, "YouTube Shorts": 2}

if st.button("Predict Attention Span"):
    
    # Encode categorical variables
    focus_encoded = focus_map[focus_level]
    stress_encoded = stress_map[stress_level]
    platform_encoded = platform_map[platform]

    # Create input array
    input_array = [[age, reels_watch_time, daily_screen_time, sleep_hours, focus_encoded, task_completion_rate, stress_encoded, platform_encoded]]

    # Scale the input
    input_scaled = scaler.transform(input_array)

    # Make prediction
    prediction = model.predict(input_scaled)

    st.write(f"Predicted Attention Span: {prediction[0]:.2f} minutes")