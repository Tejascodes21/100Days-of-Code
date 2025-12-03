import streamlit as st
st.set_page_config(page_title="Flight Delay Prediction", page_icon="✈️", layout="centered")

import pandas as pd
import joblib
import os


@st.cache_resource
def load_model():
    model_path = "trained_model.pkl"
    if not os.path.exists(model_path):
        st.error("trained_model.pkl not found. Please copy it from Day30_ModelDeployment.")
        return None
    model = joblib.load(model_path)
    return model

model = load_model()


st.title("✈️ Flight Delay Prediction App")
st.write("""
This app uses a trained Machine Learning model to predict whether a flight  
is **likely to be delayed** based on airline, origin and weather conditions.  
""")

st.markdown("---")

with st.sidebar:
    st.header("About")
    st.write("**Model:** Random Forest (Scikit-Learn Pipeline)")
    st.write("**Features used:**")
    st.markdown("""
        - Airline  
        - Origin Airport  
        - Temperature  
        - Humidity  
        - Wind Speed  
        - Visibility  
        - Precipitation  
    """)
    st.caption("Built as part of Day 32 of #100DaysOfCode")

st.subheader("Enter Flight & Weather Details")

airlines = ["Indigo", "SpiceJet", "AirIndia", "Vistara"]
origins = ["DEL", "BOM", "BLR"]

col1, col2 = st.columns(2)

with col1:
    airline = st.selectbox("Airline", airlines)
    origin = st.selectbox("Origin Airport", origins)
    temperature = st.number_input("Temperature (°C)", min_value=-10, max_value=50, value=22)

with col2:
    humidity = st.slider("Humidity (%)", min_value=0, max_value=100, value=70)
    wind_speed = st.number_input("Wind Speed (km/h)", min_value=0, max_value=100, value=12)
    visibility = st.slider("Visibility (km)", min_value=0, max_value=10, value=3)
    precipitation = st.number_input("Precipitation (mm)", min_value=0, max_value=50, value=4)

st.markdown("")

predict_btn = st.button("Predict Delay ")

if predict_btn:
    if model is None:
        st.error("Model not loaded. Please ensure trained_model.pkl is present.")
    else:
        input_data = pd.DataFrame(
            {
                "airline": [airline],
                "origin": [origin],
                "temperature": [temperature],
                "humidity": [humidity],
                "wind_speed": [wind_speed],
                "visibility": [visibility],
                "precipitation": [precipitation],
            }
        )

        prediction = model.predict(input_data)[0]

        proba_text = ""
        try:
            proba = model.predict_proba(input_data)[0][1]
            proba_text = f" (Delay probability: **{proba:.2f}**)"
        except Exception:
            pass

        st.markdown("---")
        if prediction == 1:
            st.error(f"The model predicts this flight is LIKELY TO BE DELAYED.{proba_text}")
        else:
            st.success(f"The model predicts this flight will LIKELY BE ON-TIME.{proba_text}")
