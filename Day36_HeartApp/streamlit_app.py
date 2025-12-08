import streamlit as st
import numpy as np
import joblib

# Load Model
model = joblib.load("heart_model.pkl")

# Streamlit UI
st.set_page_config(page_title="Heart Disease Prediction App ❤️", page_icon="❤️", layout="centered")

st.title("❤️ Heart Disease Risk Prediction")
st.write("Provide the patient details below to estimate heart disease risk.")

# Sidebar Info
with st.sidebar:
    st.markdown("### ℹ About the Model")
    st.info("This app uses a Logistic Regression model trained on Heart Disease Dataset.\n"
            "- Prediction: Low / High Risk\n"
            "- Probability score (0 - 1)")

# User Inputs
age = st.slider("Age", 20, 90, 45)
sex = st.selectbox("Gender", ["Male", "Female"])
cp = st.selectbox("Chest Pain Type",
                  ["Typical Angina", "Atypical Angina", "Non-Anginal", "Asymptomatic"])
trestbps = st.slider("Resting Blood Pressure (mm Hg)", 80, 200, 120)
chol = st.slider("Cholesterol (mg/dl)", 100, 600, 240)
thalach = st.slider("Max Heart Rate Achieved", 60, 220, 150)
oldpeak = st.number_input("ST Depression (Oldpeak)", 0.0, 6.5, 1.0, step=0.1)
exang = st.selectbox("Exercise-induced Angina", ["No", "Yes"])
ca = st.selectbox("Number of Major Vessels Colored by Fluoroscopy (0–3)", [0, 1, 2, 3])

# Convert input for model
gender_val = 1 if sex == "Male" else 0
angina_val = 1 if exang == "Yes" else 0
cp_val = ["Typical Angina", "Atypical Angina", "Non-Anginal", "Asymptomatic"].index(cp)

input_data = np.array([[age, gender_val, cp_val, trestbps, chol, thalach, oldpeak, angina_val, ca]])

# Predict Button
if st.button("Predict Risk"):
    prediction = model.predict(input_data)[0]
    proba = model.predict_proba(input_data)[0][1]

    st.subheader(" Prediction Result")
    st.metric(
        label="Heart Disease Risk",
        value="HIGH RISK" if prediction == 1 else "LOW RISK",
    )

    # Probability progress bar
    st.progress(float(proba))

    st.write(f"**Risk Probability:** {proba:.3f}")

    # Feedback Messages
    if prediction == 1:
        st.error("High chances of heart disease. Seek medical checkup!")
    else:
        st.success(" You are likely healthy — keep it up!")

    # Show input details
    with st.expander("Model Input Summary"):
        st.json({
            "Age": age,
            "Gender": sex,
            "Chest Pain Type": cp,
            "Resting BP": trestbps,
            "Cholesterol": chol,
            "Max HR": thalach,
            "Oldpeak": oldpeak,
            "Exercise Angina": exang,
            "Major Vessels": ca
        })

   

