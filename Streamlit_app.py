import streamlit as st
import joblib
import requests
import numpy as np
from io import BytesIO

# Function to load the model from GitHub
@st.cache_resource
def load_model_from_github():
    url = "https://github.com/josephboban2000/AIML_JOSEPH_INFY/raw/main/adaboost_model.joblib"
    response = requests.get(url)
    if response.status_code == 200:
        model = joblib.load(BytesIO(response.content))
        return model
    else:
        st.error("Error: Unable to load the model from GitHub.")
        return None

# Load the model
model = load_model_from_github()

# Descriptive statistics for rotary slider ranges
variables = {
    "perimeter1": {"min": 43.79, "max": 188.5},
    "smoothness1": {"min": 0.05263, "max": 0.1634},
    "symmetry1": {"min": 0.106, "max": 0.304},
    "fractal_dimension1": {"min": 0.04996, "max": 0.09744},
    "texture2": {"min": 0.3602, "max": 4.885},
    "perimeter2": {"min": 0.757, "max": 21.98},
    "smoothness2": {"min": 0.001713, "max": 0.03113},
    "symmetry2": {"min": 0.007882, "max": 0.07895},
    "compactness3": {"min": 0.02729, "max": 1.058},
    "symmetry3": {"min": 0.1565, "max": 0.6638},
}

# Function to set background color
def set_background(color):
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-color: {color};
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )

# Set initial black background
set_background("#000000")  # Black background

# App title
st.title("Breast Cancer Diagnosis Predictor with Rotary Sliders")

st.write(
    "Use the rotary sliders below to input normalized values for each variable. "
    "The range is normalized to [0, 1.5 × max] based on the descriptive statistics provided."
)

# Input feature sliders
inputs = {}
for var, stats in variables.items():
    max_val = stats["max"] * 1.5  # 1.5 times the maximum value
    min_val = stats["min"]
    inputs[var] = st.slider(
        label=f"{var} (Normalized)",
        min_value=0.0,
        max_value=max_val,
        value=min_val,
        step=max_val / 100,  # Adjust precision as needed
        format="%.4f",
    )

# Prepare input for prediction
if model:
    input_features = np.array(list(inputs.values())).reshape(1, -1)

    # Prediction button
    if st.button("Predict"):
        prediction = model.predict(input_features)[0]
        if prediction == 0:
            set_background("#013220")  # Dark green for benign
            st.success("The prediction is: Benign (0)")
        else:
            set_background("#8B0000")  # Dark red for malignant
            st.error("The prediction is: Malignant (1)")
else:
    st.error("Model not loaded. Please check the GitHub URL.")
