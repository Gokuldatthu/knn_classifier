import json
import os

import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Student Performance Classifier", layout="wide")

BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "models", "knn_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "models", "scaler.pkl")
FEATURES_PATH = os.path.join(BASE_DIR, "models", "features.pkl")
DATA_PATH = os.path.join(BASE_DIR, "data", "student_performance_dataset.csv")


@st.cache_resource
def load_artifacts():
    if not os.path.exists(MODEL_PATH):
        st.error(f"Model file '{MODEL_PATH}' not found.")
        st.stop()

    model = joblib.load(MODEL_PATH)

    if not os.path.exists(SCALER_PATH):
        st.error(f"Scaler file '{SCALER_PATH}' not found.")
        st.stop()

    scaler = joblib.load(SCALER_PATH)

    if not os.path.exists(FEATURES_PATH):
        st.error(f"Features file '{FEATURES_PATH}' not found.")
        st.stop()

    features = joblib.load(FEATURES_PATH)

    return model, scaler, features


@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)


model, scaler, features = load_artifacts()
df = load_data()

st.title("Student Performance Classifier")
st.write("Predict performance category (Low / Medium / High) using a K-Nearest Neighbors classifier.")

col1, col2 = st.columns(2)

input_data = {}

with col1:
    input_data["age"] = st.slider("Age", 15, 30, 20)
    input_data["gender"] = st.selectbox("Gender", ["Male", "Female"])
    input_data["city_type"] = st.selectbox("City Type", ["Urban", "Semi-Urban", "Rural"])
    input_data["study_hours_per_day"] = st.slider("Study Hours Per Day", 0.0, 10.0, 5.0, 0.1)
    input_data["sleep_hours"] = st.slider("Sleep Hours Per Night", 3.0, 12.0, 7.0, 0.1)
    input_data["stress_level"] = st.slider("Stress Level (1-10)", 1, 10, 5)

with col2:
    input_data["motivation_level"] = st.slider("Motivation Level (1-10)", 1, 10, 6)
    input_data["focus_score"] = st.slider("Focus Score (1-10)", 1, 10, 6)
    input_data["attendance_percentage"] = st.slider("Attendance (%)", 0, 100, 75)
    input_data["assignment_completion_rate"] = st.slider("Assignment Completion (%)", 0, 100, 80)
    input_data["procrastination_index"] = st.slider("Procrastination Index (1-10)", 1, 10, 5)
    input_data["final_exam_score"] = st.slider("Final Exam Score", 0, 100, 50)

input_df = pd.DataFrame([input_data])

# Ensure all features are present, set missing ones to 0
for feature in features:
    if feature not in input_df.columns:
        input_df[feature] = 0

# Reorder columns to match feature order
input_df = input_df[features]

# Scale the input
input_scaled = scaler.transform(input_df)

if st.button("Predict Performance Category", use_container_width=True):
    prediction = model.predict(input_scaled)[0]

    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(input_scaled)[0]
        prob_df = pd.DataFrame(
            {
                "Class": model.classes_,
                "Probability": [float(p) for p in probs],
            }
        ).sort_values("Probability", ascending=False)
    else:
        prob_df = pd.DataFrame({"Class": model.classes_, "Probability": [0.0] * len(model.classes_)})

    st.subheader("Prediction")
    st.success(f"Predicted category: {prediction}")

    st.subheader("Class probabilities")
    st.dataframe(prob_df, use_container_width=True)

st.sidebar.header("Model info")
st.sidebar.write(f"Model type: K-Nearest Neighbors")
st.sidebar.write(f"Number of neighbors: {model.n_neighbors}")
st.sidebar.write(f"Number of features: {len(features)}")
st.sidebar.write("Target classes: Low, Medium, High")

if st.sidebar.checkbox("Show raw dataset preview"):
    st.dataframe(load_data().head(20), use_container_width=True)
