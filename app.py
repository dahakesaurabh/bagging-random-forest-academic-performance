import streamlit as st
import pickle
import pandas as pd
import numpy as np

# Page configuration
st.set_page_config(page_title="Student Performance Prediction", layout="centered")

# App Header
st.title("Student Performance")
st.markdown("### By Saurabh Dahake")
st.write("Provide the student's details below to predict performance.")

# Load the trained model
@st.cache_resource
def load_model():
    with open("RF_model.pkl", "rb") as f:
        model = pickle.load(f)
    return model

model = load_model()

# Sidebar inputs for the 9 features found in the model
st.sidebar.header("Student Details Input")

def user_input_features():
    gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
    age = st.sidebar.slider("Age", 10, 30, 18)
    study_hours_per_week = st.sidebar.slider("Study Hours Per Week", 0.0, 50.0, 10.0)
    attendance_rate = st.sidebar.slider("Attendance Rate (0.0 - 1.0)", 0.0, 1.0, 0.85)
    parent_education = st.sidebar.selectbox("Parent Education Level", ["High School", "Bachelor", "Master", "PhD", "Other"])
    internet_access = st.sidebar.selectbox("Internet Access", ["Yes", "No"])
    extracurricular = st.sidebar.selectbox("Extracurricular Activities", ["Yes", "No"])
    previous_score = st.sidebar.number_input("Previous Score", 0.0, 100.0, 75.0)
    final_score = st.sidebar.number_input("Final Score", 0.0, 100.0, 80.0)

    data = {
        'gender': [gender],
        'age': [age],
        'study_hours_per_week': [study_hours_per_week],
        'attendance_rate': [attendance_rate],
        'parent_education': [parent_education],
        'internet_access': [internet_access],
        'extracurricular': [extracurricular],
        'previous_score': [previous_score],
        'final_score': [final_score]
    }
    return pd.DataFrame(data)

df = user_input_features()

# Display input summary
st.subheader("Submitted Parameters")
st.dataframe(df)

# Prediction Button
if st.button("Predict Performance"):
    try:
        prediction = model.predict(df)
        prediction_proba = model.predict_proba(df) if hasattr(model, "predict_proba") else None

        st.subheader("Prediction Result")
        st.success(f"Predicted Class: **{prediction[0]}**")
        
        if prediction_proba is not None:
            confidence = np.max(prediction_proba) * 100
            st.info(f"Confidence: {confidence:.2f}%")
            
    except Exception as e:
        st.error(f"Error during prediction: {e}")
        st.warning("Tip: Make sure categorical features match the exact encoding/format (Label Encoding vs One-Hot Encoding) used when the model was originally trained.")
                
    except Exception as e:
      st.error(f"Error during prediction: {e}")
      st.info("Note: If your model requires categorical encoding (like OneHotEncoding or LabelEncoding) that was baked into a pipeline rather than raw strings, make sure your pipeline object is what was saved inside `RF_model.pkl`.")
